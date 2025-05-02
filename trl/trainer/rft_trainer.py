# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch.utils.data
import Pyro5.api                      # Add Pyro5 import
from Pyro5.errors import CommunicationError # Import specific error
from contextlib import contextmanager, nullcontext
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    TrainingArguments
)
from transformers.utils import is_peft_available

from ..import_utils import is_vllm_available
from .rft_config import RFTConfig
from .utils import generate_model_card, get_comet_experiment_url, pad
from trl.extras.rft_utils import printc, evaluate_state_gemini, evaluate_state_oai,split_cot, process_rewards, print_thoughts_colored, get_critic_prompt


import json
from accelerate import Accelerator
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

import torch.nn.functional as F # Ensure F is imported

from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

import statistics


def get_per_token_logps(model, input_ids, num_logits_to_keep):
    # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)



# =====================================================
# RFT Trainer Class (Only __init__ and train are shown for brevity, assuming others are unchanged)
# =====================================================
class RFTTrainer(Trainer):
    """
    Trainer for Reinforcement Fine-Tuning (RFT) with step-wise rewards.
    Performs one backward pass per sequence after accumulating step losses.
    """
    _tag_names = ["trl", "rft"]

    # --- __init__ method remains the same as in the provided code ---
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Optional[Union[str, PreTrainedModel]],
        rft_config: RFTConfig,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
        **kwargs,
    ):
        if not isinstance(rft_config, RFTConfig):
            raise TypeError("`rft_config` must be an instance of RFTConfig.")
        self.rft_config = rft_config

        model_kwargs = rft_config.model_init_kwargs or {}
        if isinstance(model, str):
             model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        self.accelerator = Accelerator(gradient_accumulation_steps=rft_config.gradient_accumulation_steps)
        self.processing_class = processing_class # Store tokenizer alias

        # --- Attach Value Head if not present ---
        # Check if value head already exists (e.g., from checkpoint)
        if not hasattr(model, 'value_head'):
            model_config = model.config
            if is_peft_available() and isinstance(model, PeftModel):
                model_config = model.base_model.model.config
            
            # Ensure model_config has hidden_size
            if not hasattr(model_config, 'hidden_size'):
                 raise AttributeError("Model config does not have 'hidden_size' attribute.")
                 
            value_head = torch.nn.Linear(model_config.hidden_size, 1)
            with torch.no_grad():
                value_head.weight.data.normal_(mean=0.0, std=1/(model_config.hidden_size + 1))
                value_head.bias.data.zero_()
            # Use setattr AFTER model preparation if possible, or ensure it's handled correctly by Accelerator
            # For simplicity here, setting it before, but this might need adjustment depending on deepspeed/FSDP
            setattr(model, 'value_head', value_head)
            print("Value head added to the model.")
        else:
            print("Value head already exists on the model.")

        self.is_peft_model = False
        if peft_config is not None:
             if not is_peft_available(): raise ImportError("PEFT not available.")
             # Get peft model *before* passing to Trainer's __init__
             model = get_peft_model(model, peft_config)
             self.is_peft_model = True
             print("PEFT model created.")

        self.remote_ref_proxy = None
        self._ref_model_internal = None

        if rft_config.evalulate_state_func is None:
            rft_config.evalulate_state_func = evaluate_state_gemini

        if self.rft_config.remote_ref_model:
            if not self.rft_config.remote_ref_model_uri: raise ValueError("`remote_ref_model_uri` missing.")
            try:
                self.remote_ref_proxy = Pyro5.api.Proxy(self.rft_config.remote_ref_model_uri); self.remote_ref_proxy._pyroTimeout = 10; self.remote_ref_proxy._pyroBind()
                print(f"Connected to remote ref model: {self.rft_config.remote_ref_model_uri}")
            except Exception as e: raise ConnectionError(f"Remote ref connection failed: {e}") from e
            if ref_model is not None: print("Warning: Using remote ref, ignoring local `ref_model`.")
        elif self.is_peft_model: print("Using PEFT base model as reference.")
        elif ref_model is not None:
            if isinstance(ref_model, str): self._ref_model_internal = AutoModelForCausalLM.from_pretrained(ref_model, **model_kwargs)
            elif isinstance(ref_model, PreTrainedModel): self._ref_model_internal = ref_model
            else: raise TypeError("`ref_model` must be str or PreTrainedModel.")
        else: print("Warning: No reference model configured. KL penalty will be zero.")

        self.generation_config = GenerationConfig(
            max_new_tokens=self.rft_config.response_length, temperature=self.rft_config.temperature + 1e-7,
            top_k=self.rft_config.top_k, top_p=self.rft_config.top_p, do_sample=True,
            repetition_penalty=self.rft_config.repetition_penalty,
            pad_token_id=processing_class.pad_token_id if processing_class.pad_token_id is not None else processing_class.eos_token_id, # Ensure pad_token_id is set
            eos_token_id=processing_class.eos_token_id,
        )
        # Add a check for pad_token_id validity
        if self.generation_config.pad_token_id is None:
             printc("Warning: pad_token_id is None. Using eos_token_id for padding during generation.", "yellow")
             self.generation_config.pad_token_id = self.generation_config.eos_token_id
        if self.generation_config.pad_token_id is None:
             raise ValueError("Cannot proceed without a valid pad_token_id or eos_token_id in the tokenizer.")


        if self.rft_config.critic_prompt_func is None: self.rft_config.critic_prompt_func = get_critic_prompt
        if self.rft_config.process_rewards_func is None: self.rft_config.process_rewards_func = process_rewards

        def data_collator(features): return features

        # Pass the correct argument name (`tokenizer`) to the parent class
        # Also pass model here, it will be prepared by Trainer.__init__
        super().__init__(
            model=model, # Pass the potentially PEFT-wrapped model
            args=rft_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class, # Use 'tokenizer' argument name
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )
        # Now self.model is the prepared model (potentially wrapped by Accelerator)
        # self.tokenizer is set by the parent Trainer

        # --- Prepare Local Reference Model ---
        self.ref_model = None
        if self._ref_model_internal is not None:
            policy_model_dtype = next(self.model.parameters()).dtype # Get dtype from prepared policy model
            if self._ref_model_internal.dtype != policy_model_dtype:
                print(f"Casting local reference model to {policy_model_dtype} before preparation.")
                self._ref_model_internal = self._ref_model_internal.to(dtype=policy_model_dtype)
            # Prepare using accelerator AFTER Trainer's __init__ has run
            self.ref_model = self.accelerator.prepare_model(self._ref_model_internal, evaluation_mode=True)
            print(f"Local reference model prepared on device: {self.ref_model.device}")

        # --- Final check on value head dtype/device after preparation ---
        # self.model is now the *prepared* model from super().__init__
        prepared_model_for_check = self.accelerator.unwrap_model(self.model)
        if hasattr(prepared_model_for_check, 'value_head'):
             value_head_param = next(iter(prepared_model_for_check.value_head.parameters()))
             value_head_dtype = value_head_param.dtype
             value_head_device = value_head_param.device
             # Get a parameter from the main body of the model to compare
             # Need to handle PEFT case where base_model holds the main params
             if self.is_peft_model and hasattr(prepared_model_for_check, 'base_model'):
                 main_model_param = next(iter(prepared_model_for_check.base_model.model.parameters()))
             else:
                 # Attempt to get a parameter, handle potential edge cases
                 try:
                     main_model_param = next(iter(p for p in prepared_model_for_check.parameters() if p.requires_grad))
                 except StopIteration:
                      printc("Warning: Could not find a main model parameter to check dtype/device against value head.", "yellow")
                      main_model_param = None # Cannot perform check

             if main_model_param is not None:
                 model_dtype = main_model_param.dtype
                 model_device = main_model_param.device # Device check might be less critical if accelerator handled placement

                 if value_head_dtype != model_dtype:
                     print(f"Warning: Value head dtype ({value_head_dtype}) differs from model body dtype ({model_dtype}). Casting value head.")
                     # Cast the value head on the *original unwrapped* model object if possible,
                     # or ensure the prepared object reflects this. This can be tricky with wrappers.
                     # Direct casting on the unwrapped object is often safer:
                     prepared_model_for_check.value_head.to(dtype=model_dtype)
                     # Re-check after casting
                     value_head_param = next(iter(prepared_model_for_check.value_head.parameters()))
                     value_head_dtype = value_head_param.dtype
                     print(f"Value head dtype after casting: {value_head_dtype}")

                 print(f"Value head checked. Final Dtype: {value_head_dtype}, Device: {value_head_device}")
             else:
                 print(f"Value head exists but couldn't verify dtype/device against main model. Dtype: {value_head_dtype}, Device: {value_head_device}")

        elif not hasattr(prepared_model_for_check, 'value_head'):
             printc("Warning: Value head was not found on the *prepared* model. Check initialization and preparation steps.", "red")


    @contextmanager
    def _optional_no_grad(self, condition: bool):
        # Helper remains the same
        if condition:
            with torch.no_grad():
                yield
        else:
            yield

    def _compute_logprobs_and_states(self, model_instance, input_ids, attention_mask, get_hidden_states: bool):
        # Helper remains the same
        if model_instance is None: raise ValueError("_compute_logprobs_and_states called with None.")
        is_ref_computation = (model_instance == self.ref_model)

        enable_grads = not is_ref_computation
        peft_adapter_manager = nullcontext()
        model_to_use = model_instance

        # When using PEFT base as ref, disable adapter temporarily
        if self.is_peft_model and self.ref_model is None and is_ref_computation:
             unwrapped_main_model = self.accelerator.unwrap_model(self.model) # Use the main prepared model
             if hasattr(unwrapped_main_model, 'disable_adapter'):
                 peft_adapter_manager = unwrapped_main_model.disable_adapter()
                 model_to_use = self.model # Use the main model instance, adapter context handles behavior
             else:
                 printc("PEFT base ref failed: Cannot disable adapter.", "red"); return None, None

        # Use the context manager for no_grad
        with self._optional_no_grad(not enable_grads), peft_adapter_manager:
            # Always unwrap the specific model instance being used (policy or ref)
            # However, if using PEFT base as ref, we already got the unwrapped main model
            if not (self.is_peft_model and self.ref_model is None and is_ref_computation):
                 unwrapped_model = self.accelerator.unwrap_model(model_to_use)
            else:
                 unwrapped_model = unwrapped_main_model # Already unwrapped above

            # Check if input tensors are on the same device as the model
            target_device = next(unwrapped_model.parameters()).device
            if input_ids.device != target_device:
                 input_ids = input_ids.to(target_device)
            if attention_mask.device != target_device:
                 attention_mask = attention_mask.to(target_device)
                 
            outputs = unwrapped_model(input_ids, attention_mask=attention_mask, output_hidden_states=get_hidden_states, return_dict=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if get_hidden_states and outputs.hidden_states else None # Get last layer

        # Ensure logits dtype matches expected float type (e.g., float32) before softmax
        if logits.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
             logits = logits.float() # Cast to float32 for stability if needed

        # Shift logits/labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()

        # Compute log probabilities
        full_log_probs = F.log_softmax(shift_logits, dim=-1)

        # Ensure the output requires grad if grads were enabled
        if enable_grads and not full_log_probs.requires_grad:
            # This case should ideally not happen if input requires grad and ops are differentiable
            printc("Warning: Logits computed but lost gradient tracking unexpectedly.", "yellow")
            # Try to re-enable grad if possible? Usually indicates an upstream issue.
            # full_log_probs = full_log_probs.clone().requires_grad_(True) # Avoid this unless debugging

        return full_log_probs, hidden_states


    def get_ref_log_probs(self, full_input_ids, full_attention_mask):
        # Helper remains the same
        if self.remote_ref_proxy is not None:
            if full_input_ids.shape[0] != 1: raise NotImplementedError("Remote ref supports batch size 1 only.")
            try:
                ids_list=full_input_ids.squeeze(0).tolist(); mask_list=full_attention_mask.squeeze(0).tolist()
                logp_list = self.remote_ref_proxy.compute_logprobs(ids_list, mask_list)
                logp_tensor = torch.tensor(logp_list, dtype=torch.float32, device=self.accelerator.device)
                return logp_tensor.unsqueeze(0)
            except Exception as e: printc(f"Remote ref failed: {e}", "red"); return None
        elif self.is_peft_model and self.ref_model is None:
            # Use the main model (which is prepared) but with adapter disabled via _compute_logprobs_and_states
            log_probs, _ = self._compute_logprobs_and_states(self.model, full_input_ids, full_attention_mask, get_hidden_states=False)
            return log_probs
        elif self.ref_model is not None:
            # Use the prepared reference model
            log_probs, _ = self._compute_logprobs_and_states(self.ref_model, full_input_ids, full_attention_mask, get_hidden_states=False)
            return log_probs
        else: return None # No reference model configured

    def get_full_kl_divergence(self, full_policy_log_probs, full_ref_log_probs, full_input_ids):
        # Helper remains the same
        if full_ref_log_probs is None or full_policy_log_probs is None: return None
        if full_input_ids.shape[1] <= 1: return None # Need at least 2 tokens for KL calculation (shifted)

        # Check shapes carefully - log probs are shifted (L-1)
        expected_len = full_input_ids.shape[1] - 1
        if full_policy_log_probs.shape[1] != expected_len or full_ref_log_probs.shape[1] != expected_len:
             printc(f"Shape mismatch KL calc: policy_logp={full_policy_log_probs.shape}, ref_logp={full_ref_log_probs.shape}, expected_len={expected_len}", "red")
             return None

        # Get actual next tokens (indices) - shape (B, L-1)
        actual_next_tokens_indices = full_input_ids[:, 1:].contiguous()

        # Ensure indices are within vocab bounds for gathering
        vocab_size = full_policy_log_probs.shape[-1]
        if actual_next_tokens_indices.max() >= vocab_size or actual_next_tokens_indices.min() < 0:
            printc(f"Invalid indices for KL gather: min={actual_next_tokens_indices.min()}, max={actual_next_tokens_indices.max()}, vocab_size={vocab_size}", "red")
            # Handle invalid indices (e.g., clamp or return None) - Returning None is safer
            return None

        # Gather the log_probs for the actual tokens generated
        # Add unsqueeze(-1) for gather dimension, then squeeze(-1)
        policy_token_log_probs = torch.gather(full_policy_log_probs, -1, actual_next_tokens_indices.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = torch.gather(full_ref_log_probs, -1, actual_next_tokens_indices.unsqueeze(-1)).squeeze(-1)

        # KL divergence: policy_logp(token) - ref_logp(token)
        # Clamp to avoid large values if needed
        kl_div = torch.clamp(policy_token_log_probs - ref_token_log_probs.detach(), min=-30, max=30) # Detach ref_log_probs as we don't backprop through ref model

        return kl_div # Shape should be (B, L-1)


    def extract_step_values(self, full_sequence_tensor, full_input_ids, step_input_ids):
        # Helper remains the same, added more verbose checks
        if full_sequence_tensor is None or step_input_ids is None or step_input_ids.shape[1] == 0:
            return None


        batch_size, seq_len = full_input_ids.shape
        step_len = step_input_ids.shape[1]

        # full_sequence_tensor is typically SHIFTED (log_probs or KL), so its length is seq_len - 1
        full_tensor_len = full_sequence_tensor.shape[1]
        if full_tensor_len != seq_len - 1:
                # This might happen if generation stopped early, but should be handled carefully
                printc(f"Extract step values: Warning - full tensor len ({full_tensor_len}) != seq_len-1 ({seq_len-1}).", "yellow")
                # Adjust seq_len based on the tensor we actually have? Or assume error?
                # Let's assume the tensor length is the source of truth for slicing
                seq_len = full_tensor_len + 1 # Implied sequence length based on tensor
                if seq_len <= step_len : # Cannot extract if total length is not greater than step length
                    printc(f"Extract step values: Cannot extract, adjusted seq_len ({seq_len}) <= step_len ({step_len}).", "red")
                    return None


        # Calculate start/end index in the SHIFTED tensor (length seq_len - 1)
        # The slice corresponds to the predictions for step_input_ids tokens
        # Example: full = [t1, t2, t3, t4, t5], step = [s1, s2] (tokens t4, t5)
        # Logprobs correspond to preds for t2, t3, t4, t5. Length 4. seq_len = 5
        # We want logprobs corresponding to step tokens s1, s2 (which are t4, t5)
        # These are at indices 2 and 3 in the logprob tensor.
        # start_idx = (seq_len - 1) - step_len = 4 - 2 = 2
        # end_idx = (seq_len - 1) = 4
        start_idx = full_tensor_len - step_len
        end_idx = full_tensor_len # Exclusive end index

        # Basic boundary checks
        if start_idx < 0 or start_idx >= full_tensor_len:
            printc(f"Extract step values: Invalid start index {start_idx} for tensor length {full_tensor_len}.", "red")
            return None
        # end_idx is naturally <= full_tensor_len

        slice_len = end_idx - start_idx
        if slice_len <= 0:
             printc(f"Extract step values: Invalid slice length {slice_len}.", "red")
             return None

        try:
            # Slice based on tensor dimension
            if len(full_sequence_tensor.shape) == 3: # (B, L-1, V) -> Log Probs Dist
                step_values = full_sequence_tensor[:, start_idx:end_idx, :]
            elif len(full_sequence_tensor.shape) == 2: # (B, L-1) -> KL Div per token
                step_values = full_sequence_tensor[:, start_idx:end_idx]
            else:
                printc(f"Extract step values: Unexpected tensor dimension {len(full_sequence_tensor.shape)}.", "red")
                return None
        except IndexError as e:
            printc(f"Extract step values: Slicing error - {e}. Indices: {start_idx}:{end_idx}, Shape: {full_sequence_tensor.shape}", "red")
            return None

        # Check if the extracted slice length matches the expected step length
        # It might be shorter if the original generation was shorter than max_length + step_length
        if step_values.shape[1] != step_len:
            # printc(f"Extract step values: Warning - Extracted slice length ({step_values.shape[1]}) != step token length ({step_len}). May happen if generation ended early.", "grey")
            # This is often acceptable, the loss calculation needs to handle potentially shorter sequences.
            pass # Allow shorter slice

        if step_values.shape[1] == 0:
             # printc("Extract step values: Resulting slice is empty.", "grey")
             return None # Return None if slice ended up empty

        return step_values


    # ==================================
    # ===== REWRITTEN TRAIN FUNCTION =====
    # ==================================
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Main training loop for RFT. With robust error handling and grad checks."""

        # --- Initial Setup ---
        device = self.accelerator.device
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        tokenizer = self.tokenizer # Use self.tokenizer set by parent Trainer
        model = self.model         # Use self.model prepared by parent Trainer

        if tokenizer is None:
            raise ValueError("Tokenizer (self.tokenizer) is not set. Check RFTTrainer initialization.")

        # Determine vocab size safely
        try:
            # Unwrap the model to get the base config, handle PEFT potentially
            unwrapped_model_for_config = self.accelerator.unwrap_model(model)
            if self.is_peft_model and hasattr(unwrapped_model_for_config, 'base_model'):
                 model_config = unwrapped_model_for_config.base_model.model.config
            else:
                 model_config = unwrapped_model_for_config.config
            model_vocab_size = model_config.vocab_size
            print(f"Using Model Vocab Size: {model_vocab_size}")
        except AttributeError as e:
            raise ValueError(f"Could not determine model vocabulary size from config: {e}")


        if optimizer is None: raise ValueError("Optimizer is None.")
        try:
            # Check if optimizer has parameters assigned to it
            if not any(len(pg['params']) > 0 for pg in optimizer.param_groups):
                 raise ValueError("Optimizer parameter list is empty or all parameter groups are empty.")
            num_param_tensors = sum(len(pg['params']) for pg in optimizer.param_groups)
            print(f"Optimizer has {num_param_tensors} parameter tensors across {len(optimizer.param_groups)} groups.")
        except Exception as e: # Catch broader issues like invalid optimizer state
             raise ValueError(f"Optimizer check failed: {e}")


        model.train() # Ensure model is in training mode
        train_dataloader = self.get_train_dataloader()
        if train_dataloader is None:
            raise ValueError("No train dataloader found. train_dataset may be None or empty.")

        # Calculate num_update_steps_per_epoch robustly (essential for scheduler and logging)
        try:
            len_dataloader = len(train_dataloader)
        except TypeError: # Handle IterableDataset case
            printc("Warning: Could not determine dataloader length (likely IterableDataset). Estimating based on args.max_steps if provided, otherwise scheduler/logging steps might be inaccurate.", "yellow")
            if self.args.max_steps > 0:
                # Estimate based on max_steps, epochs, and accumulation. This isn't perfect.
                num_train_epochs = int(self.rft_config.num_train_epochs) if self.rft_config.num_train_epochs > 0 else 1
                len_dataloader = (self.args.max_steps * self.rft_config.gradient_accumulation_steps) // num_train_epochs
                if len_dataloader == 0: len_dataloader = 1 # Avoid division by zero
                print(f"Estimated dataloader length: {len_dataloader} based on max_steps")
            else:
                # Cannot determine length, use a large placeholder or default, TQDM might not work
                len_dataloader = None # Indicate unknown length
                printc("Cannot estimate dataloader length. Progress bar and exact epoch counts may be unavailable.", "yellow")


        if len_dataloader is not None:
            num_update_steps_per_epoch = len_dataloader // self.rft_config.gradient_accumulation_steps
            num_train_epochs = int(self.rft_config.num_train_epochs)
            # Calculate max_steps based on epochs * steps_per_epoch if not provided
            if self.args.max_steps <= 0:
                max_steps = num_update_steps_per_epoch * num_train_epochs
                print(f"Calculated max_steps: {max_steps}")
            else:
                # If max_steps is provided, it overrides the calculation based on epochs
                max_steps = self.args.max_steps
                # Adjust num_train_epochs based on max_steps if necessary for accurate epoch reporting
                num_train_epochs = math.ceil(max_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 1
                print(f"Using provided max_steps: {max_steps}. Adjusted num_epochs for reporting: {num_train_epochs}")
        else:
             # Handle unknown dataloader length
             if self.args.max_steps <= 0:
                  raise ValueError("Cannot determine training steps. Provide args.max_steps when using IterableDataset without a defined length.")
             max_steps = self.args.max_steps
             num_update_steps_per_epoch = max_steps # Treat as one long epoch if length unknown
             num_train_epochs = 1 # Only one epoch in this view
             print(f"Using provided max_steps: {max_steps} with unknown dataloader length.")


        # Ensure max_steps is correctly set in Trainer state if not already done by super().__init__
        self.state.max_steps = max_steps
        num_train_optimization_steps = max_steps # Total optimizer steps

        print(f"Dataloader length (batches): {'Unknown' if len_dataloader is None else len_dataloader}")
        print(f"Num Epochs: {num_train_epochs}")
        print(f"Gradient Accumulation Steps: {self.rft_config.gradient_accumulation_steps}")
        print(f"Updates per Epoch: {'Unknown' if len_dataloader is None else num_update_steps_per_epoch}")
        print(f"Total optimization steps: {num_train_optimization_steps}")

        # --- Initialize TrainerState and TrainerControl ---
        # This is usually handled by Trainer's internal _maybe_log_save_evaluate methods etc.
        # but ensuring they exist early might help callbacks.
        if self.state is None:
            from transformers.trainer_utils import TrainerState
            self.state = TrainerState()
        if resume_from_checkpoint is None:
            self.state.global_step = 0 # Start from step 0 if not resuming
        else:
            # Add logic here if resuming checkpoint loading affects global_step
            pass

        if self.control is None:
             from transformers.trainer_utils import TrainerControl
             self.control = TrainerControl()

        # --- Initialize Callback Handler ---
        # Crucial for logging, progress bars, and other integrations.
        # Re-instantiating it or ensuring it's properly configured if it was set to None.
        if self.callback_handler is None:
             from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback, PrinterCallback, ProgressCallback
             # Get default callbacks, including ones for integrations like WandB, TensorBoard
             from transformers.integrations import get_reporting_integration_callbacks
             default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
             # Add progress callback if TQDM is enabled
             if not self.args.disable_tqdm and self.is_local_process_zero():
                 try:
                     # Try importing ProgressCallback which uses TQDM
                     from transformers.trainer_callback import ProgressCallback
                     default_callbacks.append(ProgressCallback)
                 except ImportError:
                     # Fallback to PrinterCallback if TQDM is not available/broken
                     printc("TQDM ProgressCallback not available or disabled, using PrinterCallback.", "yellow")
                     default_callbacks.append(PrinterCallback)
             elif self.is_local_process_zero():
                  # If TQDM disabled but we are main process, add PrinterCallback for basic step output
                  default_callbacks.append(PrinterCallback)


             self.callback_handler = CallbackHandler(
                 callbacks=default_callbacks + (self.callbacks), # Combine default and user callbacks
                 model=self.model,
                 tokenizer=self.tokenizer,
                 optimizer=self.optimizer,
                 lr_scheduler=self.lr_scheduler,
             )
             print("Callback handler initialized.")


        # --- Trigger Train Begin Callback ---
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # --- Training Loop ---
        global_step_for_loop = self.state.global_step # Track steps for max_steps condition

        for epoch in range(num_train_epochs):
            printc(f"Starting Epoch {epoch+1}/{num_train_epochs}", "yellow")
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            # --- Batch Loop ---
            # Use enumerate(train_dataloader) for step counting within epoch
            # If len_dataloader is None (IterableDataset), this just runs until dataset exhausted or max_steps hit
            batch_iterator = iter(train_dataloader)
            current_epoch_step = 0
            while True: # Loop until break condition (dataset end or max_steps)

                # Check max_steps condition FIRST
                if max_steps > 0 and global_step_for_loop >= max_steps:
                    printc(f"Reached max_steps ({max_steps}). Stopping training.", "yellow")
                    self.control.should_training_stop = True
                    break # Exit inner batch loop

                # Get next batch, handle end of dataset
                try:
                    batch = next(batch_iterator)
                    # Move batch to device if it's a tensor or dict of tensors (handled by default Trainer Dataloader usually)
                    # If using custom collate_fn, might need manual moving here. Assuming simple list for now.
                except StopIteration:
                    printc(f"End of dataloader reached for epoch {epoch+1}.", "blue")
                    break # Exit inner batch loop, go to next epoch

                # Increment step counter for the current epoch
                current_epoch_step += 1
                # Actual step number considering accumulation starts from 1 for logging/callbacks
                train_step_display = self.state.global_step * self.rft_config.gradient_accumulation_steps + (current_epoch_step % self.rft_config.gradient_accumulation_steps)


                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # Initialize variables used across try/except blocks for this item/batch
                prompt_completion_ids = None
                full_policy_log_probs = None
                full_hidden_states = None
                full_ref_log_probs = None
                full_per_token_kl = None
                total_loss = None
                perform_backward = False
                gradient_lost_in_loop = False
                log_total_loss = 0.0; log_policy_loss = 0.0; log_kl_loss = 0.0; log_value_loss = 0.0


                # --- Item Processing Block (using batch[0] as before) ---
                try:
                    # --- 1 & 2. Prompt Setup & Validation ---
                    if not isinstance(batch, list) or not batch:
                         printc(f"Skipping empty or invalid batch at step {train_step_display}", "grey")
                         continue
                    # Assuming batch contains list of dicts, process first item
                    batch_item = batch[0]
                    if not isinstance(batch_item, dict):
                        printc(f"Skipping invalid batch item (not a dict) at step {train_step_display}", "grey")
                        continue

                    # Safely get question and solution
                    question = batch_item.get(self.rft_config.question)
                    solution = batch_item.get(self.rft_config.answer)
                    if question is None or solution is None:
                        printc(f"Skipping batch item due to missing '{self.rft_config.question}' or '{self.rft_config.answer}' key at step {train_step_display}", "yellow")
                        continue

                    messages = [{'role': 'user', 'content': self.rft_config.system_prompt + question}]
                    try:
                         q_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                         q_text += self.rft_config.b_think
                    except Exception as e:
                         printc(f"Error applying chat template at step {train_step_display}: {e}", "red")
                         continue # Skip item if template fails

                    prompt_tokens_dict = tokenizer(
                        q_text, return_tensors="pt", padding=False, add_special_tokens=True,
                        max_length=self.rft_config.max_prompt_length, truncation=True
                    )

                    input_ids_cpu = prompt_tokens_dict["input_ids"]
                    if input_ids_cpu.numel() == 0:
                         printc(f"ERROR: Empty prompt after tokenization at step {train_step_display}!", "red"); continue
                    min_id, max_id = input_ids_cpu.min().item(), input_ids_cpu.max().item()
                    if min_id < 0 or max_id >= model_vocab_size:
                        printc(f"ERROR: Invalid initial prompt token IDs (min={min_id}, max={max_id}) at step {train_step_display}!", "red"); continue

                    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens_dict.items()}
                    prompt_length = prompt_tokens["input_ids"].size(1)


                    # --- 3. Generate full completion ---
                    try:
                        # Ensure generation config pad_token_id is valid
                        current_pad_id = self.generation_config.pad_token_id
                        if current_pad_id is None or current_pad_id < 0 or current_pad_id >= model_vocab_size:
                            safe_pad_id = tokenizer.eos_token_id
                            if safe_pad_id is None or safe_pad_id < 0 or safe_pad_id >= model_vocab_size:
                                raise ValueError(f"Cannot generate: Invalid pad_token_id ({current_pad_id}) and invalid EOS token ({tokenizer.eos_token_id})")
                            printc(f"Warning: Invalid pad_token_id ({current_pad_id}). Using EOS token ({safe_pad_id}) for padding during generation.", "yellow")
                            self.generation_config.pad_token_id = safe_pad_id

                        with torch.no_grad(): # Generation should not track gradients
                            unwrapped_model = self.accelerator.unwrap_model(model)
                            model_device = unwrapped_model.device # Get device from actual model
                            # Ensure inputs are on the model's device for generate
                            prompt_tokens_model_device = {k: v.to(model_device) for k, v in prompt_tokens.items()}

                            # *** Assign prompt_completion_ids here ***
                            prompt_completion_ids = unwrapped_model.generate(
                                **prompt_tokens_model_device,
                                generation_config=self.generation_config,
                                # Return dict is useful for debugging if needed, but sequence is primary output
                                # return_dict_in_generate=True, output_scores=True # Optional for debug
                            )

                            # Validate generated IDs
                            if prompt_completion_ids.numel() == 0:
                                printc(f"ERROR: Generation produced empty output at step {train_step_display}!", "red"); continue
                            min_gen_id, max_gen_id = prompt_completion_ids.min().item(), prompt_completion_ids.max().item()
                            if min_gen_id < 0 or max_gen_id >= model_vocab_size:
                                 printc(f"ERROR: Invalid generated token IDs (min={min_gen_id}, max={max_gen_id}) at step {train_step_display}!", "red"); continue # Skip to next item

                    except Exception as gen_e:
                        printc(f"Generation failed for item at step {train_step_display}: {gen_e}", "red")
                        continue # Go to the next item in the dataloader

                    # --- Move generated IDs to accelerator default device and create mask ---
                    prompt_completion_ids = prompt_completion_ids.to(device)
                    full_generated_attention_mask = torch.ones_like(prompt_completion_ids, device=device)
                    if prompt_completion_ids.shape[1] <= 1:
                        printc(f"Skipping item {train_step_display}: generated sequence too short (<=1 token).", "yellow"); continue


                    # --- Compute full logprobs and KL ---
                    # This block is now entered ONLY if generation succeeded and sequence is long enough
                    full_policy_log_probs, full_hidden_states = self._compute_logprobs_and_states(
                        model, prompt_completion_ids, full_generated_attention_mask, get_hidden_states=True
                    )

                    # ***** THE CRITICAL CHECK *****
                    if full_policy_log_probs is None:
                         printc(f"Policy forward pass failed (returned None) item {train_step_display}. Skipping.", "red"); continue
                    if not full_policy_log_probs.requires_grad:
                         printc(f"FATAL: full_policy_log_probs NO grad item {train_step_display}. Skipping batch item.", "red")
                         # Potentially log more context here: model state, input shapes etc.
                         continue # Skip to the next item in the batch/dataloader

                    # Compute reference log probs only if policy forward pass succeeded
                    full_ref_log_probs = self.get_ref_log_probs(prompt_completion_ids, full_generated_attention_mask)
                    # Compute KL divergence (will be None if ref log probs are None)
                    full_per_token_kl = self.get_full_kl_divergence(full_policy_log_probs, full_ref_log_probs, prompt_completion_ids)


                    # --- 4, 5, 6. Decode, Split CoT, Split Steps ---
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                    if completion_ids.shape[1] == 0:
                        printc(f"Skipping item {train_step_display}: No completion tokens generated.", "yellow"); continue
                    # Basic validation on completion IDs
                    min_comp_id, max_comp_id = completion_ids.min().item(), completion_ids.max().item()
                    if min_comp_id < 0 or max_comp_id >= model_vocab_size:
                        printc(f"ERROR: Invalid completion token IDs (min={min_comp_id}, max={max_comp_id}) at step {train_step_display}!", "red"); continue

                    try:
                        full_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    except Exception as decode_e:
                         printc(f"Error decoding completion at step {train_step_display}: {decode_e}", "red"); continue



                    delimiter = self.rft_config.e_think
                    full_text = full_text[len(self.rft_config.e_think)+1:]
                    
                    index = full_text.find(delimiter)

                    if index != -1:
                        split_result = [full_text[:index], full_text[index:]]
                    else:
                        split_result = [full_text]

                    if len(split_result) >= 2: cot, answer = split_result[0].strip(), split_result[1].strip()
                    else: cot, answer = split_result[0].strip(), self.rft_config.answer_default # Use default if no split

                    if not cot: # Handle empty CoT after stripping
                         printc(f"Skipping item {train_step_display}: Empty Chain-of-Thought after splitting.", "yellow"); continue

                    steps_text_raw = split_cot(cot, self.rft_config.delimiter)
                    if not steps_text_raw:
                         printc(f"Skipping item {train_step_display}: No steps found after splitting CoT.", "yellow"); continue


                    # --- 7. Evaluate rewards & Whiten ---
                    cumulative_reasons = [{f"thought_{i}": txt} for i, txt in enumerate(steps_text_raw)]
                    critic_prompt = self.rft_config.critic_prompt_func(question, cumulative_reasons, answer, solution)

                    try: # Wrap external evaluation call
                        eval_data = self.rft_config.evalulate_state_func(critic_prompt, cumulative_reasons)
                    except Exception as eval_e:
                        printc(f"Error during state evaluation call at step {train_step_display}: {eval_e}", "red")
                        continue # Skip item if evaluation fails

                    if eval_data is None or 'steps' not in eval_data or not eval_data['steps']:
                         printc(f"Skipping item {train_step_display}: Invalid or empty evaluation data received.", "yellow"); continue

                    try: # Wrap reward processing
                        steps_data = self.rft_config.process_rewards_func(eval_data.get("overall_score", 0), eval_data['steps'], self.rft_config)
                    except Exception as reward_proc_e:
                        printc(f"Error during reward processing call at step {train_step_display}: {reward_proc_e}", "red")
                        continue # Skip item if processing fails

                    if not steps_data:
                         printc(f"Skipping item {train_step_display}: No step data after reward processing.", "yellow"); continue

                    # --- Whiten Rewards (if enabled) ---
                    if self.rft_config.whiten_rewards and len(steps_data) > 0:
                        try:
                            all_rewards = torch.tensor([s['combined_score'] for s in steps_data if 'combined_score' in s], device=device, dtype=torch.float32)
                            if all_rewards.numel() > 1:
                                mean, std = all_rewards.mean(), all_rewards.std()
                                # Prevent division by zero or very small std dev
                                whitened = (all_rewards - mean) / (std + 1e-8) if std > 1e-5 else (all_rewards - mean)
                            elif all_rewards.numel() == 1:
                                whitened = all_rewards - all_rewards.mean() # Center single reward
                            else:
                                whitened = torch.tensor([], device=device, dtype=torch.float32) # Empty if no valid scores

                            # Add whitened scores back, checking for NaN/Inf from whitening
                            score_idx = 0
                            for i in range(len(steps_data)):
                                if 'combined_score' in steps_data[i] and score_idx < len(whitened):
                                    w_score_tensor = whitened[score_idx]
                                    if not (torch.isnan(w_score_tensor) or torch.isinf(w_score_tensor)):
                                        steps_data[i]['whitened_score'] = w_score_tensor.item()
                                    else:
                                        printc(f"Warning: NaN/Inf detected in whitened score for step {i} at train_step {train_step_display}. Skipping whitening for this step.", "yellow")
                                        # Optionally fall back to combined_score or zero?
                                        # steps_data[i]['whitened_score'] = steps_data[i]['combined_score'] # Fallback option
                                    score_idx += 1
                        except Exception as whiten_e:
                             printc(f"Error during reward whitening at step {train_step_display}: {whiten_e}", "red")
                             # Decide whether to continue without whitening or skip item
                             continue # Skip item is safer if whitening fails badly

                    # --- ✚ Add final answer as its own “step” ---
                    # You can choose what reward to assign your answer; here we reuse overall_score
                    avg_score = eval_data.get("overall_score", 0)
                    # if avg_score > 0.1 and avg_score < 1.0:
                    #     avg_score = avg_score / 100
                    if avg_score > 10 and avg_score < 100:
                        avg_score = avg_score / 100
       
                    steps_data.append({
                        "txt": answer,
                        "combined_score": avg_score,
                        # if you whiten, you'll pick up 'whitened_score' automatically below
                    })
                    # --- 8. Accumulate Losses Over Steps ---
                    all_policy_loss_terms, all_kl_loss_terms, all_value_losses = [], [], []
                    total_steps_len_tokens = 0
                    gradient_lost_in_loop = False # Reset flag for this item
                    
                    print(q_text)
                    print_thoughts_colored(steps_data)
                    
                    for step_idx, step_info in enumerate(steps_data):
                        if gradient_lost_in_loop:
                            printc(f"Gradient lost in previous step, skipping remaining steps for item {train_step_display}", "yellow")
                            break # Stop processing steps for this item if gradient was lost

                        try:
                            # --- Determine reward ---
                            reward_key = 'whitened_score' if self.rft_config.whiten_rewards and 'whitened_score' in step_info else 'combined_score'
                            if reward_key not in step_info:
                                printc(f"Skipping step {step_idx} (missing reward key '{reward_key}') at train_step {train_step_display}", "green")
                                continue
                            step_reward = step_info[reward_key]
                            # Validate reward value
                            if torch.isnan(torch.tensor(step_reward)) or torch.isinf(torch.tensor(step_reward)):
                                printc(f"Warning: NaN/Inf reward found for step {step_idx} at train_step {train_step_display}. Using 0.0.", "yellow")
                                step_reward = 0.0
                            step_reward_tensor = torch.tensor(step_reward, device=device, dtype=torch.float32)

                            step_text = step_info.get("txt")
                            if not step_text:
                                 printc(f"Skipping step {step_idx} (empty text) at train_step {train_step_display}", "yellow")
                                 continue
                            step_tokenized = tokenizer(step_text, return_tensors="pt", padding=False, add_special_tokens=False)
                            step_input_ids = step_tokenized['input_ids'].to(device)

                            if step_input_ids.shape[1] == 0:
                                printc(f"Skipping step {step_idx} (empty after tokenization) at train_step {train_step_display}", "grey")
                                continue # Skip if step tokenization results in empty tensor

                            min_step_id, max_step_id = step_input_ids.min().item(), step_input_ids.max().item()
                            if min_step_id < 0 or max_step_id >= model_vocab_size:
                                printc(f"ERROR: Invalid step token IDs (min={min_step_id}, max={max_step_id}) step {step_idx}, train_step {train_step_display}!", "red"); continue # Skip step

                            # --- Extract Step Log Probs & Check Grad ---
                            step_policy_log_probs_dist = self.extract_step_values(full_policy_log_probs, prompt_completion_ids, step_input_ids)
                            if step_policy_log_probs_dist is None:
                                printc(f"Skipping step {step_idx} (failed to extract policy log probs) at train_step {train_step_display}", "yellow"); continue
                            if not step_policy_log_probs_dist.requires_grad:
                                printc(f"ERROR: step policy log_probs NO grad after extract! item {train_step_display}, step {step_idx}", "red")
                                gradient_lost_in_loop = True; break # Stop processing steps for this item

                            # --- Extract Step KL & Align ---
                            step_kl_div = self.extract_step_values(full_per_token_kl, prompt_completion_ids, step_input_ids)
                            actual_slice_len = step_policy_log_probs_dist.shape[1] # Use length from policy slice

                            if step_kl_div is None:
                                # If KL couldn't be extracted (e.g., ref model failed or KL calc failed), use zeros
                                # Ensure shape matches the policy log probs slice shape (B, slice_len)
                                step_kl_div = torch.zeros(step_policy_log_probs_dist.shape[0], actual_slice_len, device=device, dtype=torch.float32)
                                # printc(f"Using zero KL for step {step_idx} (extract failed) at train_step {train_step_display}", "grey")
                            elif step_kl_div.shape[1] != actual_slice_len:
                                # If KL slice length doesn't match policy slice length (e.g., due to padding/eos differences)
                                printc(f"Warning: KL div slice length ({step_kl_div.shape[1]}) != policy slice length ({actual_slice_len}) for step {step_idx}, train_step {train_step_display}. Padding/truncating KL.", "yellow")
                                # Pad or truncate KL slice to match policy slice length
                                diff = actual_slice_len - step_kl_div.shape[1]
                                if diff > 0: # Pad KL
                                    step_kl_div = F.pad(step_kl_div, (0, diff), value=0.0) # Pad last dimension (length)
                                elif diff < 0: # Truncate KL
                                    step_kl_div = step_kl_div[:, :actual_slice_len]
                                # Double check shape after adjustment
                                if step_kl_div.shape[1] != actual_slice_len:
                                      printc(f"ERROR: KL div alignment failed for step {step_idx}, train_step {train_step_display}. Using zero KL.", "red")
                                      step_kl_div = torch.zeros(step_policy_log_probs_dist.shape[0], actual_slice_len, device=device, dtype=torch.float32)


                            # --- Align step_input_ids length if needed ---
                            if actual_slice_len != step_input_ids.shape[1]:
                                # This happens if extract_step_values returned a shorter slice than step tokens
                                # Use the shorter length from the extracted log_probs/KL
                                # printc(f"Aligning step_input_ids length ({step_input_ids.shape[1]}) to actual slice length ({actual_slice_len}) for step {step_idx}", "grey")
                                step_input_ids = step_input_ids[:, :actual_slice_len]

                            if step_input_ids.shape[1] == 0:
                                printc(f"Skipping step {step_idx} (empty after alignment) at train_step {train_step_display}", "grey")
                                continue # Skip if alignment results in empty IDs

                            # --- Gather Log Probs for Actual Step Tokens & Check Grad ---
                            step_indices_for_gather = step_input_ids.unsqueeze(-1)
                            # Validate indices before gather
                            min_gather_idx, max_gather_idx = step_indices_for_gather.min().item(), step_indices_for_gather.max().item()
                            log_probs_vocab_size = step_policy_log_probs_dist.shape[-1]
                            if min_gather_idx < 0 or max_gather_idx >= log_probs_vocab_size:
                                printc(f"ERROR: Invalid indices for gather (min={min_gather_idx}, max={max_gather_idx}, vocab={log_probs_vocab_size}) step {step_idx}, train_step {train_step_display}!", "red"); continue

                            policy_log_probs_for_step_tokens = torch.gather(step_policy_log_probs_dist, -1, step_indices_for_gather).squeeze(-1)
                            if not policy_log_probs_for_step_tokens.requires_grad:
                                printc(f"ERROR: step policy log_probs NO grad after gather! item {train_step_display}, step {step_idx}", "red")
                                gradient_lost_in_loop = True; break # Stop processing steps

                            # --- Calculate Step Losses ---
                            # Policy Loss (Reinforce-style: -log_prob * reward)
                            policy_loss_term_unreduced = -policy_log_probs_for_step_tokens * step_reward_tensor.unsqueeze(-1)

                            # KL Loss (KL * beta)
                            kl_loss_term_unreduced = step_kl_div * self.rft_config.beta # step_kl_div should be (B, slice_len)

                            # Value Loss (Currently disabled/zero)
                            # TODO: Implement value head prediction and loss if needed
                            step_value_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                            # Check final loss terms for grad before appending
                            if not policy_loss_term_unreduced.requires_grad:
                                printc(f"ERROR: step policy loss term NO grad! item {train_step_display}, step {step_idx}", "red")
                                gradient_lost_in_loop = True; break

                            # Check for NaN/Inf in loss terms before appending
                            if torch.isnan(policy_loss_term_unreduced).any() or torch.isinf(policy_loss_term_unreduced).any():
                                printc(f"Warning: NaN/Inf detected in policy loss term for step {step_idx} at train_step {train_step_display}. Skipping step.", "yellow")
                                continue # Skip this step's contribution
                            if torch.isnan(kl_loss_term_unreduced).any() or torch.isinf(kl_loss_term_unreduced).any():
                                printc(f"Warning: NaN/Inf detected in KL loss term for step {step_idx} at train_step {train_step_display}. Using zero KL for this step.", "yellow")
                                kl_loss_term_unreduced = torch.zeros_like(kl_loss_term_unreduced) # Zero out KL if problematic

                            # Append valid loss terms (shape: B, step_slice_len)
                            all_policy_loss_terms.append(policy_loss_term_unreduced)
                            all_kl_loss_terms.append(kl_loss_term_unreduced)
                            all_value_losses.append(step_value_loss * self.rft_config.vf_coef) # Append value loss (even if zero)
                            total_steps_len_tokens += actual_slice_len # Use actual slice len processed

                        except Exception as step_e:
                             printc(f"Error during step processing loop for item {train_step_display}, step {step_idx}: {step_e}", "red")
                             # Decide if we should break the inner loop or just skip the step
                             continue # Skip this step, try the next one


                    # --- Combine Accumulated Losses (if steps processed and no grad lost) ---
                    if total_steps_len_tokens > 0 and all_policy_loss_terms and not gradient_lost_in_loop:
                        try:
                            # Concatenate losses across steps (dim=1 is the sequence dimension)
                            total_policy_loss_terms = torch.cat(all_policy_loss_terms, dim=1) # Shape (B, total_steps_len_tokens)
                            total_kl_loss_terms = torch.cat(all_kl_loss_terms, dim=1)       # Shape (B, total_steps_len_tokens)

                            # Check gradients after concatenation (should be preserved)
                            if not total_policy_loss_terms.requires_grad:
                                printc(f"ERROR: Policy loss NO grad after cat! item {train_step_display}", "red")
                                continue # Skip backward for this item

                            # Calculate average loss across all tokens in all steps
                            avg_policy_loss = total_policy_loss_terms.mean()
                            avg_kl_loss = total_kl_loss_terms.mean()
                            # Average value loss across steps (currently zero)
                            avg_value_loss = torch.stack(all_value_losses).mean() if all_value_losses else torch.tensor(0.0, device=device)

                            # Check gradients after mean (should be preserved)
                            if not avg_policy_loss.requires_grad:
                                printc(f"ERROR: Policy loss NO grad after mean! item {train_step_display}", "red")
                                continue # Skip backward

                            # --- Final Loss Calculation & Validation ---
                            if not (torch.isnan(avg_policy_loss) or torch.isinf(avg_policy_loss) or
                                    torch.isnan(avg_kl_loss) or torch.isinf(avg_kl_loss) or
                                    torch.isnan(avg_value_loss) or torch.isinf(avg_value_loss)):

                                total_loss = avg_policy_loss + avg_kl_loss + avg_value_loss

                                # Final gradient check on the combined loss
                                if total_loss.requires_grad:
                                    perform_backward = True # Set flag to perform backward pass
                                    # Store item losses for logging (average over the item)
                                    log_policy_loss = avg_policy_loss.item()
                                    log_kl_loss = avg_kl_loss.item()
                                    log_value_loss = avg_value_loss.item()
                                    log_total_loss = total_loss.item()
                                else:
                                    # This should ideally not happen if previous checks passed
                                    printc(f"ERROR: Final total_loss NO grad! item {train_step_display}", "red")
                            else:
                                printc(f"ERROR: NaN/Inf in final average losses item {train_step_display}. Skipping backward.", "red")
                                total_loss = None # Ensure total_loss is None if invalid

                        except Exception as loss_combine_e:
                            printc(f"Error combining step losses for item {train_step_display}: {loss_combine_e}", "red")
                            total_loss = None # Ensure no backward pass if combination fails
                            perform_backward = False

                    elif gradient_lost_in_loop:
                         printc(f"Skipping backward for item {train_step_display} due to gradient lost during step processing.", "yellow")
                         total_loss = None
                         perform_backward = False
    


                # --- Catch errors from the entire item processing block ---
                except Exception as item_processing_error:
                     printc(f"Error during main processing block for item {train_step_display}: {item_processing_error}", "red")
                     # Ensure we skip backward/optimizer step for this item
                     perform_backward = False
                     total_loss = None # Prevent potential use later


                # --- Perform Backward Pass (if loss is valid and requires grad) ---
                if perform_backward and total_loss is not None:
                    # Apply loss scaling for accumulation
                    loss_to_backward = total_loss / self.rft_config.gradient_accumulation_steps
                    try:
                        # Use accelerator for backward pass
                        self.accelerator.backward(loss_to_backward)
                        # print(f"Backward pass succeeded for step {train_step_display}") # Debug print
                    except Exception as backward_e:
                         printc(f"Error during backward pass for item {train_step_display}: {backward_e}", "red")
                         # Decide if this error should prevent optimizer step. Usually yes.
                         # Maybe zero gradients here if backward failed partially?
                         # optimizer.zero_grad() # Optional: zero grads if backward fails to prevent bad update
                         perform_backward = False # Prevent optimizer step if backward failed
                # else: # Log if backward was skipped
                     # if total_loss is None and not gradient_lost_in_loop and not item_processing_error: # Avoid double logging
                     #    printc(f"Skipping backward for item {train_step_display} (no valid loss computed or grad lost).", "grey")


                # --- Gradient Update Step (Optimizer Step) ---
                # Check if we have processed enough steps for one update
                is_accumulation_step_complete = (current_epoch_step % self.rft_config.gradient_accumulation_steps) == 0
                
                # Check if we are on the last batch of the epoch (if length is known)
                is_last_batch_in_epoch = (len_dataloader is not None and current_epoch_step == len_dataloader)

                # Perform optimizer step if accumulation is complete OR it's the last batch (to avoid dropping gradients)
                # Also check if *any* backward pass was successfully performed in this accumulation cycle
                # (This requires tracking if perform_backward was true at least once since last optimizer step - more complex state needed)
                # Simpler check: perform update if accumulation complete or last batch, assuming backward *might* have happened.
                # The gradient clipping and optimizer step should handle zero gradients correctly if no backward passes occurred.
                if is_accumulation_step_complete or is_last_batch_in_epoch:
                    # print(f"Attempting optimizer step at epoch_step {current_epoch_step}, global_step {self.state.global_step}") # Debug print
                    
                    # Gradient Clipping (Optional but recommended)
                    if self.rft_config.max_grad_norm is not None and self.rft_config.max_grad_norm > 0:
                        try:
                            # Accelerator handles unscaling automatically if needed before clipping
                            self.accelerator.clip_grad_norm_(model.parameters(), self.rft_config.max_grad_norm)
                        except Exception as clip_e:
                             printc(f"Error during gradient clipping at global step {self.state.global_step}: {clip_e}", "red")
                             # Continue without clipping? Or stop? Depends on severity.

                    # Optimizer Step
                    try:
                        optimizer.step()
                        # print(f"Optimizer step succeeded at global step {self.state.global_step}") # Debug print
                    except Exception as optim_e:
                         printc(f"Error during optimizer.step() at global step {self.state.global_step}: {optim_e}", "red")
                         # Consider stopping training or just logging the error

                    # Learning Rate Scheduler Step (if applicable)
                    if lr_scheduler is not None:
                        try:
                            # Use accelerator.main_process_first to ensure scheduler step happens correctly in distributed settings
                            with self.accelerator.main_process_first():
                                 lr_scheduler.step()
                        except Exception as sched_e:
                             printc(f"Error during lr_scheduler.step() at global step {self.state.global_step}: {sched_e}", "red")

                    # Zero Gradients
                    optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings

                    # Increment global step counter AFTER optimizer step
                    self.state.global_step += 1
                    global_step_for_loop += 1 # Update loop counter as well

                    # --- Logging ---
                    # Log based on global_step and logging_steps configuration
                    if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
                         # Gather logs. Use the losses from the *last* item processed in the accumulation cycle
                         # More sophisticated logging would average losses over the accumulation cycle.
                         logs = {
                             "train/loss": log_total_loss, # Loss of the last contributing item
                             "train/policy_loss": log_policy_loss,
                             "train/kl_loss": log_kl_loss,
                             "train/value_loss": log_value_loss, # Currently zero
                             "step": self.state.global_step, # Current global step
                             # Calculate epoch float value carefully
                             "epoch": round(epoch + (current_epoch_step / len_dataloader if len_dataloader else 0.0), 2)
                         }
                         if lr_scheduler is not None:
                             try:
                                 lr_list = lr_scheduler.get_last_lr()
                                 logs["train/learning_rate"] = lr_list[0] if lr_list else 0.0
                             except Exception:
                                  logs["train/learning_rate"] = "Error"
                         else:
                              # Log optimizer's LR if no scheduler
                              try:
                                   logs["train/learning_rate"] = optimizer.param_groups[0]['lr']
                              except Exception:
                                   logs["train/learning_rate"] = "N/A"

                         # Use self.log() which handles accelerator processes and integrations (like WandB)
                         self.log(logs)


                # --- Trigger Step End Callback ---
                self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                # --- Check for Training Stop Signal ---
                if self.control.should_training_stop:
                    printc("Training stopping signal received from callbacks or max_steps.", "yellow")
                    break # Exit inner batch loop


            # --- End of Epoch Handling ---
            printc(f"Epoch {epoch+1} finished. Global step: {self.state.global_step}", "blue")
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

            # Check stop signal again after epoch end callbacks
            if self.control.should_training_stop:
                break # Exit outer epoch loop


        # --- End of Training ---
        printc(f"Training finished after {self.state.global_step} global steps.", "blue")
        # Trigger Train End Callbacks
        try:
            self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        except AttributeError as e:
             # Catch specific error related to progress bar closure if it happens
             if "'NoneType' object has no attribute 'close'" in str(e):
                  printc("Info: Progress bar closure failed (likely due to non-interactive environment or callback issue). Training itself completed.", "yellow")
             else:
                  # Re-raise other unexpected attribute errors
                  printc(f"AttributeError during on_train_end: {e}", "red")
                  raise e
        except Exception as train_end_e:
             # Catch any other errors during final callbacks
             printc(f"Error during on_train_end callbacks: {train_end_e}", "yellow")


# end of file