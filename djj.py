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
from typing import Optional, Tuple, Dict
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
            printc(f"Extract step values: Warning - Extracted slice length ({step_values.shape[1]}) != step token length ({step_len}). May happen if generation ended early.", "yellow")
            # This is often acceptable, the loss calculation needs to handle potentially shorter sequences.
            pass # Allow shorter slice

        if step_values.shape[1] == 0:
             printc("Extract step values: Resulting slice is empty.", "yellow")
             return None # Return None if slice ended up empty

        return step_values

import math
import torch
import torch.nn.functional as F
# ... other imports ...
import torch.nn.functional as F
# from transformers.trainer_utils import TrainerState, TrainerControl # OLD LINE
from transformers.trainer_pt_utils import TrainerState, TrainerControl # NEW LINE
# ... rest of the imports ...
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback, PrinterCallback
from transformers.integrations import get_reporting_integration_callbacks
import traceback
import linecache

# Assume printc, split_cot are defined elsewhere
# def printc(text, color): print(text) # Placeholder
# def split_cot(cot, delimiter): return cot.split(delimiter) # Placeholder
# def print_thoughts_colored(steps_data): pass # Placeholder


class RFTTrainer: # Or whatever your class is named
    # Assume __init__ and other methods like:
    # self.accelerator, self.optimizer, self.lr_scheduler, self.tokenizer, self.model,
    # self.is_peft_model, self.args (HuggingFace TrainingArguments), self.rft_config,
    # self.generation_config, self.gen(), self._compute_logprobs_and_states(),
    # self.get_ref_log_probs(), self.get_full_kl_divergence(), self.extract_step_values()
    # are initialized and available.

    def _validate_initial_setup(self):
        """Validates critical components like tokenizer and optimizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer (self.tokenizer) is not set.")
        if self.optimizer is None:
            raise ValueError("Optimizer (self.optimizer) is None.")
        if not any(len(pg['params']) > 0 for pg in self.optimizer.param_groups):
            raise ValueError("Optimizer parameter list is empty or all parameter groups are empty.")
        
        num_param_tensors = sum(len(pg['params']) for pg in self.optimizer.param_groups)
        print(f"Optimizer has {num_param_tensors} parameter tensors across {len(self.optimizer.param_groups)} groups.")

    def _get_model_vocab_size(self):
        """Determines and returns the model's vocabulary size."""
        try:
            unwrapped_model_for_config = self.accelerator.unwrap_model(self.model)
            if self.is_peft_model and hasattr(unwrapped_model_for_config, 'base_model'):
                model_config = unwrapped_model_for_config.base_model.model.config
            else:
                model_config = unwrapped_model_for_config.config
            vocab_size = model_config.vocab_size
            print(f"Using Model Vocab Size: {vocab_size}")
            return vocab_size
        except AttributeError as e:
            raise ValueError(f"Could not determine model vocabulary size from config: {e}")

    def _calculate_dataloader_dependent_steps(self, train_dataloader):
        """
        Calculates dataloader length, update steps per epoch, total epochs, and max steps.
        Handles IterableDataset case.
        """
        try:
            len_dataloader = len(train_dataloader)
        except TypeError: # Handle IterableDataset case
            printc("Warning: Could not determine dataloader length (likely IterableDataset).", "yellow")
            len_dataloader = None

        if len_dataloader is not None:
            num_update_steps_per_epoch = len_dataloader // self.rft_config.gradient_accumulation_steps
            num_train_epochs = int(self.rft_config.num_train_epochs)
            if self.args.max_steps <= 0:
                max_steps = num_update_steps_per_epoch * num_train_epochs
                print(f"Calculated max_steps: {max_steps}")
            else:
                max_steps = self.args.max_steps
                num_train_epochs = math.ceil(max_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 1
                print(f"Using provided max_steps: {max_steps}. Adjusted num_epochs for reporting: {num_train_epochs}")
        else: # IterableDataset with no __len__
            if self.args.max_steps <= 0:
                raise ValueError("Cannot determine training steps. Provide args.max_steps when using IterableDataset without a defined length.")
            max_steps = self.args.max_steps
            num_update_steps_per_epoch = max_steps # Treat as one long epoch if length unknown
            num_train_epochs = 1
            print(f"Using provided max_steps: {max_steps} with unknown dataloader length.")
        
        if num_update_steps_per_epoch == 0 and len_dataloader is not None:
            # This can happen if dataloader is shorter than accumulation steps
            printc(f"Warning: num_update_steps_per_epoch is 0. Dataloader length ({len_dataloader}) might be less than gradient_accumulation_steps ({self.rft_config.gradient_accumulation_steps}). max_steps will be primary driver if set.", "yellow")
            if max_steps <= 0: # If max_steps wasn't overriding, this is problematic
                 raise ValueError("num_update_steps_per_epoch is 0 and max_steps not set. Training cannot proceed.")


        print(f"Dataloader length (batches): {'Unknown' if len_dataloader is None else len_dataloader}")
        print(f"Num Epochs for planning: {num_train_epochs}")
        print(f"Gradient Accumulation Steps: {self.rft_config.gradient_accumulation_steps}")
        print(f"Updates per Epoch: {'Unknown' if len_dataloader is None else num_update_steps_per_epoch}")
        print(f"Total optimization steps planned: {max_steps}")

        return len_dataloader, num_update_steps_per_epoch, num_train_epochs, max_steps

    def _initialize_trainer_internals(self, resume_from_checkpoint, max_steps):
        """Initializes TrainerState, TrainerControl, and CallbackHandler."""
        if self.state is None:
            self.state = TrainerState()
        if resume_from_checkpoint is None:
            self.state.global_step = 0
        # Add logic if resuming checkpoint loading affects global_step

        self.state.max_steps = max_steps # Ensure state knows the true max_steps

        if self.control is None:
            self.control = TrainerControl()

        if self.callback_handler is None:
            default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
            if not self.args.disable_tqdm and self.is_local_process_zero():
                try:
                    from transformers.trainer_callback import ProgressCallback
                    default_callbacks.append(ProgressCallback)
                except ImportError:
                    printc("ProgressCallback (TQDM) not available. Falling back to PrinterCallback.", "yellow")
                    default_callbacks.append(PrinterCallback)
            elif self.is_local_process_zero():
                default_callbacks.append(PrinterCallback)
            
            self.callback_handler = CallbackHandler(
                callbacks=default_callbacks + (self.callbacks or []),
                model=self.model, tokenizer=self.tokenizer, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            )
            print("Callback handler initialized.")

    def _prepare_item_prompt(self, batch_item, model_vocab_size, train_step_display):
        """Prepares and tokenizes the prompt for a single batch item."""
        question = batch_item.get(self.rft_config.question)
        solution = batch_item.get(self.rft_config.answer) # Keep solution for later, though not used in prompt directly
        if question is None or solution is None:
            printc(f"Skipping item {train_step_display}: missing '{self.rft_config.question}' or '{self.rft_config.answer}'.", "yellow")
            return None, None, None

        messages = [{'role': 'user', 'content': self.rft_config.system_prompt + question}]
        try:
            q_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            q_text += self.rft_config.b_think
        except Exception as e:
            printc(f"Error applying chat template for item {train_step_display}: {e}", "red")
            return None, None, None

        prompt_tokens_dict = self.tokenizer(
            q_text, return_tensors="pt", padding=False, add_special_tokens=True,
            max_length=self.rft_config.max_prompt_length, truncation=True
        )
        input_ids_cpu = prompt_tokens_dict["input_ids"]

        if input_ids_cpu.numel() == 0:
            printc(f"ERROR: Empty prompt after tokenization for item {train_step_display}.", "red"); return None, None, None
        min_id, max_id = input_ids_cpu.min().item(), input_ids_cpu.max().item()
        if min_id < 0 or max_id >= model_vocab_size:
            printc(f"ERROR: Invalid initial prompt token IDs (min={min_id}, max={max_id}) for item {train_step_display}.", "red"); return None, None, None

        prompt_tokens = {k: v.to(self.accelerator.device) for k, v in prompt_tokens_dict.items()}
        prompt_length = prompt_tokens["input_ids"].size(1)
        return prompt_tokens, prompt_length, question, solution # Pass question/solution for reward step

    def _generate_and_validate_completion(self, prompt_tokens, model_vocab_size, train_step_display):
        """Generates completion and validates the generated token IDs."""
        # Ensure generation config pad_token_id is valid
        current_pad_id = self.generation_config.pad_token_id
        if current_pad_id is None or current_pad_id < 0 or current_pad_id >= model_vocab_size:
            safe_pad_id = self.tokenizer.eos_token_id
            if safe_pad_id is None or safe_pad_id < 0 or safe_pad_id >= model_vocab_size:
                # This is a critical configuration error
                raise ValueError(f"Cannot generate: Invalid pad_token_id ({current_pad_id}) and invalid EOS token ({self.tokenizer.eos_token_id})")
            printc(f"Warning: Invalid pad_token_id ({current_pad_id}). Using EOS token ({safe_pad_id}) for padding during generation.", "yellow")
            self.generation_config.pad_token_id = safe_pad_id
        
        try:
            prompt_completion_ids = self.gen(self.model, prompt_tokens) # self.gen should use self.generation_config
        except Exception as gen_e:
            printc(f"Error during model generation for item {train_step_display}: {gen_e}", "red")
            return None

        if prompt_completion_ids.numel() == 0:
            printc(f"ERROR: Generation produced empty output for item {train_step_display}.", "red"); return None
        min_gen_id, max_gen_id = prompt_completion_ids.min().item(), prompt_completion_ids.max().item()
        if min_gen_id < 0 or max_gen_id >= model_vocab_size:
            printc(f"ERROR: Invalid generated token IDs (min={min_gen_id}, max={max_gen_id}) for item {train_step_display}.", "red"); return None
        
        if prompt_completion_ids.shape[1] <= 1: # prompt_length implicitly included here
            printc(f"Skipping item {train_step_display}: generated sequence too short (<=1 token including prompt).", "yellow"); return None
            
        return prompt_completion_ids.to(self.accelerator.device)

    def _compute_policy_and_reference_outputs(self, prompt_completion_ids, train_step_display):
        """Computes policy logprobs/states, reference logprobs, and KL divergence."""
        full_generated_attention_mask = torch.ones_like(prompt_completion_ids, device=prompt_completion_ids.device)
        
        full_policy_log_probs, _ = self._compute_logprobs_and_states( # Assuming _ is full_hidden_states
            self.model, prompt_completion_ids, full_generated_attention_mask, get_hidden_states=True # Or False if states not needed later
        )
        if full_policy_log_probs is None:
            printc(f"Policy forward pass failed (returned None) for item {train_step_display}. Skipping.", "red"); return None, None, None
        if not full_policy_log_probs.requires_grad:
            printc(f"FATAL: full_policy_log_probs NO grad for item {train_step_display}. Skipping.", "red"); return None, None, None

        full_ref_log_probs = self.get_ref_log_probs(prompt_completion_ids, full_generated_attention_mask)
        # full_per_token_kl can be None if ref_log_probs is None. Handle downstream.
        full_per_token_kl = self.get_full_kl_divergence(full_policy_log_probs, full_ref_log_probs, prompt_completion_ids)
        
        return full_policy_log_probs, full_ref_log_probs, full_per_token_kl

    def _decode_and_split_completion(self, prompt_completion_ids, prompt_length, model_vocab_size, train_step_display):
        """Decodes completion, splits CoT, and extracts steps."""
        completion_ids = prompt_completion_ids[:, prompt_length:]
        if completion_ids.shape[1] == 0:
            printc(f"Skipping item {train_step_display}: No completion tokens generated.", "yellow"); return None, None, None
        
        min_comp_id, max_comp_id = completion_ids.min().item(), completion_ids.max().item()
        if min_comp_id < 0 or max_comp_id >= model_vocab_size:
            printc(f"ERROR: Invalid completion token IDs (min={min_comp_id}, max={max_comp_id}) for item {train_step_display}.", "red"); return None, None, None

        try:
            full_text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception as decode_e:
            printc(f"Error decoding completion for item {train_step_display}: {decode_e}", "red"); return None, None, None
        
        # Original logic for removing b_think and e_think prefix
        # Assuming self.rft_config.b_think was already added to prompt, and e_think is expected at start of generation
        # The original code was: full_text = full_text[len(self.rft_config.e_think)+1:] which seems to skip e_think itself.
        # This needs to be consistent with how e_think is used.
        # Let's assume e_think is a delimiter like "</scratchpad>" or similar.
        if full_text.startswith(self.rft_config.e_think): # A common pattern for models outputting structured thoughts
             full_text = full_text[len(self.rft_config.e_think):].lstrip() # Remove and strip leading space
        # else:
            # printc(f"Warning: Generated text for item {train_step_display} did not start with e_think ('{self.rft_config.e_think}'). Processing as is.", "yellow")


        delimiter = self.rft_config.e_think # Re-check if this is the intended delimiter for CoT vs Answer.
                                            # The original code uses e_think twice, which might be a bug.
                                            # If e_think marks end of CoT, then use it as delimiter.
                                            # If there's another delimiter, use that.
                                            # Example: CoT ends with </คิด>, answer follows.
                                            # Let's assume self.rft_config.answer_delimiter exists for splitting CoT and answer
        answer_delimiter = getattr(self.rft_config, 'answer_delimiter', self.rft_config.e_think) # Fallback to e_think if not defined

        index = full_text.find(answer_delimiter)
        if index != -1:
            cot, answer = full_text[:index].strip(), full_text[index + len(answer_delimiter):].strip()
        else:
            cot, answer = full_text.strip(), self.rft_config.answer_default

        if not cot:
            printc(f"Skipping item {train_step_display}: Empty Chain-of-Thought after splitting.", "yellow"); return None, None, None

        steps_text_raw = split_cot(cot, self.rft_config.delimiter) # self.rft_config.delimiter for steps
        if not steps_text_raw:
            printc(f"Skipping item {train_step_display}: No steps found after splitting CoT.", "yellow"); return None, None, None
            
        return cot, answer, steps_text_raw

    def _evaluate_and_process_item_rewards(self, question, steps_text_raw, answer, solution, train_step_display):
        """Evaluates steps, processes rewards, and applies whitening if enabled."""
        cumulative_reasons = [{f"thought_{i}": txt} for i, txt in enumerate(steps_text_raw)]
        critic_prompt = self.rft_config.critic_prompt_func(question, cumulative_reasons, answer, solution)

        try:
            eval_data = self.rft_config.evalulate_state_func(critic_prompt, cumulative_reasons)
        except Exception as eval_e:
            printc(f"Error during state evaluation call for item {train_step_display}: {eval_e}", "red")
            return None
        if eval_data is None or 'steps' not in eval_data or not eval_data['steps']:
            printc(f"Skipping item {train_step_display}: Invalid or empty evaluation data.", "yellow"); return None

        try:
            steps_data = self.rft_config.process_rewards_func(eval_data.get("overall_score", 0), eval_data['steps'], self.rft_config)
        except Exception as reward_proc_e:
            printc(f"Error during reward processing for item {train_step_display}: {reward_proc_e}", "red")
            return None
        if not steps_data:
            printc(f"Skipping item {train_step_display}: No step data after reward processing.", "yellow"); return None

        # Whiten Rewards
        if self.rft_config.whiten_rewards and len(steps_data) > 0:
            try:
                all_rewards = torch.tensor([s['combined_score'] for s in steps_data if 'combined_score' in s], device=self.accelerator.device, dtype=torch.float32)
                if all_rewards.numel() > 1:
                    mean, std = all_rewards.mean(), all_rewards.std()
                    whitened = (all_rewards - mean) / (std + 1e-8) if std > 1e-5 else (all_rewards - mean)
                elif all_rewards.numel() == 1:
                    whitened = all_rewards - all_rewards.mean()
                else: # No valid scores found
                    whitened = torch.empty(0, device=self.accelerator.device, dtype=torch.float32)

                score_idx = 0
                for i in range(len(steps_data)):
                    if 'combined_score' in steps_data[i] and score_idx < len(whitened):
                        w_score_tensor = whitened[score_idx]
                        if not (torch.isnan(w_score_tensor) or torch.isinf(w_score_tensor)):
                            steps_data[i]['whitened_score'] = w_score_tensor.item()
                        else:
                            printc(f"Warning: NaN/Inf in whitened score for step {i}, item {train_step_display}. Using original score.", "yellow")
                            # Optionally, remove 'whitened_score' or keep original if problematic
                        score_idx += 1
            except Exception as whiten_e:
                printc(f"Error during reward whitening for item {train_step_display}: {whiten_e}. Proceeding without whitening.", "red")
                # Potentially remove 'whitened_score' key if it was partially added or just let it be
        
        # Add final answer as a step
        avg_score = eval_data.get("overall_score", 0)
        # Original scaling:
        if avg_score > 10 and avg_score < 100: avg_score = avg_score / 100

        final_answer_step = {"txt": answer, "combined_score": avg_score}
        # If whitening was successful and applied, the final step's reward might also need to be part of the whitening
        # or be handled separately. The original code appends it *after* whitening the CoT steps.
        # For simplicity, let's assume it gets the raw overall_score, or if you want to whiten it, it should be included *before* whitening.
        steps_data.append(final_answer_step)
        return steps_data

    def _perform_optimizer_step_and_log(self, loss_for_item, item_log_metrics, current_epoch_step, epoch, len_dataloader):
        """Handles gradient accumulation, optimizer step, LR scheduling, and logging."""
        perform_update = False
        if self.rft_config.gradient_accumulation_steps == 1:
            perform_update = True
        elif (current_epoch_step % self.rft_config.gradient_accumulation_steps) == 0:
            perform_update = True
        # Also update if it's the last batch of an epoch with known length, to not lose gradients
        elif len_dataloader is not None and current_epoch_step == len_dataloader:
            printc(f"Forcing optimizer step on last batch of epoch (step {current_epoch_step}/{len_dataloader})", "grey")
            perform_update = True
        
        # If this item contributed a loss, it has been backwarded.
        # The perform_update flag determines if we finalize the optimizer step.

        if perform_update:
            if self.rft_config.max_grad_norm is not None and self.rft_config.max_grad_norm > 0:
                try:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.rft_config.max_grad_norm)
                except Exception as clip_e:
                    printc(f"Error during gradient clipping at global step {self.state.global_step}: {clip_e}", "red")

            try:
                self.optimizer.step()
            except Exception as optim_e:
                printc(f"Error during optimizer.step() at global step {self.state.global_step}: {optim_e}", "red")
                # Potentially stop training or try to recover

            if self.lr_scheduler is not None:
                try:
                    with self.accelerator.main_process_first():
                        self.lr_scheduler.step()
                except Exception as sched_e:
                    printc(f"Error during lr_scheduler.step() at global step {self.state.global_step}: {sched_e}", "red")
            
            self.optimizer.zero_grad(set_to_none=True)
            self.state.global_step += 1 # Increment global step only after a successful optimizer step

            # Logging
            if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
                logs = {
                    "train/loss": item_log_metrics.get("total_loss", 0.0), # From last item in accumulation
                    "train/policy_loss": item_log_metrics.get("policy_loss", 0.0),
                    "train/kl_loss": item_log_metrics.get("kl_loss", 0.0),
                    "train/value_loss": item_log_metrics.get("value_loss", 0.0),
                    "step": self.state.global_step,
                    "epoch": round(epoch + (current_epoch_step / len_dataloader if len_dataloader and len_dataloader > 0 else 0.0), 2)
                }
                try:
                    lr_list = self.lr_scheduler.get_last_lr() if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                    logs["train/learning_rate"] = lr_list[0] if isinstance(lr_list, list) else lr_list
                except Exception: logs["train/learning_rate"] = "Error/NA"
                
                self.log(logs) # self.log is part of HuggingFace Trainer, handles distributed logging

    # Main Train Function Refactored

    def _generate_sequence_and_get_model_outputs(self, 
                                           model_obj, 
                                           prompt_tokens,
                                           train_step_display,
                                           is_trainable_policy_model,
                                           get_hidden_states_flag=True):
        """
        Generates a sequence using model_obj and then computes its logprobs and 
        hidden states by performing a forward pass with the same model_obj.
        """
        device = self.accelerator.device 
        model_obj_vocab_size = self._get_model_config_vocab_size(model_obj)
        tokenizer_vocab_size = self._get_tokenizer_vocab_size()

        temp_gen_config = self.generation_config 
        original_pad_token_id = temp_gen_config.pad_token_id
        pad_id_to_use = original_pad_token_id

        if pad_id_to_use is None or pad_id_to_use < 0 or pad_id_to_use >= tokenizer_vocab_size:
            safe_pad_id = self.tokenizer.eos_token_id
            if safe_pad_id is None or safe_pad_id < 0 or safe_pad_id >= tokenizer_vocab_size:
                raise ValueError(f"Cannot generate with {model_obj.__class__.__name__}: Invalid pad_token_id ({pad_id_to_use}) and invalid EOS token ({safe_pad_id}) for tokenizer vocab size {tokenizer_vocab_size}")
            printc(f"Warning ({model_obj.__class__.__name__}): Invalid pad_token_id ({pad_id_to_use}). Using EOS token ({safe_pad_id}) for padding during this generation.", "yellow")
            pad_id_to_use = safe_pad_id
            current_gen_config = self.generation_config.copy() 
            current_gen_config.pad_token_id = pad_id_to_use
        else:
            current_gen_config = self.generation_config 

        try:

            prompt_completion_ids = self.gen(model_obj, prompt_tokens, generation_config=current_gen_config)
        except Exception as gen_e:
            printc(f"Error during model generation with {model_obj.__class__.__name__} for item {train_step_display}: {gen_e}", "red")
            return None, None, None 

        if prompt_completion_ids.numel() == 0:
            printc(f"ERROR ({model_obj.__class__.__name__}): Generation produced empty output for item {train_step_display}.", "red")
            return None, None, None

        min_gen_id, max_gen_id = prompt_completion_ids.min().item(), prompt_completion_ids.max().item()
        if min_gen_id < 0 or max_gen_id >= model_obj_vocab_size:
            printc(f"ERROR ({model_obj.__class__.__name__}): Invalid generated token IDs (min={min_gen_id}, max={max_gen_id}, model_vocab={model_obj_vocab_size}) for item {train_step_display}.", "red")
            return None, None, None

        if prompt_completion_ids.shape[1] <= prompt_tokens["input_ids"].shape[1]:
             printc(f"Skipping item {train_step_display} ({model_obj.__class__.__name__}): no new tokens generated or sequence became shorter.", "yellow")
             return None, None, None

        prompt_completion_ids = prompt_completion_ids.to(device)

        full_attention_mask = torch.ones_like(prompt_completion_ids, device=device)

        try:

            if not is_trainable_policy_model and hasattr(model_obj, 'eval'): 
                model_obj.eval() 

            outputs = model_obj(
                input_ids=prompt_completion_ids, 
                attention_mask=full_attention_mask, 
                output_hidden_states=get_hidden_states_flag

            )
            logits = outputs.logits 
            log_probs = F.log_softmax(logits, dim=-1)
            hidden_states = outputs.hidden_states if get_hidden_states_flag and hasattr(outputs, 'hidden_states') else None

        except Exception as forward_e:
            printc(f"Error during forward pass (logprob computation) with {model_obj.__class__.__name__} for item {train_step_display}: {forward_e}", "red")

            return None, None, None

        if log_probs is None: 
            printc(f"Log_probs are None after forward pass for {model_obj.__class__.__name__} for item {train_step_display}.", "red")
            return None, None, None

        if is_trainable_policy_model and not log_probs.requires_grad:
            printc(f"FATAL: log_probs from TRAINABLE policy model ({model_obj.__class__.__name__}) have NO grad for item {train_step_display}. Skipping item.", "red")
            return None, None, None 

        return prompt_completion_ids, log_probs, hidden_states

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        self._validate_initial_setup()
        main_model_vocab_size = self._get_model_config_vocab_size(self.model)
        device = self.accelerator.device
        self.model.train() 

        train_dataloader = self.get_train_dataloader()
        if train_dataloader is None: raise ValueError("No train dataloader found.")

        len_dataloader, _, num_train_epochs, max_steps = \
            self._calculate_dataloader_dependent_steps(train_dataloader)

        self._initialize_trainer_internals(resume_from_checkpoint, max_steps)
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for epoch in range(num_train_epochs):
            printc(f"Starting Epoch {epoch+1}/{num_train_epochs}", "yellow")
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            batch_iterator = iter(train_dataloader)
            current_epoch_step = 0
            last_successful_item_log_metrics = {} 

            while True: 
                if self.state.global_step >= max_steps and max_steps > 0:
                    self.control.should_training_stop = True; break 
                try:
                    batch = next(batch_iterator)
                except StopIteration: break 

                current_epoch_step += 1
                train_step_display = self.state.global_step * self.rft_config.gradient_accumulation_steps + \
                                     (current_epoch_step % self.rft_config.gradient_accumulation_steps if self.rft_config.gradient_accumulation_steps > 0 else 0)
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if not isinstance(batch, list) or not batch or not isinstance(batch[0], dict):
                    printc(f"Skipping malformed batch/item at display step {train_step_display}", "grey"); continue
                batch_item = batch[0]

                item_total_loss = None
                perform_backward_for_item = False

                try: 
                    prompt_tokens, prompt_length, question, solution = self._prepare_item_prompt(batch_item, main_model_vocab_size, train_step_display)
                    if prompt_tokens is None: continue

                    policy_outputs_tuple = self._generate_sequence_and_get_model_outputs(
                        self.model,    
                        prompt_tokens,
                        train_step_display,
                        is_trainable_policy_model=True, 
                        get_hidden_states_flag=True 
                    )
                    if policy_outputs_tuple is None or policy_outputs_tuple[0] is None: continue
                    prompt_completion_ids, full_policy_log_probs, full_policy_hidden_states = policy_outputs_tuple

                    full_ref_log_probs = None
                    if self.ref_model: 

                        full_generated_attention_mask = torch.ones_like(prompt_completion_ids, device=device)
                        full_ref_log_probs = self.get_ref_log_probs(prompt_completion_ids, full_generated_attention_mask)

                    else:
                        printc(f"No reference model provided. KL divergence will be zero or based on placeholder values for item {train_step_display}.", "grey")

                    full_per_token_kl = self.get_full_kl_divergence(full_policy_log_probs, full_ref_log_probs, prompt_completion_ids)

                    decode_outputs = self._decode_and_split_completion(prompt_completion_ids, prompt_length, main_model_vocab_size, train_step_display)
                    if decode_outputs is None: continue
                    _, answer, steps_text_raw = decode_outputs

                    print(f"Q: {question}") 
                    steps_data = self._evaluate_and_process_item_rewards(question, steps_text_raw, answer, solution, train_step_display)
                    if steps_data is None: continue
                    print_thoughts_colored(steps_data)

                    item_total_loss, item_log_metrics, gradient_lost_processing_item = \
                        self._calculate_step_losses_for_item(steps_data, full_policy_log_probs, prompt_completion_ids,
                                                             full_per_token_kl, main_model_vocab_size, train_step_display)

                    if item_total_loss is not None and not gradient_lost_processing_item:
                        perform_backward_for_item = True
                    elif not gradient_lost_processing_item:
                         printc(f"No valid loss computed for item {train_step_display}, skipping backward.", "yellow")

                except Exception as item_processing_error:
                    perform_backward_for_item = False 
                    tb = item_processing_error.__traceback__
                    while tb.tb_next: tb = tb.tb_next
                    filename, lineno = tb.tb_frame.f_code.co_filename, tb.tb_lineno
                    code_line = linecache.getline(filename, lineno).strip()
                    printc(f"Unhandled error processing item {train_step_display} at {filename}:{lineno}\n  {code_line}\n  → {item_processing_error}", "red")

                    continue 

                if perform_backward_for_item and item_total_loss is not None:
                    loss_to_backward = item_total_loss / self.rft_config.gradient_accumulation_steps
                    try:
                        self.accelerator.backward(loss_to_backward)
                        last_successful_item_log_metrics = item_log_metrics 
                    except Exception as backward_e:
                        printc(f"Error during backward pass for item {train_step_display}: {backward_e}", "red")

                self._perform_optimizer_step_and_log(
                    item_total_loss if perform_backward_for_item else None, 
                    last_successful_item_log_metrics, 
                    current_epoch_step, epoch, len_dataloader
                )
                self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                if self.control.should_training_stop: break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            if self.control.should_training_stop: break

        printc(f"Training finished after {self.state.global_step} global steps.", "blue")
        try:
            self.callback_handler.on_train_end(self.args, self.state, self.control)
        except Exception as train_end_e:

            is_progress_bar_error = (
                "'NoneType' object has no attribute 'close'" in str(train_end_e) and
                hasattr(train_end_e, '__traceback__') and 
                train_end_e.__traceback__ is not None and 
                any("ProgressCallback" in frame.name for frame in traceback.extract_tb(train_end_e.__traceback__))
            )
            if is_progress_bar_error:
                 printc("Info: Progress bar closure failed (likely TQDM issue or callback misconfiguration). Training itself completed.", "yellow")
            else:
                 printc(f"Error during on_train_end callbacks: {train_end_e}", "yellow")

