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
from transformers.trainer_callback import TrainerState, TrainerControl
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback, PrinterCallback, ProgressCallback
from transformers.integrations import get_reporting_integration_callbacks
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
        self.processing_class = processing_class # Store processing_class alias

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
             raise ValueError("Cannot proceed without a valid pad_token_id or eos_token_id in the processing_class.")


        if self.rft_config.critic_prompt_func is None: self.rft_config.critic_prompt_func = get_critic_prompt
        if self.rft_config.process_rewards_func is None: self.rft_config.process_rewards_func = process_rewards

        def data_collator(features): return features

        # Pass the correct argument name (`processing_class`) to the parent class
        # Also pass model here, it will be prepared by Trainer.__init__
        super().__init__(
            model=model, # Pass the potentially PEFT-wrapped model
            args=rft_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class, # Use 'processing_class' argument name
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )
        # Now self.model is the prepared model (potentially wrapped by Accelerator)
        # self.processing_class is set by the parent Trainer

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



    def _validate_initial_setup(self):
        """Validates critical components like processing_class and optimizer."""
        if self.processing_class is None:
            raise ValueError("processing_class (self.processing_class) is not set.")
        if self.optimizer is None:
            raise ValueError("Optimizer (self.optimizer) is None.")
        if not any(len(pg['params']) > 0 for pg in self.optimizer.param_groups):
            raise ValueError("Optimizer parameter list is empty or all parameter groups are empty.")
        
        num_param_tensors = sum(len(pg['params']) for pg in self.optimizer.param_groups)
        print(f"Optimizer has {num_param_tensors} parameter tensors across {len(self.optimizer.param_groups)} groups.")


    def _get_processing_class_vocab_size(self):
        """Safely gets the vocabulary size of self.processing_class."""
        try:
            # Try common attributes for vocab size
            if hasattr(self.processing_class, 'vocab_size') and self.processing_class.vocab_size is not None:
                return self.processing_class.vocab_size
            # Fallback for some processing_classs that use len()
            return len(self.processing_class)
        except Exception as e:
            raise ValueError(f"Cannot determine processing_class vocabulary size: {e}")

    def _generate_sequence_and_get_model_outputs(
        self,
        model_obj,
        prompt_tokens,
        train_step_display,
        is_trainable_policy_model,
        get_hidden_states_flag=True
    ):
        """
        Generates a sequence using model_obj (with optional forced-answer logic),
        then computes its log_probs and hidden_states via a forward pass.
        """
        device = self.accelerator.device
        model_vocab_size = self._get_model_config_vocab_size(model_obj)
        processing_class_vocab_size = self._get_processing_class_vocab_size()

        # Ensure processing_class has a valid pad token
        if self.processing_class.pad_token_id is None or self.processing_class.pad_token_id < 0:
            eos_tok = self.processing_class.eos_token or self.processing_class.eos_token_id
            self.processing_class.add_special_tokens({'pad_token': eos_tok})
            model_obj.resize_token_embeddings(self._get_processing_class_vocab_size())

        # Determine pad ID
        pad_id = self.generation_config.pad_token_id or self.processing_class.pad_token_id
        if pad_id < 0 or pad_id > processing_class_vocab_size:
            safe_id = self.processing_class.eos_token_id
            if safe_id is None or safe_id < 0 or safe_id > processing_class_vocab_size:
                raise ValueError(
                    f"Cannot generate with {model_obj.__class__.__name__}: "
                    f"Invalid pad_token_id ({pad_id}) and invalid EOS token ({safe_id}) "
                    f"for processing_class vocab size {processing_class_vocab_size}"
                )
            pad_id = safe_id

        # Copy config and set pad_token_id
        gen_cfg = self.generation_config
        gen_cfg.pad_token_id = pad_id

        # Token generation
        try:
            special_token_id = 151668
            pad_token_id = self.processing_class.pad_token_id
            unwrapped = self.accelerator.unwrap_model(model_obj)
            unwrapped_device = unwrapped.device

            with torch.no_grad():
                # Move inputs
                inputs = {k: v.to(unwrapped_device) for k, v in prompt_tokens.items()}

                if not self.rft_config.force_answer:
                    # Standard generation
                    generated = unwrapped.generate(
                        **inputs,
                        generation_config=gen_cfg
                    )
                else:
                    # Force-answer: generate until special_token_id appears
                    batch_seqs = []
                    for i in range(inputs['input_ids'].size(0)):
                        seq = unwrapped.generate(
                            **{k: v[i:i+1] for k, v in inputs.items()},
                            generation_config=gen_cfg
                        )[0]
                        # Append special token until seen
                        while special_token_id not in seq:
                            seq = torch.cat([seq, torch.tensor([special_token_id], device=unwrapped_device)])
                            seq = unwrapped.generate(
                                input_ids=seq.unsqueeze(0),
                                generation_config=gen_cfg
                            )[0]
                        batch_seqs.append(seq)
                    # Pad sequences
                    max_len = max(s.size(0) for s in batch_seqs)
                    generated = torch.stack([
                        F.pad(s, (0, max_len - s.size(0)), value=pad_token_id)
                        for s in batch_seqs
                    ], dim=0)

            prompt_completion_ids = generated.to(device)
        except Exception as e:
            printc(e)
            return None, None, None

        # Validate generation output
        if prompt_completion_ids.numel() == 0:
            return None, None, None
        min_id, max_id = prompt_completion_ids.min().item(), prompt_completion_ids.max().item()
        if min_id < 0 or max_id >= model_vocab_size:
            return None, None, None
        # Skip if no new tokens
        if prompt_completion_ids.size(1) <= prompt_tokens['input_ids'].size(1):
            return None, None, None

        # Build attention mask
        attention_mask = torch.ones_like(prompt_completion_ids, device=device)

        # Forward pass for log_probs and hidden_states
        try:
            if not is_trainable_policy_model and hasattr(model_obj, 'eval'):
                model_obj.eval()
            outputs = model_obj(
                input_ids=prompt_completion_ids,
                attention_mask=attention_mask,
                output_hidden_states=get_hidden_states_flag
            )
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            hidden_states = outputs.hidden_states if get_hidden_states_flag and hasattr(outputs, 'hidden_states') else None
        except Exception as e:
            printc(e)
            return None, None, None

        if is_trainable_policy_model and not log_probs.requires_grad:
            return None, None, None

        return prompt_completion_ids, log_probs, hidden_states


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
    
    
    def _prepare_item_prompt(self, batch_item, model_vocab_size, train_step_display):
        """Prepares and tokenizes the prompt for a single batch item."""
        question = batch_item.get(self.rft_config.question)
        solution = batch_item.get(self.rft_config.answer) # Keep solution for later, though not used in prompt directly
        if question is None or solution is None:
            printc(f"Skipping item {train_step_display}: missing '{self.rft_config.question}' or '{self.rft_config.answer}'.", "yellow")
            return None, None, None

        messages = [{'role': 'user', 'content': self.rft_config.system_prompt + question}]
        try:
            q_text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            q_text += self.rft_config.b_think
        except Exception as e:
            printc(f"Error applying chat template for item {train_step_display}: {e}", "red")
            return None, None, None

        prompt_tokens_dict = self.processing_class(
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
    
    
    def _get_model_config_vocab_size(self, model_obj):
        """Determines and returns a specific model's vocabulary size from its config."""
        try:
            # Unwrap the model to get the base config, handle PEFT potentially
            unwrapped_model_for_config = self.accelerator.unwrap_model(model_obj)
            if hasattr(unwrapped_model_for_config, 'is_peft_model') and unwrapped_model_for_config.is_peft_model and hasattr(unwrapped_model_for_config, 'base_model'):
                 model_config = unwrapped_model_for_config.base_model.model.config
            else:
                 model_config = unwrapped_model_for_config.config
            return model_config.vocab_size
        except AttributeError as e:
            raise ValueError(f"Could not determine model vocabulary size from {model_obj.__class__.__name__} config: {e}") 
    
    
    
    
    
    def init_callbacks(self):
        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        self.callback_handler = CallbackHandler(
                 default_callbacks,
                 self.model,
                 self.processing_class,
                 self.optimizer,
                 self.lr_scheduler,
        )
        self.callback_handler.on_train_begin(self.args, self.state, self.control)
        self.state, self.control = TrainerState(), TrainerControl()
        self.state.global_step = 0
        
        
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
                model=self.model, processing_class=self.processing_class, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            )
            print("Callback handler initialized.")    

    def get_full_kl_divergence(self, full_policy_log_probs, full_ref_log_probs, full_input_ids):
        """
        Compute KL divergence between policy and reference log-probs.
        """
        if full_ref_log_probs is None or full_policy_log_probs is None:
            return None
        if full_input_ids.shape[1] <= 1:
            return None  # Need at least 2 tokens for KL calculation

        actual_len = min(
            full_policy_log_probs.shape[1],
            full_ref_log_probs.shape[1],
            full_input_ids.shape[1] - 1
        )

        if actual_len <= 0:
            return None

        # Truncate tensors to actual length
        policy_logp = full_policy_log_probs[:, :actual_len, :]
        ref_logp = full_ref_log_probs[:, :actual_len, :]
        token_ids = full_input_ids[:, 1:1+actual_len]  # Skip first token

        vocab_size = policy_logp.shape[-1]
        if token_ids.max() >= vocab_size or token_ids.min() < 0:
            return None  # Invalid token indices

        # Gather log-probs for actual tokens
        policy_token_log_probs = torch.gather(policy_logp, -1, token_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = torch.gather(ref_logp, -1, token_ids.unsqueeze(-1)).squeeze(-1)

        # Compute KL divergence
        kl_div = torch.clamp(
            policy_token_log_probs - ref_token_log_probs.detach(),
            min=-30, max=30
        )

        return kl_div  # Shape: (B, actual_len)




    def _decode_and_split_completion(
       self,
       prompt_completion_ids: torch.Tensor,
       prompt_length: int,
       model_vocab_size: int,
       prompt_tokens,             # we don’t actually use this
       train_step_display: str,
       ):
       # 1) Extract the “response” IDs and bail early if empty
       completion_ids = prompt_completion_ids[:, prompt_length:]
       if completion_ids.numel() == 0:
          printc(f"Skipping item {train_step_display}: no completion tokens.", "yellow")
          return None, None, None, None

       # 2) Validate ID range
       mn, mx = int(completion_ids.min()), int(completion_ids.max())
       if mn < 0 or mx >= model_vocab_size:
          printc(
             f"ERROR: invalid token IDs (min={mn}, max={mx}) "
             f"for item {train_step_display}.", 
             "red"
          )
          return None, None, None, None

      # 3) Decode to text
       try:
          decoded = self.processing_class.decode(
             completion_ids[0],
             skip_special_tokens=True,
             clean_up_tokenization_spaces=True,
          )
       except Exception as e:
          printc(f"Error decoding completion for item {train_step_display}: {e}", "red")
          return None, None, None, None

      # 4) Strip any “end-of-think” prefix
       text = decoded
       e_think = getattr(self.rft_config, "e_think", None)
       if e_think is not None and text.startswith(e_think):
          text = text[len(e_think):].lstrip()

      # 5) Split off answer
       answer_delim = getattr(self.rft_config, "answer_delimiter", e_think)
       if answer_delim and answer_delim in text:
          cot_text, answer_text = text.split(answer_delim, 1)
          cot_text, answer_text = cot_text.strip(), answer_text.strip()
       else:
          cot_text, answer_text = text.strip(), None

       if not cot_text:
          printc(f"Skipping item {train_step_display}: empty CoT after split.", "yellow")
          return None, None, answer_text, decoded

      # 6) Break CoT into steps
       delim = getattr(self.rft_config, "delimiter", None)
       if delim is None:
          printc(f"ERROR: no `rft_config.delimiter` defined.", "red")
          return None, None, answer_text, decoded

       steps_text_raw = split_cot(cot_text, delim)
       if len(steps_text_raw) == 0:
          printc(
             f"Skipping item {train_step_display}: no steps found "
             f"in CoT '{cot_text[:50]}...'.",
             "yellow",
          )
          return None, None, answer_text, decoded

      # 7) Tokenize each step and build a masked‐label tensor for it
       steps_ids_raw = []
       for step in steps_text_raw:
          step = step.strip()
          if step == "":
             printc(f"Warning: empty step for item {train_step_display}, skipping.", "grey")
             continue

          try:
             tok_output = self.processing_class(
                   step,
                   return_tensors="pt",
                   padding=False,
                   add_special_tokens=False,
             )
             step_ids = tok_output["input_ids"][0]  # shape (step_len,)
          except Exception as e:
             printc(
                   f"Error tokenizing step fooo '{step[:30]}...' "
                   f"for item {train_step_display}: {e}",
                   "red",
             )
             return None, None, answer_text, decoded

         # don’t do “if not step_ids”—that would trigger your boolean‐tensor error!
          if step_ids.numel() == 0:
             printc(
                   f"Warning: step '{step[:30]}...' tokenized to zero length, skipping.",
                   "grey",
             )
             continue

         # build a labels tensor full of -100, same shape as prompt_completion_ids
          labels = torch.full_like(prompt_completion_ids, -100)
          end = prompt_length + step_ids.size(0)
          labels[0, prompt_length:end] = step_ids
          steps_ids_raw.append(labels)

       if len(steps_ids_raw) == 0:
          printc(
             f"All steps for item {train_step_display} were empty or un‐tokenizable.",
             "yellow",
          )
          return None, None, answer_text, decoded

       return steps_text_raw, steps_ids_raw, answer_text, decoded

            


   


    def gen_masks(self, full_tensor, steps_data, prompt, step_input_ids):
        foo = self.processing_class.encode
        print()




    def _calculate_step_losses_for_item(self, steps_data, full_policy_log_probs, prompt_completion_ids, 
                                    full_per_token_kl, model_vocab_size, train_step_display):
        """Calculates and returns a list of per-step losses for an item."""
        step_losses = []
        step_metrics = []  # optional: track individual metrics per step
        gradient_lost = False
        device = self.accelerator.device

        for step_idx, step_info in enumerate(steps_data):
            if gradient_lost:
                printc(f"Gradient lost before step {step_idx} (item {train_step_display}), skipping remaining steps.", "yellow")
                break

            # Reward extraction
            reward_key = (
                'whitened_score' 
                if self.rft_config.whiten_rewards and 'whitened_score' in step_info 
                else 'combined_score'
            )
            if reward_key not in step_info:
                printc(f"Skipping step {step_idx}: missing '{reward_key}'","green")
                continue
            
            raw_reward = step_info[reward_key]
            if math.isnan(raw_reward) or math.isinf(raw_reward):
                printc(f"NaN/Inf reward at step {step_idx}, using 0.0","yellow")
                raw_reward = 0.0
            reward = torch.tensor(raw_reward, device=device)

            # Tokenize step text
            txt = step_info.get("txt")
            if not txt:
                printc(f"Skipping step {step_idx}: empty text","yellow")
                continue
            tokens = self.processing_class(txt, return_tensors="pt", padding=False, add_special_tokens=False)
            step_ids = tokens['input_ids'].to(device)
            if step_ids.size(1) == 0:
                printc(f"Skipping step {step_idx}: zero-length after tokenization","grey")
                continue

            # Extract distributions
            logp_dist = self.extract_step_values(full_policy_log_probs, prompt_completion_ids, step_ids)
            if logp_dist is None or not logp_dist.requires_grad:
                printc(f"Skipping step {step_idx}: cannot extract policy log_probs or no grad","yellow")
                gradient_lost = (logp_dist is not None and not logp_dist.requires_grad)
                if gradient_lost: break
                continue

            # Gather log-prob for actual tokens
            idx = step_ids.unsqueeze(-1)
            logp = torch.gather(logp_dist, -1, idx).squeeze(-1)
            if not logp.requires_grad:
                printc(f"ERROR: no grad after gather at step {step_idx}","red")
                gradient_lost = True
                break

            # KL term
            if full_per_token_kl is not None:
                kl_term = self.extract_step_values(full_per_token_kl, prompt_completion_ids, step_ids)
                if kl_term is None:
                    kl_term = torch.zeros_like(logp)
            else:
                kl_term = torch.zeros_like(logp)

            # Compute losses for this step
            policy_loss = -(logp * reward.unsqueeze(-1))
            kl_loss = kl_term * self.rft_config.beta
            value_loss = torch.tensor(0.0, device=device)

            # Validate
            for name, term in [('policy', policy_loss), ('kl', kl_loss), ('value', value_loss)]:
                if torch.isnan(term).any() or torch.isinf(term).any():
                    printc(f"NaN/Inf in {name}_loss at step {step_idx}, skipping step","yellow")
                    break
            else:
                # Compute scalar total
                scalar_loss = (policy_loss.mean() + kl_loss.mean() + value_loss.mean())
                step_losses.append(scalar_loss)
                step_metrics.append({
                    'policy_loss': policy_loss.mean().item(),
                    'kl_loss': kl_loss.mean().item(),
                    'value_loss': value_loss.mean().item(),
                    'total_loss': scalar_loss.item()
                })
                continue
            # If broken out of inner validation loop, skip
            continue

        return step_losses, step_metrics, gradient_lost

    
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        self._validate_initial_setup()
        main_model_vocab_size = self._get_model_config_vocab_size(self.model)
        device = self.accelerator.device
        self.model.train() 
        
        
        train_dataloader = self.get_train_dataloader()
        if train_dataloader is None: raise ValueError("No train dataloader found.")

        len_dataloader, _, num_train_epochs, max_steps = \
            self._calculate_dataloader_dependent_steps(train_dataloader)
        global_step_for_loop = self.state.global_step 

        self._initialize_trainer_internals(resume_from_checkpoint, max_steps)
        self.init_callbacks()
        for epoch in range(num_train_epochs):
            printc(f"Starting Epoch {epoch+1}/{num_train_epochs}", "yellow")
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            batch_iterator = iter(train_dataloader)
            current_epoch_step = 0
            while True: # Loop until break condition (dataset end or max_steps)

                # Check max_steps condition FIRST
                if max_steps > 0 and global_step_for_loop >= max_steps:
                    printc(f"Reached max_steps ({max_steps}). Stopping training.", "yellow")
                    self.control.should_training_stop = True
                    break # Exit inner batch loop
                try:
                    batch = next(batch_iterator)
                    # Move batch to device if it's a tensor or dict of tensors (handled by default Trainer Dataloader usually)
                    # If using custom collate_fn, might need manual moving here. Assuming simple list for now.
                except StopIteration:
                    printc(f"End of dataloader reached for epoch {epoch+1}.", "blue")
                    break # Exit inner batch loop, go to next epoch

                # Increment step counter for the current epoch
                current_epoch_step += 1
                train_step_display = self.state.global_step * self.rft_config.gradient_accumulation_steps + (current_epoch_step % self.rft_config.gradient_accumulation_steps)
                print(train_step_display)
                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)
                prompt_completion_ids = None
                full_policy_log_probs = None
                full_hidden_states = None
                full_ref_log_probs = None
                full_per_token_kl = None
                total_loss = None
                perform_backward = False
                gradient_lost_in_loop = False
                log_total_loss = 0.0; log_policy_loss = 0.0; 
                log_kl_loss = 0.0; log_value_loss = 0.0
                batch_item = batch[0] #todo fix
                item_total_loss = None
                perform_backward_for_item = False
                
                prompt_tokens, prompt_length, question, solution = self._prepare_item_prompt(
                    batch_item, 
                    main_model_vocab_size, 
                    train_step_display
                )
                # printc(prompt_tokens)
                # import sys
                # sys.exit(0)
                printc(prompt_tokens,'red')
                
                if prompt_tokens is None:
                    continue
                prompt_completion_ids, full_policy_log_probs, full_policy_hidden_states = self._generate_sequence_and_get_model_outputs(
                    self.model,    
                    prompt_tokens,
                    train_step_display,
                    is_trainable_policy_model=True, 
                    get_hidden_states_flag=True 
                )
                if prompt_completion_ids is None:
                    continue
                print(prompt_completion_ids)
                printc(prompt_tokens['input_ids'].shape)
                printc(prompt_completion_ids.shape)
                full_ref_log_probs = None
                foo = self._generate_sequence_and_get_model_outputs(
                    self.ref_model,    
                    prompt_tokens,
                    train_step_display,
                    is_trainable_policy_model=False, 
                    get_hidden_states_flag=False 
                )


                
                full = torch.cat([prompt_tokens['input_ids'], prompt_completion_ids], dim=1)

                printc(full)

                full_per_token_kl = self.get_full_kl_divergence(
                    full_policy_log_probs,
                    full_ref_log_probs,
                    prompt_tokens['input_ids']
                )
                steps_text_raw, steps_ids_raw, answer, decode_outputs = self._decode_and_split_completion(prompt_completion_ids, prompt_length, main_model_vocab_size,prompt_tokens, train_step_display)
                printc(steps_ids_raw)
                import sys
                sys.exit(0)
                printc(steps_ids_raw, 'red')
                if decode_outputs is None:
                    continue

                print(f"Q: {question}") 
                steps_data = self._evaluate_and_process_item_rewards(question, steps_text_raw, answer, solution, train_step_display)
                if steps_data is None: 
                    continue

                msks = self.gen_masks(full,steps_data,prompt_tokens['input_ids'] ,steps_ids_raw )

                print_thoughts_colored(steps_data)
                print(steps_data)
                # bar = self._calculate_step_losses_for_item( steps_data, full_policy_log_probs, prompt_completion_ids, 
                #                        full_per_token_kl, main_model_vocab_size, train_step_display)
                # printc(bar,'green')
                # all_policy_loss_terms, all_kl_loss_terms, all_value_losses = [], [], []
                # total_steps_len_tokens = 0
                # gradient_lost_in_loop = False # Reset flag for this item
                # for step_idx, step_info in enumerate(steps_data):
                #     print(step_idx)
                #     print(step_info)

            