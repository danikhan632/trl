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
import sys
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
from typing import List


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







def zero_out_prompt(policy_log_probs: torch.Tensor,
                    ref_log_probs: torch.Tensor,
                    prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns new tensors where all timesteps < prompt_len have been zeroed out.

    Args:
      policy_log_probs: [B, T, V]
      ref_log_probs:    [B, T, V]
      prompt_len:       int

    Returns:
      (policy_masked, ref_masked) both shaped [B, T, V]
      with [:, :prompt_len, :] == 0.
    """
    B, T, V = policy_log_probs.shape

    # make a mask of shape [T], 0 for prompt positions, 1 for gen positions
    device = policy_log_probs.device
    idx = torch.arange(T, device=device)  # [T]
    mask = (idx >= prompt_len).to(policy_log_probs.dtype)  # [T], 0/1

    # expand to [B, T, V]
    mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    mask = mask.expand(B, T, V)             # [B, T, V]

    # multiply to zero out
    policy_masked = policy_log_probs * mask
    ref_masked    = ref_log_probs    * mask

    return policy_masked, ref_masked







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
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
        **kwargs
    ):
        self.rft_config = self._validate_rft_config(rft_config)
        model_kwargs = self.rft_config.model_init_kwargs or {}

        # Load and prepare the base model
        base_model = self._load_model(model, model_kwargs)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.rft_config.gradient_accumulation_steps
        )

        # Attach value head if missing
        self._attach_value_head(base_model)

        # Wrap with PEFT if requested
        self.model, self.is_peft_model = self._setup_peft(base_model, peft_config)

        # Prepare reference model (remote or local)
        self._ref_internal = None
        self.remote_ref_proxy = None
        self._setup_reference(ref_model)

        # Set defaults for evaluation and critic functions
        self._set_default_functions()

        # Build generation config
        self.generation_config = self._configure_generation(processing_class)

        # Define a simple collator
        def data_collator(features): return features

        # Initialize parent Trainer
        super().__init__(
            model=self.model,
            args=self.rft_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )

        # Finalize reference model preparation
        self.ref_model = self._prepare_ref_model()

        # Validate value head placement and dtype
        self._check_value_head()

    def _validate_rft_config(self, cfg: RFTConfig) -> RFTConfig:
        if not isinstance(cfg, RFTConfig):
            raise TypeError("`rft_config` must be an instance of RFTConfig.")
        return cfg

    def _load_model(
        self,
        model: Union[str, PreTrainedModel],
        kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        if isinstance(model, str):
            return AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if isinstance(model, PreTrainedModel):
            return model
        raise TypeError("`model` must be a model name or PreTrainedModel instance.")

    def _attach_value_head(self, model: PreTrainedModel):
        if hasattr(model, 'value_head'):
            print("Value head already exists on the model.")
            return

        cfg = getattr(model, 'config', None)
        if is_peft_available() and isinstance(model, PeftModel):
            cfg = model.base_model.model.config
        if not hasattr(cfg, 'hidden_size'):
            raise AttributeError("Model config missing 'hidden_size'.")

        head = torch.nn.Linear(cfg.hidden_size, 1, dtype=torch.float32)
        with torch.no_grad():
            std = 1 / (cfg.hidden_size + 1)
            head.weight.normal_(0.0, std)
            head.bias.zero_()

        setattr(model, 'value_head', head)
        print("Value head added to the model.")

    def _setup_peft(
        self,
        model: PreTrainedModel,
        peft_cfg: Optional[Any]
    ) -> tuple[PreTrainedModel, bool]:
        if peft_cfg is None:
            return model, False
        if not is_peft_available():
            raise ImportError("PEFT not available.")
        wrapped = get_peft_model(model, peft_cfg)
        print("PEFT model created.")
        return wrapped, True

    def _setup_reference(self, ref_model: Optional[Union[str, PreTrainedModel]]):
        cfg = self.rft_config
        if cfg.remote_ref_model:
            uri = cfg.remote_ref_model_uri
            if not uri:
                raise ValueError("`remote_ref_model_uri` missing.")
            try:
                from pyro5.api import Proxy
                proxy = Proxy(uri)
                proxy._pyroTimeout = 10
                proxy._pyroBind()
                self.remote_ref_proxy = proxy
                print(f"Connected to remote ref model: {uri}")
            except Exception as e:
                raise ConnectionError(f"Remote ref connection failed: {e}") from e
            if ref_model:
                print("Warning: remote ref in use; local `ref_model` ignored.")
        elif self.is_peft_model:
            print("Using PEFT base model as reference.")
        elif ref_model:
            self._ref_internal = self._load_model(ref_model, self.rft_config.model_init_kwargs or {})
        else:
            print("Warning: No reference model configured. KL penalty = 0.")

    def _set_default_functions(self):
        if self.rft_config.evalulate_state_func is None:
            self.rft_config.evalulate_state_func = evaluate_state_gemini
        if self.rft_config.critic_prompt_func is None:
            self.rft_config.critic_prompt_func = get_critic_prompt
        if self.rft_config.process_rewards_func is None:
            self.rft_config.process_rewards_func = process_rewards

    def _configure_generation(
        self,
        processing_class: PreTrainedTokenizerBase
    ) -> GenerationConfig:
        cfg = self.rft_config
        pad_id = getattr(processing_class, 'pad_token_id', None) or processing_class.eos_token_id
        gen_cfg = GenerationConfig(
            max_new_tokens=cfg.response_length,
            temperature=cfg.temperature + 1e-7,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            do_sample=True,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=processing_class.eos_token_id,
        )
        if gen_cfg.pad_token_id is None:
            printc(
                "Warning: pad_token_id None; using eos_token_id.", "yellow"
            )
            gen_cfg.pad_token_id = gen_cfg.eos_token_id
        if gen_cfg.pad_token_id is None:
            raise ValueError(
                "Valid pad_token_id or eos_token_id required."
            )
        return gen_cfg

    def _prepare_ref_model(self) -> Optional[PreTrainedModel]:
        if self._ref_internal is None:
            return None
        dtype = next(self.model.parameters()).dtype
        if self._ref_internal.dtype != dtype:
            print(f"Casting local reference to {dtype}.")
            self._ref_internal = self._ref_internal.to(dtype=dtype)
        return self.accelerator.prepare_model(
            self._ref_internal, evaluation_mode=True
        )

    def _check_value_head(self):


        if not hasattr(self.model, 'value_head'):


            # Ensure model_config has hidden_size
            if not hasattr(self.model, 'hidden_size'):
                raise AttributeError("Model config does not have 'hidden_size' attribute.")

            # OPTION A: specify dtype at creation
            value_head = torch.nn.Linear(
                self.model.hidden_size,
                1,
                dtype=torch.float32
            )


            with torch.no_grad():
                # initialize in float32
                value_head.weight.data.normal_(mean=0.0, std=1/(self.model.hidden_size + 1))
                value_head.bias.data.zero_()

            # If your model is in a mixed-precision context (e.g. AMP), you may still need to
            # explicitly move it to the right device:
            value_head = value_head.to(device=self.model.device, dtype=torch.float32)
            value_head = value_head.float()

            setattr(self.model, 'value_head', value_head)
            print("Value head (float32) added to the model.")
        else:
            # Ensure it's in float32 if it already existed
            self.model.value_head = self.model.value_head.float()
            print("Value head already exists; converted to float32.")




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
    
    
    



    def _decode_and_split_completion(self, prompt_completion_ids, prompt_length, model_vocab_size, train_step_display):
        """Decodes completion, splits CoT, and extracts steps."""
        completion_ids = prompt_completion_ids[:, prompt_length:]
        if completion_ids.shape[1] == 0:
            printc(f"Skipping item {train_step_display}: No completion tokens generated.", "yellow"); return None, None, None
        
        min_comp_id, max_comp_id = completion_ids.min().item(), completion_ids.max().item()
        if min_comp_id < 0 or max_comp_id >= model_vocab_size:
            printc(f"ERROR: Invalid completion token IDs (min={min_comp_id}, max={max_comp_id}) for item {train_step_display}.", "red"); return None, None, None

        try:
            full_text = self.processing_class.decode(completion_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
    



    def extract_step_values(self, full_sequence_tensor, full_input_ids, step_input_ids):
        # Nothing to do if there’s no data or no step tokens
        if full_sequence_tensor is None or step_input_ids is None or step_input_ids.size(1) == 0:
            return None

        # How many output-tokens do we actually have?
        full_tensor_len = full_sequence_tensor.size(1)
        step_len = step_input_ids.size(1)

        # If your model stopped early and you don't even have
        # step_len of outputs, give up.
        if full_tensor_len < step_len:
            printc(f"Extract step values: full tensor length ({full_tensor_len}) < step length ({step_len}), cannot extract.", "red")
            return None

        # Align to the rightmost step_len tokens
        start_idx = full_tensor_len - step_len
        end_idx = full_tensor_len  # exclusive

        try:
            if full_sequence_tensor.dim() == 3:
                # (B, L, V)
                step_values = full_sequence_tensor[:, start_idx:end_idx, :]
            elif full_sequence_tensor.dim() == 2:
                # (B, L)
                step_values = full_sequence_tensor[:, start_idx:end_idx]
            else:
                printc(f"Extract step values: Unexpected tensor dimension {full_sequence_tensor.dim()}.", "red")
                return None
        except IndexError as e:
            printc(f"Extract step values: Slicing error - {e}.", "red")
            return None

        # If the slice is somehow empty, skip it
        if step_values.size(1) == 0:
            printc("Extract step values: Resulting slice is empty.", "yellow")
            return None

        return step_values



    def _evaluate_and_process_item_rewards(
        self,
        question,
        steps_text_raw,
        answer,
        solution,
        train_step_display
    ):
        """Evaluates steps, processes rewards, scales & caps the final answer, then normalizes/whitens all rewards."""
        # 1. Build prompt & call evaluator
        cumulative_reasons = [{f"thought_{i}": txt} for i, txt in enumerate(steps_text_raw)]
        critic_prompt = self.rft_config.critic_prompt_func(
            question, cumulative_reasons, answer, solution
        )
        try:
            eval_data = self.rft_config.evalulate_state_func(critic_prompt, cumulative_reasons)
        except Exception as e:
            printc(f"Error during state evaluation for item {train_step_display}: {e}", "red")
            return None
        if not eval_data or not eval_data.get("steps"):
            printc(f"Skipping item {train_step_display}: no valid eval data.", "yellow")
            return None

        # 2. Process raw step-by-step rewards
        try:
            steps_data = self.rft_config.process_rewards_func(
                eval_data.get("overall_score", 0),
                eval_data["steps"],
                self.rft_config
            )
        except Exception as e:
            printc(f"Error during reward processing for item {train_step_display}: {e}", "red")
            return None
        if not steps_data:
            printc(f"Skipping item {train_step_display}: no step data after processing.", "yellow")
            return None

        # 3. Compute & scale the raw overall_score
        raw_score = eval_data.get("overall_score", 0)
        if 10 < raw_score < 100:
            raw_score /= 100.0

        # force it above caps so the cap logic always triggers
        scale_factor = 2.0
        scaled_score = raw_score * scale_factor

        # 4. Cap against CoT step scores
        cot_scores = [s["combined_score"] for s in steps_data if "combined_score" in s]
        if cot_scores:
            min_cot, max_cot = min(cot_scores), max(cot_scores)
            cap1 = 1.5 * min_cot
            cap2 = max_cot + 1.5
            capped_score = min(scaled_score, cap1, cap2)
        else:
            capped_score = scaled_score

        # 5. Append final answer step
        steps_data.append({
            "txt": answer,
            "combined_score": capped_score
        })

        # 6. Normalize & whiten *all* combined_scores (including the answer)
        if self.rft_config.whiten_rewards and steps_data:
            try:
                # collect into tensor
                all_scores = torch.tensor(
                    [s["combined_score"] for s in steps_data],
                    device=self.accelerator.device,
                    dtype=torch.float32
                )
                if all_scores.numel() > 1:
                    mean, std = all_scores.mean(), all_scores.std()
                    whitened = ((all_scores - mean) / (std + 1e-8)
                                if std > 1e-5 else (all_scores - mean))
                else:
                    # single value: just zero-center
                    whitened = all_scores - all_scores.mean()

                # write back into each step
                for idx, step in enumerate(steps_data):
                    w = whitened[idx]
                    if not (torch.isnan(w) or torch.isinf(w)):
                        step["whitened_score"] = w.item()
                    else:
                        tag = "answer" if idx == len(steps_data)-1 else f"step {idx}"
                        printc(f"Warning: NaN/Inf in whitened score for {tag}, item {train_step_display}.",
                               "yellow")
            except Exception as e:
                printc(f"Error during reward whitening for item {train_step_display}: {e}.", "red")

        return steps_data


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

    def cumulative_slices(
        self,
        x: torch.Tensor,
        boundaries: List[int],
        model,
    ) -> List[torch.Tensor]:
        """
        Given:
        x          -- a tensor of shape [1, n, 1024]
        boundaries -- a list of integers in [0, n], e.g. [0, 404, 449, ..., n]
        Returns:
        A list of tensors [ x[:, :b, :] for each b in boundaries if b > 0 ].
        Their shapes will be:
            [1, 404, 1024],
            [1, 449, 1024],
            ...,
            [1, 904, 1024].
        """
        # ensure sorted & unique
        bounds = sorted(set(boundaries))
        out = []
        for b in bounds:
            if b <= 0:
                # skip the 0 boundary (would give shape [1,0,1024])
                continue
            if b > x.size(1):
                raise ValueError(f"Boundary {b} exceeds tensor length {x.size(1)}")
            foo = x[:, :b, :]
            foo = foo.to(torch.float32)
            out.append(foo)
        value_estimates = []  # will hold one (B,) tensor per segment

        for seg in out:
            # seg: (B, Li, hidden_size)
            # 1) apply the value head → (B, Li, 1)
            v_logits = model.value_head(seg)
            # 2) drop the last dim → (B, Li)
            v_logits = v_logits.squeeze(-1)
            # 3) pool over the token axis → (B,)
            #    here we simply take the mean across Li
            v_scalar = v_logits.mean(dim=1)
            value_estimates.append(v_scalar*  self.rft_config.vf_coef)
        return value_estimates



    def decode_and_split_completion(
        self,
        combined_token_ids: torch.Tensor,
        prompt_length: int,
        vocab_size: int,
        prompt_tokens: dict,  # unused, only shape is used for index tracking
        step_label: str,
    ):
        """
        Decodes the completion tokens from combined_token_ids, splits into CoT steps and final answer,
        then builds a list of label tensors for each step (masking the prompt tokens).
        """
        # Extract completion tokens
        completion_ids = combined_token_ids[:, prompt_length:]
        if completion_ids.numel() == 0:
            print(f"Skipping {step_label}: no completion tokens.")
            return None, None, None, None

        # Validate token ID range
        min_id, max_id = int(completion_ids.min()), int(completion_ids.max())
        if min_id < 0 or max_id >= vocab_size:
            print(
                f"ERROR: invalid token IDs (min={min_id}, max={max_id}) for {step_label}."
            )
            return None, None, None, None

        # Decode to text
        try:
            raw_text = self.processing_class.decode(
                completion_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        except Exception as e:
            print(f"Error decoding completion for {step_label}: {e}")
            return None, None, None, None

        # Remove any prefix (e.g., end-of-think marker)
        e_think_marker = getattr(self.rft_config, "e_think", None)
        if e_think_marker and raw_text.startswith(e_think_marker):
            raw_text = raw_text[len(e_think_marker):].lstrip()

        # Split into CoT and answer
        answer_delimiter = getattr(self.rft_config, "answer_delimiter", e_think_marker)
        if answer_delimiter and answer_delimiter in raw_text:
            cot_text, answer_text = map(str.strip, raw_text.split(answer_delimiter, 1))
        else:
            cot_text, answer_text = raw_text.strip(), None

        if not cot_text:
            print(f"Skipping {step_label}: empty CoT after split.")
            return None, None, answer_text, raw_text

        # Split CoT into individual steps
        step_separator = getattr(self.rft_config, "delimiter", None)
        if not step_separator:
            print("ERROR: `rft_config.delimiter` not defined.")
            return None, None, answer_text, raw_text

        cot_steps = split_cot(cot_text, step_separator)
        if not cot_steps:
            preview = cot_text[:50] + ('...' if len(cot_text) > 50 else '')
            print(
                f"Skipping {step_label}: no steps found in CoT '{preview}'."
            )
            return None, None, answer_text, raw_text

        # Prepare prompt mask for labels
        device = combined_token_ids.device
        dtype = combined_token_ids.dtype
        prompt_mask = torch.full(
            (1, prompt_length),
            -100,
            device=device,
            dtype=dtype,
        )

        labeled_steps = []
        accumulated_ids = None
        # Track step boundary indices: start with prompt end
        step_boundaries = [0, prompt_tokens['input_ids'].shape[1]]

        # Build label tensors for each CoT step
        for step in cot_steps:
            step_text = step.strip()
            if not step_text:
                print("Warning: empty step, skipping.")
                continue

            tokens = self.processing_class(
                step_text,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            ).to(device)
            current_ids = tokens["input_ids"]

            # Accumulate step tokens
            if accumulated_ids is None:
                accumulated_ids = current_ids
            else:
                accumulated_ids = torch.cat([accumulated_ids, current_ids], dim=1)

            # Create label tensor: mask prompt, then include all accumulated step tokens
            labels = torch.cat([prompt_mask, accumulated_ids], dim=1)
            step_boundaries.append(labels.shape[1])
            labeled_steps.append(labels)

        if not labeled_steps:
            print(f"All steps for {step_label} were empty or un-tokenizable.")
            return None, None, answer_text, raw_text

        # Final boundary at end of all tokens
        step_boundaries.append(combined_token_ids.shape[1])


        return cot_steps, answer_text, step_boundaries



    
    def soft_reset(self):
        self.idxs=[]
        self.hidden_states=[]

    def _get_new_logprobs_value_and_kl_for_thought_segment(
        self,
        policy_model_current, # current self.model (policy + value head)
        ref_model_current,    # self.ref_model
        full_generated_ids: torch.Tensor, # tensor (1, full_len), e.g., prompt_completion_ids
        segment_prompt_and_thought_start_idx: int, # Index in full_generated_ids where the relevant prompt part for this thought begins
        segment_action_start_idx: int,   # Index in full_generated_ids where the current action (thought tokens) begins
        segment_action_end_idx: int,     # Index in full_generated_ids where the current action (thought tokens) ends
    ):
        """
        Performs a forward pass for a given segment to get new logprobs for the action,
        a new value estimate for the state *before* the action, and KL divergence for the action.
        """
        device = self.accelerator.device

        # 1. Prepare inputs
        action_tokens = full_generated_ids[:, segment_action_start_idx:segment_action_end_idx]
        if action_tokens.size(1) == 0:
            return (torch.tensor(0.0, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device)) # Zero loss contribution

        # Input for the forward pass: from where the relevant context for this thought starts, up to its end
        # This ensures hidden states are computed correctly for value and policy logits.
        input_ids_for_segment_forward = full_generated_ids[:, segment_prompt_and_thought_start_idx:segment_action_end_idx]
        attention_mask_for_segment_forward = torch.ones_like(input_ids_for_segment_forward, device=device)

        # Determine relative start of action within input_ids_for_segment_forward
        relative_action_start_idx = segment_action_start_idx - segment_prompt_and_thought_start_idx
        
        # 2. Forward pass with current policy model
        # Ensure gradients are enabled if in training mode
        grad_context = torch.enable_grad() if policy_model_current.training else torch.no_grad()
        with grad_context:
            outputs = policy_model_current(
                input_ids=input_ids_for_segment_forward,
                attention_mask=attention_mask_for_segment_forward,
                output_hidden_states=True  # Ensure hidden states are output
            )
            all_logits_segment = outputs.logits  # (1, len(segment_forward), vocab_size)
            # Ensure hidden_states are available
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                raise RuntimeError("Model did not return hidden_states. Ensure `output_hidden_states=True` is effective.")
            all_hidden_states_segment = outputs.hidden_states[-1]  # (1, len(segment_forward), hidden_size)

        # 3. Extract new policy log probabilities for the action_tokens
        # Logits for predicting action_tokens[j] are at all_logits_segment[:, relative_action_start_idx + j -1, :]
        logits_for_action = all_logits_segment[:, (relative_action_start_idx -1) : (relative_action_start_idx + action_tokens.size(1) -1) , :]
        
        log_probs_dist_new = F.log_softmax(logits_for_action, dim=-1)
        new_action_log_probs_per_token = torch.gather(log_probs_dist_new, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
        new_action_log_probs_sum = new_action_log_probs_per_token.sum()

        # 4. Extract new value V(S_prev)
        # S_prev is the state just before the action starts. Its hidden state is at relative_action_start_idx - 1.
        if relative_action_start_idx > 0:
            s_prev_hidden_state = all_hidden_states_segment[:, relative_action_start_idx - 1, :]
        else: # Action starts at the very beginning of input_ids_for_segment_forward
              # This means S_prev is effectively the state *before* segment_prompt_and_thought_start_idx
              # This case requires careful handling of what V(S_prev) means.
              # For simplicity, if S_prev is empty, value_head might take BOS or learn V(empty).
              # Here, assume if relative_action_start_idx is 0, we might need a different V_S_prev logic (e.g. from a BOS token hs)
              # Let's assume S_prev always has some content for CoT steps, so relative_action_start_idx > 0.
            printc("Warning: relative_action_start_idx is 0. V(S_prev) might be inaccurate.", "yellow")
            # Fallback: use the first hidden state if available, or zero. Could lead to issues.
            s_prev_hidden_state = all_hidden_states_segment[:, 0, :]


        # Ensure value_head input is of the correct dtype (often float32)
        s_prev_hidden_state_for_value = s_prev_hidden_state.to(policy_model_current.value_head.weight.dtype)
        new_V_S_prev_recomputed = policy_model_current.value_head(s_prev_hidden_state_for_value).squeeze(-1) # (1,)

        # 5. Calculate KL divergence for the current action_tokens
        kl_for_action_sum = torch.tensor(0.0, device=device)
        if ref_model_current is not None and self.rft_config.beta > 0: # beta is kl_coef
            with torch.no_grad(): # No grads for ref model
                ref_outputs = ref_model_current(
                    input_ids=input_ids_for_segment_forward, # Same input as policy
                    attention_mask=attention_mask_for_segment_forward,
                )
                ref_all_logits_segment = ref_outputs.logits
                ref_logits_for_action = ref_all_logits_segment[:, (relative_action_start_idx -1) : (relative_action_start_idx + action_tokens.size(1) -1) , :]
                
                ref_log_probs_dist = F.log_softmax(ref_logits_for_action, dim=-1)
                ref_action_log_probs_per_token = torch.gather(ref_log_probs_dist, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
                
                # KL per token: policy_log_p(token) - ref_log_p(token)
                # Use new_action_log_probs_per_token (which has grad) and ref_action_log_probs_per_token.detach()
                kl_div_per_token = new_action_log_probs_per_token - ref_action_log_probs_per_token.detach()
                kl_for_action_sum = kl_div_per_token.sum()
                # Clamp KL to avoid large values
                kl_for_action_sum = torch.clamp(kl_for_action_sum, min=-30, max=30)

        return new_action_log_probs_sum, new_V_S_prev_recomputed, kl_for_action_sum
    
    
    
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        self._validate_initial_setup()
        main_model_vocab_size = self._get_model_config_vocab_size(self.model)
        device = self.accelerator.device
        
        # Ensure model is in training mode for policy and value head updates
        self.model.train()
        if self.ref_model: # Ref model should always be in eval mode
            self.ref_model.eval()

        train_dataloader = self.get_train_dataloader()
        if train_dataloader is None: raise ValueError("No train dataloader found.")

        len_dataloader, _, num_train_epochs, max_steps = \
            self._calculate_dataloader_dependent_steps(train_dataloader)
        
        self._initialize_trainer_internals(resume_from_checkpoint, max_steps)
        # self.init_callbacks() # Called by _initialize_trainer_internals or parent

        # For tracking accumulation across items if necessary
        self.state.num_backward_passes_accumulated = 0
        # For display purposes: items processed in current accumulation cycle
        items_processed_in_accumulation_cycle = 0

        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for epoch in range(num_train_epochs):
            printc(f"Starting Epoch {epoch+1}/{num_train_epochs}", "yellow")
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            
            if hasattr(train_dataloader, "set_epoch"): # For DistributedSampler
                train_dataloader.set_epoch(epoch)

            batch_iterator = iter(train_dataloader)
            
            while True: # Loop over batches/items
                if self.state.global_step >= max_steps and max_steps > 0 :
                    printc(f"Reached max_steps ({max_steps}). Stopping training.", "yellow")
                    self.control.should_training_stop = True
                    break 
                try:
                    # Assuming batch is a list containing one item dictionary
                    batch_list = next(batch_iterator)
                    if not isinstance(batch_list, list) or not batch_list:
                        printc("Warning: Batch is not a list or is empty, skipping.", "yellow")
                        continue
                    batch_item = batch_list[0] 
                except StopIteration:
                    printc(f"End of dataloader reached for epoch {epoch+1}.", "blue")
                    break 

                self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control) # "Step" here is per item/micro-batch
                
                # Soft reset per item for idxs
                self.soft_reset() 

                # --- 1. Prepare Prompt ---
                prompt_tokens, prompt_length, question_text, solution_text = self._prepare_item_prompt(
                    batch_item, main_model_vocab_size, f"E{epoch+1}_S{self.state.global_step}"
                )
                if prompt_tokens is None: 
                    continue

                # --- 2. Generate Full CoT Sequence (Policy Model "Old" Outputs) ---
                # This provides old_log_probs and old_hidden_states for PPO
                prompt_completion_ids, full_policy_log_probs_initial, full_policy_hidden_states_initial = \
                    self._generate_sequence_and_get_model_outputs(
                        self.model, prompt_tokens, f"E{epoch+1}_S{self.state.global_step}_PolicyRollout",
                        is_trainable_policy_model=False, # Treat as fixed for this rollout; grad comes from re-eval
                        get_hidden_states_flag=True
                    )
                if prompt_completion_ids is None or full_policy_log_probs_initial is None or full_policy_hidden_states_initial is None:
                    printc(f"Skipping item E{epoch+1}_S{self.state.global_step}: Policy generation failed.", "yellow")
                    continue
                
                # --- 3. Get Reference Log Probabilities ---
                _, full_ref_log_probs, _ = self._generate_sequence_and_get_model_outputs(
                    self.ref_model, prompt_tokens, f"E{epoch+1}_S{self.state.global_step}_RefRollout",
                    is_trainable_policy_model=False, get_hidden_states_flag=False,
                    # Provide generated_ids_override=prompt_completion_ids to ensure ref model scores the same sequence
                    # This needs to be added to _generate_sequence_and_get_model_outputs if not present
                    # For now, assuming it generates and then we align. More robust is override.
                )
                # If using override, ensure generated_ids_override is handled in _generate_sequence_and_get_model_outputs
                # For now, let's assume it's handled by aligning based on prompt_tokens and prompt_completion_ids

                # --- 4. Decode and Segment the CoT ---
                # `self.idxs` will be populated here.
                # Example: [0, prompt_len, prompt_len+len(T1), ..., end_of_last_thought, full_completion_len]
                steps_text_raw,  answer_text_decoded, self.idxs = self.decode_and_split_completion(
                        prompt_completion_ids, prompt_length, main_model_vocab_size,
                        prompt_tokens, f"E{epoch+1}_S{self.state.global_step}"
                    )
                if steps_text_raw is None or not self.idxs or len(self.idxs) < 2: # Need at least prompt_len and one step end
                    printc(f"Skipping item E{epoch+1}_S{self.state.global_step}: CoT splitting failed.", "yellow")
                    continue
                
                # --- 5. Get Value Estimates V_old(S_boundary) from initial rollout ---
                # values_at_boundaries[k] = V_old(State up to self.idxs[k+1])
                values_at_boundaries_old = self.cumulative_slices(
                    full_policy_hidden_states_initial[-1], self.idxs, self.model
                )
                if not values_at_boundaries_old or len(values_at_boundaries_old) != (len(self.idxs) -1):
                    printc(f"Skipping item E{epoch+1}_S{self.state.global_step}: Value estimation mismatch. Got {len(values_at_boundaries_old)} values for {len(self.idxs)-1} boundaries.", "red")
                    continue
                
                # --- 6. Evaluate Rewards for each generated thought step ---
                # rewards_for_thoughts should align with steps_text_raw
                rewards_data_full = self._evaluate_and_process_item_rewards(
                    question_text, steps_text_raw, answer_text_decoded, solution_text,
                    f"E{epoch+1}_S{self.state.global_step}"
                )
                print_thoughts_colored(rewards_data_full)
                if rewards_data_full is None: 
                    printc(f"Skipping item E{epoch+1}_S{self.state.global_step}: Reward evaluation failed.", "yellow")
                    continue
                
                # Extract just the reward values for PPO loop
                rewards_for_thoughts = []
                for step_reward_data in rewards_data_full:
                    reward_key = 'whitened_score' if self.rft_config.whiten_rewards and 'whitened_score' in step_reward_data else 'combined_score'
                    if reward_key in step_reward_data:
                        rewards_for_thoughts.append(step_reward_data[reward_key])
                    else:
                        rewards_for_thoughts.append(0.0) # Default if missing

                num_thought_segments = len(rewards_for_thoughts) # This includes the final answer step
                if num_thought_segments == 0:
                    printc(f"Skipping item E{epoch+1}_S{self.state.global_step}: No reward segments.", "yellow")
                    continue
                if num_thought_segments > len(values_at_boundaries_old): # values_at_boundaries_old has V(Prompt), V(P+T1), ... V(P+...+TN)
                     printc(f"Warning: Mismatch num_rewards ({num_thought_segments}) and num_values ({len(values_at_boundaries_old)}). Trimming rewards.", "yellow")
                     rewards_for_thoughts = rewards_for_thoughts[:len(values_at_boundaries_old)]
                     num_thought_segments = len(rewards_for_thoughts)

                # --- 7. Iterate through each THOUGHT STEP of the CoT and perform PPO update ---
                current_item_total_loss = 0.0
                for k_thought in range(num_thought_segments):
                    # Define segment boundaries using self.idxs
                    # S_prev state ends at self.idxs[k_thought + 1] (e.g. self.idxs[1]=prompt_len for 0th thought)
                    # Action (current thought) is from self.idxs[k_thought + 1] to self.idxs[k_thought + 2]
                    segment_prompt_and_thought_start_idx = self.idxs[0] # Usually 0, start of the original prompt
                    segment_action_start_idx = self.idxs[k_thought + 1]
                    segment_action_end_idx = self.idxs[k_thought + 2]

                    if segment_action_end_idx <= segment_action_start_idx:
                        printc(f"Warning: Empty thought segment for k_thought={k_thought}. Skipping.", "grey")
                        continue

                    # a. Get pre-calculated "old" data for this step
                    R_curr_thought = torch.tensor(rewards_for_thoughts[k_thought], device=device)
                    V_S_prev_old = values_at_boundaries_old[k_thought].detach() # V_old(State before action)

                    is_last_thought_in_cot = (k_thought == num_thought_segments - 1)

                    if is_last_thought_in_cot:
                        if self.rft_config.treat_last_step_terminal:
                            V_S_curr_old_for_td = torch.tensor(0.0, device=device)
                        elif (k_thought + 1) < len(values_at_boundaries_old):
                            V_S_curr_old_for_td = values_at_boundaries_old[k_thought + 1].detach()
                        else:
                            V_S_curr_old_for_td = torch.tensor(0.0, device=device)
                    else:
                        if (k_thought + 1) >= len(values_at_boundaries_old):
                            printc(f"Error: Index out of bounds for V_S_curr_old at k_thought={k_thought}", "red")
                            continue
                        V_S_curr_old_for_td = values_at_boundaries_old[k_thought + 1].detach()

                                        
                    # b. Calculate "old" policy log_probs for the action
                    action_tokens_ids = prompt_completion_ids[:, segment_action_start_idx:segment_action_end_idx]
                    if action_tokens_ids.size(1) == 0: 
                        continue

                    # Slice from full_policy_log_probs_initial (B, L-1, V)
                    # Logits for token i are at log_probs[:, i-1, :]
                    # Action tokens are from index segment_action_start_idx to segment_action_end_idx-1
                    # So, log_probs are from index segment_action_start_idx-1 to segment_action_end_idx-2
                    old_log_probs_dist_for_action = full_policy_log_probs_initial[:, (segment_action_start_idx-1):(segment_action_end_idx-1), :]
                    gathered_old_log_probs = torch.gather(old_log_probs_dist_for_action, -1, action_tokens_ids.unsqueeze(-1)).squeeze(-1)
                    old_action_log_probs_sum = gathered_old_log_probs.sum().detach()

                    # c. Get "new" log_probs, "new" V(S_prev) and KL for the segment from current model
                    new_action_log_probs_sum, new_V_S_prev_recomputed, kl_for_action_sum = self._get_new_logprobs_value_and_kl_for_thought_segment(
                            self.model, self.ref_model, prompt_completion_ids,
                            segment_prompt_and_thought_start_idx,
                            segment_action_start_idx, segment_action_end_idx
                        )
                    
                    # d. Calculate Advantage
                    td_target_for_V_S_prev = R_curr_thought + self.rft_config.gamma * V_S_curr_old_for_td # TD target uses V_old from rollout
                    advantage_for_curr_thought = (td_target_for_V_S_prev - new_V_S_prev_recomputed).detach() # Advantage uses new V_S_prev prediction

                    # e. Policy Loss (PPO-like)
                    ratio = torch.exp(new_action_log_probs_sum - old_action_log_probs_sum)
                    pg_loss1 = -advantage_for_curr_thought * ratio
                    pg_loss2 = -advantage_for_curr_thought * torch.clamp(ratio, 1.0 - self.rft_config.clip_epsilon, 1.0 + self.rft_config.clip_epsilon)
                    step_policy_loss = torch.max(pg_loss1, pg_loss2)

                    # f. Value Loss for V_S_prev (PPO-like)
                    # `new_V_S_prev_recomputed` is the prediction. `td_target_for_V_S_prev` is the target.
                    # `V_S_prev_old` is for clipping.
                    vf_loss1_step = torch.square(new_V_S_prev_recomputed - td_target_for_V_S_prev.detach())
                    if self.rft_config.cliprange_value > 0:
                        V_S_prev_clipped = V_S_prev_old + torch.clamp(
                            new_V_S_prev_recomputed - V_S_prev_old,
                            -self.rft_config.cliprange_value,
                            self.rft_config.cliprange_value,
                        )
                        vf_loss2_step = torch.square(V_S_prev_clipped - td_target_for_V_S_prev.detach())
                        step_value_loss = 0.5 * torch.max(vf_loss1_step, vf_loss2_step)
                    else:
                        step_value_loss = 0.5 * vf_loss1_step
                    
                    # g. Total Loss for this thought step
                    step_kl_penalty_loss = self.rft_config.beta * kl_for_action_sum # beta is kl_coef
                    total_loss_for_thought_step = step_policy_loss + \
                                                  self.rft_config.vf_coef * step_value_loss + \
                                                  step_kl_penalty_loss
                    
                    current_item_total_loss += total_loss_for_thought_step.item() # For logging

                    # h. Backward Pass for THIS THOUGHT STEP (accumulated)
                    # Scale by 1/gradient_accumulation_steps for manual accumulation
                    actual_loss_to_backward = total_loss_for_thought_step / self.rft_config.gradient_accumulation_steps
                    self.accelerator.backward(actual_loss_to_backward)
                    self.state.num_backward_passes_accumulated += 1

                    # i. Optimizer Step if accumulation count is met
                    if self.state.num_backward_passes_accumulated % self.rft_config.gradient_accumulation_steps == 0:
                        if self.rft_config.max_grad_norm is not None and self.rft_config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.rft_config.max_grad_norm)
                        
                        self.optimizer.step()
                        # Check if LR scheduler needs to be stepped
                        if not self.accelerator.optimizer_step_was_skipped: # Only step if optimizer actually stepped
                            if self.lr_scheduler is not None:
                                self.lr_scheduler.step()
                        
                        self.optimizer.zero_grad()
                        self.state.global_step += 1
                        items_processed_in_accumulation_cycle = 0 # Reset for display counter
                        
                        # Log metrics per optimizer step
                        log_metrics = {
                            "loss": current_item_total_loss / (k_thought + 1) if (k_thought + 1) > 0 else 0, # Avg loss over thoughts in this item leading to optim step
                            "policy_loss_step": step_policy_loss.item(), # From last thought
                            "value_loss_step": step_value_loss.item(),   # From last thought
                            "kl_loss_step": step_kl_penalty_loss.item(), # From last thought
                            "learning_rate": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch + ( (self.state.global_step * self.rft_config.gradient_accumulation_steps) / len_dataloader if len_dataloader else 0)
                        }
                        self.log(log_metrics)

                        # Check for saving/evaluation based on global_step
                        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=None)
                        if self.control.should_save: 
                            self._save_checkpoint(self.model, trial) # Or appropriate save logic
                        if self.control.should_evaluate: 
                            self.evaluate() # Or appropriate eval logic

                # End of thoughts for an item
                items_processed_in_accumulation_cycle +=1
                # Callbacks after processing an entire item
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=None)

                if self.control.should_training_stop: 
                    break # Break from item loop

            # End of batch_iterator (epoch)
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            if self.control.should_training_stop: 
                break # Break from epoch loop
        
        # End of training
        self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial) # Final save