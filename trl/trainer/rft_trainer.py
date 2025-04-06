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


def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)




def evaluate_state(critic_prompt, cumulative_reasons, model="gpt-4o"):
    from openai import OpenAI
   
    client = OpenAI() 

    response = client.chat.completions.create(
        model=model,
        messages=critic_prompt,
        functions=[
            {
                "name": "eval_chain_of_thought",
                "description": "Function to evaluate and assign scores to each step in the chain of thought.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "overall_score": {
                            "type": "integer",
                            "description": """
                            A score from -100 to 100 representing the overall correctness and quality of the answer,
                            independent of the chain of thought.

                            be critical when comes to giving the scores
                            if the agent is just waffling such as rephrasing the previous chain of thought give a negative score
                            """
                        },
                        "chain_of_thought": {
                            "type": "array",
                            "description": "An array of objects containing string identifiers and integer scores.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "thought_id": {
                                        "type": "string",
                                        "description": "A unique ID for this thought."
                                    },
                                    "thought_score": {
                                        "type": "integer", 
                                        "description": "A score -100 to 100 score indicating how helpful this step is toward the goal."
                                    },
                                    "thought_progress": {
                                        "type": "integer", 
                                        "description": "A score -100 to to 100 score indicating how much progress was made since the last step. -100 if its just waffling, 100 if it made significant progress"
                                    },                                    
                                },
                                "required": ["thought_id", "thought_score","thought_progress"]
                            }
                        }
                    },
                    "required": ["overall_score", "chain_of_thought"]
                },
            }
        ],
    )
    try:
        data = json.loads(response.choices[0].message.function_call.arguments)
        
        # Enrich the steps with their scores and highlight data
        enriched_steps = []
        # Iterate only over the minimum length of the two lists
        num_steps = min(len(cumulative_reasons), len(data["chain_of_thought"]))
        for i in range(num_steps):
            enriched_steps.append({
                "id": f"thought_{i}",
                "txt": next(iter(cumulative_reasons[i].values())),
                "score": data["chain_of_thought"][i]['thought_score'],
                "progress": data["chain_of_thought"][i]['thought_progress'],
                
            })

        
        return {'overall_score':data['overall_score'] , 'steps':enriched_steps}
    except Exception as e:
        printc("Error: " + str(e), 'red')
        return None



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



def get_critic_prompt(question,cot, final_answer, solution):


    return [
        {
            "role": "system",
            "content": """
            You are an AI evaluator tasked with critically analyzing an agent's problem-solving Chain of Thought.

            Given:
            - A question or problem statement.
            - The agent's reasoning steps.
            - A final target or solution goal.

            Your objective:
            - Rate each reasoning step individually on a scale from -100 to 100:
            - **100**: Highly logical, precise, and significantly contributes toward achieving the final goal.
            - **0**: Neutral or slightly flawed reasoning; does not substantially help or hinder reaching the goal.
            - **-100**: Extremely illogical, incorrect, or severely detracts from reaching the goal.

            Evaluation Guidelines:
            - **Be critical and precise** when assigning scores. 
            - Assign negative scores when the agent:
            - Merely rephrases or restates previous reasoning without advancing the logic ("waffling").
            - Provides obvious or trivial reasoning steps that don't meaningfully progress toward the target.
            - Reserve scores approaching 100 only for reasoning that is exceptionally insightful and directly relevant to the goal.
            - Clearly justify your scores with brief explanations to provide transparency.
            If the provided final goal state includes its own rubric or scoring criteria, use it solely as reference criteria for evaluation; do not adopt its scoring system. Maintain the -100 to 100 scale exclusively.
            This applies to the overall score and the Chain of thought scores.
            Clearly justify your scores with brief explanations to provide transparency.
            Remember, thoughtful and nuanced evaluations are essential to accurately reflect the agent's reasoning quality.
 
            """,
        },
        {
            "role": "user",
            "content": f"""
                    QUESTION/START STATE: {question}
                    AGENT CHAIN OF THOUGHT: {cot}
                    AGENT GOAL STATE: {final_answer}
                    CORRECT GOAL STATE: {solution}
                """
        },
    ]

    
    
    



def process_rewards(overall,steps, config):

    # Compute average and standard deviation of thought scores for normalization.
    all_scores = [step.get("score", 0) for step in steps]
    avg_score = statistics.mean(all_scores) if all_scores else 0
    std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 1
    std_score = std_score if std_score != 0 else 1
    combined_scores = []

    for i, step in enumerate(steps):
        raw_score = step.get("score", 0)

        # Normalize the thought score using z-score normalization.
        norm_score = (raw_score - avg_score) / std_score
        # Adjust based on whether the normalized score is above or below average.
        if norm_score >= 0:
            norm_score *= config.boost_multiplier
        else:
            norm_score *= config.dampen_multiplier

        current_progress = step.get("progress", 0)
        # Calculate the average progress over a window of previous steps.
        if i == 0:
            previous_avg = current_progress  # For the first step, use current progress.
        else:
            window_size = config.window_size
            start_idx = max(0, i - window_size)
            previous_progress_values = [steps[j].get("progress", 0) for j in range(start_idx, i)]
            previous_avg = sum(previous_progress_values) / len(previous_progress_values)

        # Compute progress delta as the difference between current progress and the average of past progresses.
        progress_delta = current_progress - previous_avg

        # Compute a base progress effect using a piecewise function:
        # - Reward a positive change with a square-root scaling.
        # - Penalize a negative change with a logarithmic scale.
        if progress_delta > 0:
            pos_divisor = config.progress_positive_divisor
            base_progress_effect = math.sqrt(progress_delta) * (current_progress / pos_divisor)
        elif progress_delta < 0:
            neg_base = config.progress_negative_base
            neg_divisor = config.progress_negative_divisor
            base_progress_effect = -math.log1p(abs(progress_delta)) * ((neg_base - current_progress) / neg_divisor)
        else:
            base_progress_effect = 0

        # Amplify the progress effect by multiplying with a factor proportional to the raw thought score.
        progress_multiplier_divisor = config.progress_multiplier_divisor
        progress_multiplier = raw_score / progress_multiplier_divisor
        progress_effect = base_progress_effect * progress_multiplier
        # Combine the overall score with the normalized thought score and the enhanced progress effect.
        combined = overall + norm_score + progress_effect

        combined_scores.append({
            "id": step.get("id"),
            "txt": step.get("txt", ""),
            "combined_score": combined,
            "raw_score": raw_score,
            "normalized_score": norm_score,
            "previous_avg": previous_avg,
            "progress_delta": progress_delta,
            "base_progress_effect": base_progress_effect,
            "progress_multiplier": progress_multiplier,
            "progress_effect": progress_effect
        })
        
    return combined_scores




def split_cot(text, delim, threshold_factor=1.0):
    raw_entries = [entry.strip() for entry in text.split(delim)]
    fused_entries = []
    n = len(raw_entries)
    lengths = [len(entry) for entry in raw_entries]

    i = 0
    while i < n:
        entry = raw_entries[i]
        current_len = len(entry)
        # Compute lengths for all other entries (exclude the current one)
        others = lengths[:i] + lengths[i+1:]
        if others:
            avg_length = sum(others) / len(others)
            # Use sample standard deviation if there are at least two other entries
            std_dev = statistics.stdev(others) if len(others) > 1 else 0
        else:
            avg_length = current_len
            std_dev = 0

        # Define a threshold based on the average minus a multiple of the standard deviation
        threshold = max(avg_length - threshold_factor * std_dev, 0)

        # If the current entry's length is below the threshold and there's a next entry, fuse them
        if current_len < threshold and i + 1 < n:
            fused_entry = entry + delim + raw_entries[i + 1]
            fused_entries.append(fused_entry)
            i += 2 
        else:
            fused_entries.append(entry)
            i += 1

    return fused_entries




# =====================================================
# RFT Trainer Class
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
        # --- Argument Validation and Merging ---
        if not isinstance(rft_config, RFTConfig):
            raise TypeError("`rft_config` must be an instance of RFTConfig.")
        self.rft_config = rft_config # Store RFT specific settings

        # --- Load Model ---
        model_kwargs = rft_config.model_init_kwargs or {}
        if isinstance(model, str):
             model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)



        # --- Initialize Accelerator Early ---
        # `gradient_accumulation_steps` is read from `args` by Accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=rft_config.gradient_accumulation_steps)
        self.processing_class = processing_class

        # --- Attach Value Head to Model ---
        model_config = model.config
        if is_peft_available() and isinstance(model, PeftModel):
             model_config = model.base_model.model.config
        value_head = torch.nn.Linear(model_config.hidden_size, 1)
        with torch.no_grad():
            value_head.weight.data.normal_(mean=0.0, std=1/(model_config.hidden_size + 1))
            value_head.bias.data.zero_()
        setattr(model, 'value_head', value_head)

        # --- Handle PEFT ---
        self.is_peft_model = False
        if peft_config is not None:
             if not is_peft_available(): raise ImportError("PEFT not available.")
             model = get_peft_model(model, peft_config)
             self.is_peft_model = True

        # --- Handle Reference Model Source ---
        self.remote_ref_proxy = None
        self._ref_model_internal = None # Holds local model before preparation
        
        if rft_config.evalulate_state_func is None:
            rft_config.evalulate_state_func = evaluate_state

        if self.rft_config.remote_ref_model:
            if not self.rft_config.remote_ref_model_uri: raise ValueError("`remote_ref_model_uri` missing.")
            try:
                self.remote_ref_proxy = Pyro5.api.Proxy(self.rft_config.remote_ref_model_uri); self.remote_ref_proxy._pyroTimeout = 10; self.remote_ref_proxy._pyroBind()
                print(f"Connected to remote ref model: {self.rft_config.remote_ref_model_uri}")
            except Exception as e: raise ConnectionError(f"Remote ref connection failed: {e}") from e
            if ref_model is not None: print("Warning: Using remote ref, ignoring local `ref_model`.")
        elif self.is_peft_model: print("Using PEFT base model as reference.")
        elif ref_model is not None: # Local model provided
            if isinstance(ref_model, str): self._ref_model_internal = AutoModelForCausalLM.from_pretrained(ref_model, **model_kwargs)
            elif isinstance(ref_model, PreTrainedModel): self._ref_model_internal = ref_model
            else: raise TypeError("`ref_model` must be str or PreTrainedModel.")
        else: print("Warning: No reference model configured. KL penalty will be zero.")

        # --- Generation Config ---
        self.generation_config = GenerationConfig(
            max_new_tokens=self.rft_config.response_length, temperature=self.rft_config.temperature + 1e-7,
            top_k=self.rft_config.top_k, top_p=self.rft_config.top_p, do_sample=True,
            repetition_penalty=self.rft_config.repetition_penalty,
            pad_token_id=processing_class.pad_token_id, eos_token_id=processing_class.eos_token_id,
        )

        # --- Default Hook Functions ---
        if self.rft_config.critic_prompt_func is None: self.rft_config.critic_prompt_func = get_critic_prompt
        if self.rft_config.process_rewards_func is None: self.rft_config.process_rewards_func = process_rewards

        # --- Data Collator ---
        def data_collator(features): return features

        # --- Call Trainer's __init__ ---
        # Pass the correct argument name (`tokenizer`) to the parent class
        super().__init__(
            model=model,
            args=rft_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )
        # Note: self.tokenizer is now set by the parent Trainer class.

        # --- Prepare Local Reference Model ---
        self.ref_model = None
        if self._ref_model_internal is not None:
            # Ensure ref model dtype matches policy model dtype *before* preparation
            policy_model_dtype = next(self.model.parameters()).dtype
            if self._ref_model_internal.dtype != policy_model_dtype:
                print(f"Casting local reference model to {policy_model_dtype} before preparation.")
                self._ref_model_internal = self._ref_model_internal.to(dtype=policy_model_dtype)
            # Prepare using accelerator
            self.ref_model = self.accelerator.prepare_model(self._ref_model_internal, evaluation_mode=True)
            print(f"Local reference model prepared on device: {self.ref_model.device}")

        # --- Final check on value head dtype/device ---
        # Value head should be prepared along with self.model by super().__init__()
        if hasattr(self.model, 'value_head'):
             prepared_model = self.accelerator.unwrap_model(self.model)
             value_head_dtype = prepared_model.value_head.weight.dtype
             model_dtype = next(prepared_model.parameters()).dtype
             if value_head_dtype != model_dtype:
                 print(f"Casting value head to {model_dtype} post-preparation.")
                 # Ensure modification happens on the prepared model object
                 prepared_model.value_head.to(dtype=model_dtype)
             print(f"Value head attached. Final Dtype: {prepared_model.value_head.weight.dtype}, Device: {prepared_model.value_head.weight.device}")



    @contextmanager
    def _optional_no_grad(self, condition: bool):
        if condition:
            with torch.no_grad():
                yield
        else:
            yield


    def _compute_logprobs_and_states(self, model_instance, input_ids, attention_mask, get_hidden_states: bool):
        """Internal helper for local model forward pass."""
        # ... (Implementation remains the same as previous version) ...
        if model_instance is None: raise ValueError("_compute_logprobs_and_states called with None.")
        is_ref_computation = (model_instance == self.ref_model) or \
                             (self.is_peft_model and self.ref_model is None and model_instance == self.model)
        enable_grads = not is_ref_computation
        peft_adapter_manager = nullcontext()
        model_to_use = model_instance
        if self.is_peft_model and self.ref_model is None and is_ref_computation:
             unwrapped_main_model = self.accelerator.unwrap_model(self.model)
             if hasattr(unwrapped_main_model, 'disable_adapter'): peft_adapter_manager = unwrapped_main_model.disable_adapter()
             else: printc("PEFT base ref failed: Cannot disable adapter.", "red"); return None, None
        with self._optional_no_grad(not enable_grads), peft_adapter_manager:
            unwrapped_model = self.accelerator.unwrap_model(model_to_use)
            outputs = unwrapped_model(input_ids, attention_mask=attention_mask, output_hidden_states=get_hidden_states, return_dict=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if get_hidden_states and outputs.hidden_states else None
        shift_logits = logits[:, :-1, :].contiguous()
        full_log_probs = F.log_softmax(shift_logits, dim=-1)
        return full_log_probs, hidden_states


    def get_ref_log_probs(self, full_input_ids, full_attention_mask):
        """Computes reference log probabilities using the configured source."""
        # ... (Implementation remains the same as previous version) ...
        if self.remote_ref_proxy is not None:
            if full_input_ids.shape[0] != 1: raise NotImplementedError("Remote ref supports batch size 1 only.")
            try:
                ids_list=full_input_ids.squeeze(0).tolist(); mask_list=full_attention_mask.squeeze(0).tolist()
                logp_list = self.remote_ref_proxy.compute_logprobs(ids_list, mask_list)
                logp_tensor = torch.tensor(logp_list, dtype=torch.float32, device=self.accelerator.device)
                return logp_tensor.unsqueeze(0)
            except Exception as e: printc(f"Remote ref failed: {e}", "red"); return None
        elif self.is_peft_model and self.ref_model is None:
            log_probs, _ = self._compute_logprobs_and_states(self.model, full_input_ids, full_attention_mask, get_hidden_states=False)
            return log_probs
        elif self.ref_model is not None:
            log_probs, _ = self._compute_logprobs_and_states(self.ref_model, full_input_ids, full_attention_mask, get_hidden_states=False)
            return log_probs
        else: return None


    def get_full_kl_divergence(self, full_policy_log_probs, full_ref_log_probs, full_input_ids):
        """Calculates per-token KL divergence for the entire sequence."""
        # ... (Implementation remains the same as previous version) ...
        if full_ref_log_probs is None or full_policy_log_probs is None: return None
        if full_input_ids.shape[1] <= 1: return None
        if full_policy_log_probs.shape[1] != full_input_ids.shape[1] - 1 or full_ref_log_probs.shape[1] != full_input_ids.shape[1] - 1:
             printc(f"Shape mismatch KL calc", "red"); return None
        actual_next_tokens_indices = full_input_ids[:, 1:].contiguous().unsqueeze(-1)
        policy_token_log_probs = torch.gather(full_policy_log_probs, -1, actual_next_tokens_indices).squeeze(-1)
        ref_token_log_probs = torch.gather(full_ref_log_probs, -1, actual_next_tokens_indices).squeeze(-1)
        return torch.clamp(policy_token_log_probs - ref_token_log_probs, min=-30, max=30)

    def extract_step_values(self, full_sequence_tensor, full_input_ids, step_input_ids):
        """Utility to extract values corresponding to a step from a SHIFTED sequence tensor."""
        # ... (Implementation remains the same as previous version, maybe add more shape checks) ...
        if full_sequence_tensor is None or step_input_ids is None or step_input_ids.shape[1] == 0: return None
        seq_len, step_len = full_input_ids.shape[1], step_input_ids.shape[1]
        start_idx = seq_len - step_len - 1; end_idx = seq_len - 1
        if start_idx < 0 or start_idx >= full_sequence_tensor.shape[1]: return None
        end_idx = min(end_idx, full_sequence_tensor.shape[1])
        slice_len = end_idx - start_idx
        if slice_len <=0: return None
        try:
            if len(full_sequence_tensor.shape) == 3: step_values = full_sequence_tensor[:, start_idx:end_idx, :]
            elif len(full_sequence_tensor.shape) == 2: step_values = full_sequence_tensor[:, start_idx:end_idx]
            else: return None
        except IndexError: return None
        if step_values.shape[1] != step_len: pass # Allow shorter slice due to generation end
        return step_values



    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Main training loop for RFT. Performs one backward pass per item."""

        device = self.accelerator.device
        optimizer = self.optimizer # Optimizer created by Trainer
        lr_scheduler = self.lr_scheduler # Scheduler created by Trainer
        tokenizer = self.processing_class # Use tokenizer from processing_class
        model = self.model # PEFT model prepared by Trainer/Accelerator

        unwrapped_model_for_config = self.accelerator.unwrap_model(model)
        if not hasattr(unwrapped_model_for_config, 'config') or not hasattr(unwrapped_model_for_config.config, 'vocab_size'):
             raise ValueError("Could not determine model vocabulary size. Model config is missing or incomplete.")
        model_vocab_size = unwrapped_model_for_config.config.vocab_size
        print(f"Using Model Vocab Size: {model_vocab_size}")

        # --- Verify Trainable Parameters Visible to Optimizer ---
        # Check if the optimizer provided by the Trainer has parameters
        if not list(optimizer.param_groups[0]['params']):
             printc("ERROR: Optimizer has no parameters! PEFT setup might be incorrect.", "red")
             # Try printing model parameters again to see requires_grad status
             print("Model Parameters' requires_grad status:")
             for name, param in model.named_parameters():
                  if param.requires_grad:
                       print(f"  Trainable: {name} - Shape: {param.shape}")
             raise ValueError("Optimizer parameter list is empty. Cannot train.")
        else:
             print(f"Optimizer successfully found {len(list(optimizer.param_groups[0]['params']))} parameter tensors.")


        model.train() # Ensure model is in training mode
        train_dataloader = self.get_train_dataloader()

        num_train_optimization_steps = len(train_dataloader) // self.rft_config.gradient_accumulation_steps * int(self.rft_config.num_train_epochs)
        print(f"Total optimization steps: {num_train_optimization_steps}")

        if self.state is None: # Basic fallback state
            from transformers.trainer_utils import TrainOutput
            self.state = TrainOutput(global_step=0, training_loss=0.0, metrics={}).state
        if resume_from_checkpoint is None:
             self.state.global_step = 0

        for epoch in range(int(self.rft_config.num_train_epochs)):
            printc(f"Starting Epoch {epoch+1}/{int(self.rft_config.num_train_epochs)}", "yellow")

            for train_step, batch in enumerate(train_dataloader):
                # Reset gradients at the start of processing potentially accumulated item
                # This might be redundant if zero_grad is called correctly after step, but adds safety.
                # Avoid if gradient accumulation relies on *not* zeroing here. Usually zeroing happens *after* optimizer.step().
                # Let's rely on zero_grad after the optimizer step.

                if not isinstance(batch, list) or not batch: continue
                batch_item = batch[0]

                # --- 1 & 2. Prompt Setup & Validation ---
                try:
                    # (Same prompt setup and CPU validation as before)
                    question = batch_item[self.rft_config.question]
                    solution = batch_item[self.rft_config.answer]
                    messages = [{'role': 'user', 'content': self.rft_config.system_prompt + question}]
                    # Adjust chat template based on actual model requirements
                    # Example: Adding the user/assistant roles explicitly if needed by template
                    q_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
                    q_text += self.rft_config.b_think # Add start thinking marker text
                    
                    prompt_tokens_dict = tokenizer(
                        q_text, return_tensors="pt", padding=False, add_special_tokens=True,
                        max_length=self.rft_config.max_prompt_length, truncation=True
                    )
                    
                    input_ids_cpu = prompt_tokens_dict["input_ids"]
                    min_id, max_id = input_ids_cpu.min().item(), input_ids_cpu.max().item()
                    if min_id < 0 or max_id >= model_vocab_size:
                        printc(f"ERROR: Invalid token IDs in initial prompt (CPU) item {train_step}!", "red")
                        printc(f"  Min ID: {min_id}, Max ID: {max_id}, Vocab Size: {model_vocab_size}", "red")
                        continue 
                        
                    prompt_tokens = {k: v.to(device) for k, v in prompt_tokens_dict.items()}
                    prompt_length = prompt_tokens["input_ids"].size(1)
                except Exception as e:
                    printc(f"Error during prompt preparation item {train_step}: {e}", "red")
                    continue

                # --- 3. Generate full completion ---
                try:
                     # (Same generation logic with no_grad as before)
                    with torch.no_grad():
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        model_device = unwrapped_model.device 
                        prompt_tokens_model_device = {k: v.to(model_device) for k, v in prompt_tokens.items()}
                        
                        if self.generation_config.pad_token_id is None or self.generation_config.pad_token_id < 0 or self.generation_config.pad_token_id >= model_vocab_size:
                             safe_pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None and 0 <= tokenizer.eos_token_id < model_vocab_size else -1
                             if safe_pad_id != -1: self.generation_config.pad_token_id = safe_pad_id
                             else: raise ValueError(f"Invalid pad_token_id ({self.generation_config.pad_token_id})")

                        prompt_completion_ids = unwrapped_model.generate(
                            **prompt_tokens_model_device, generation_config=self.generation_config,
                        )
                        min_gen_id, max_gen_id = prompt_completion_ids.min().item(), prompt_completion_ids.max().item()
                        if min_gen_id < 0 or max_gen_id >= model_vocab_size:
                             printc(f"ERROR: Invalid generated IDs item {train_step}!", "red")
                             printc(f"  Range: [{min_gen_id}, {max_gen_id}], Vocab: {model_vocab_size}", "red")
                             continue 
                except Exception as gen_e:
                    printc(f"Generation failed item {train_step}: {gen_e}", "red")
                    continue 
                
                prompt_completion_ids = prompt_completion_ids.to(device)
                full_generated_attention_mask = torch.ones_like(prompt_completion_ids, device=device)

                # --- Compute full logprobs and KL ONCE per item ---
                # *** CRITICAL: Ensure this forward pass enables gradients for the policy model ***
                try:
                    # Policy forward pass - SHOULD have grads enabled
                    full_policy_log_probs, full_hidden_states = self._compute_logprobs_and_states(
                        model, prompt_completion_ids, full_generated_attention_mask, get_hidden_states=True # Set True if value head used
                    )
                    if full_policy_log_probs is None: 
                         printc(f"Policy forward pass failed or returned None item {train_step}.", "yellow")
                         continue
                    
                    # Check requires_grad immediately after policy forward pass
                    if not full_policy_log_probs.requires_grad:
                         printc(f"ERROR: full_policy_log_probs does NOT require grad after policy forward pass! item {train_step}", "red")
                         # Check if model was in eval mode?
                         printc(f"  Model training state: {model.training}", "red") 
                         # Check a PEFT parameter's grad status
                         lora_param_name = next((n for n, p in model.named_parameters() if p.requires_grad), None)
                         if lora_param_name:
                              printc(f"  Sample LoRA param ({lora_param_name}) requires_grad: {model.get_parameter(lora_param_name).requires_grad}", "red")
                         continue # Cannot proceed without grads here

                    # Reference forward pass - SHOULD have grads disabled (handled internally by get_ref_log_probs)
                    full_ref_log_probs = self.get_ref_log_probs(
                        prompt_completion_ids, full_generated_attention_mask
                    )

                    # KL calculation
                    full_per_token_kl = self.get_full_kl_divergence(
                        full_policy_log_probs, full_ref_log_probs, prompt_completion_ids
                    )
                    # KL divergence itself doesn't *need* grads from ref model, only uses policy grads

                except Exception as forward_e: 
                    printc(f"Error during forward/KL item {train_step}: {forward_e}", "red")
                    continue

                # --- 4, 5, 6. Decode, Split CoT, Split Steps ---
                try:
                    # (Same decoding and splitting logic as before)
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                    if completion_ids.shape[1] == 0: continue
                    min_comp_id, max_comp_id = completion_ids.min().item(), completion_ids.max().item()
                    if min_comp_id < 0 or max_comp_id >= model_vocab_size:
                        printc(f"ERROR: Invalid completion IDs item {train_step}!", "red")
                        continue 
                    full_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    split_result = full_text.split(self.rft_config.e_think, 1)
                    if len(split_result) >= 2: cot, answer = split_result[0].strip(), split_result[1].strip()
                    else: cot, answer = split_result[0].strip(), self.rft_config.answer_default
                    steps_text_raw = split_cot(cot, self.rft_config.delimiter)
                    if not steps_text_raw: continue
                except Exception as decode_e:
                     printc(f"Error during decoding/splitting item {train_step}: {decode_e}", "red")
                     continue

                # --- 7. Evaluate rewards & Whiten ---
                try:
                     # (Same reward calculation and whitening as before)
                    cumulative_reasons = [{f"thought_{i}": txt} for i, txt in enumerate(steps_text_raw)]
                    critic_prompt = self.rft_config.critic_prompt_func(question, cumulative_reasons, answer, solution)
                    eval_data = self.rft_config.evalulate_state_func(critic_prompt, cumulative_reasons)
                    if eval_data is None or 'steps' not in eval_data or not eval_data['steps']: continue
                    steps_data = self.rft_config.process_rewards_func(eval_data.get("overall_score", 0), eval_data['steps'], self.rft_config)
                    if not steps_data: continue

                    if self.rft_config.whiten_rewards and len(steps_data) > 0:
                        all_rewards = torch.tensor([s['combined_score'] for s in steps_data if 'combined_score' in s], device=device, dtype=torch.float32)
                        if len(all_rewards) > 1: 
                            mean, std = all_rewards.mean(), all_rewards.std()
                            whitened = (all_rewards - mean) / (std + 1e-8) if std > 1e-5 else (all_rewards - mean)
                        elif len(all_rewards) == 1: whitened = all_rewards - all_rewards.mean()
                        else: whitened = torch.tensor([], device=device, dtype=torch.float32) 
                        score_idx = 0
                        for i in range(len(steps_data)):
                            if 'combined_score' in steps_data[i] and score_idx < len(whitened):
                                w_score = whitened[score_idx].item()
                                if not (torch.isnan(torch.tensor(w_score)) or torch.isinf(torch.tensor(w_score))):
                                    steps_data[i]['whitened_score'] = w_score
                                score_idx += 1
                except Exception as e: 
                    printc(f"Reward/Eval error item {train_step}: {e}", "red")
                    continue 

                # --- 8. Accumulate Losses Over Steps ---
                all_policy_loss_terms = []
                all_kl_loss_terms = []
                all_value_losses = []
                total_steps_len_tokens = 0 
                perform_backward = False 
                total_loss = None 

                for step_idx, step_info in enumerate(steps_data):
                    try:
                        # (Same step tokenization, validation, value extraction as before)
                        reward_key = 'whitened_score' if self.rft_config.whiten_rewards and 'whitened_score' in step_info else 'combined_score'
                        if reward_key not in step_info: continue
                        step_reward = step_info[reward_key]
                        if torch.isnan(torch.tensor(step_reward)) or torch.isinf(torch.tensor(step_reward)): step_reward = 0.0
                        step_reward_tensor = torch.tensor(step_reward, device=device, dtype=torch.float32)

                        step_tokenized = tokenizer(step_info["txt"], return_tensors="pt", padding=False, add_special_tokens=False)
                        step_input_ids = step_tokenized['input_ids'].to(device) 
                        if step_input_ids.shape[1] == 0: continue 
                        min_step_id, max_step_id = step_input_ids.min().item(), step_input_ids.max().item()
                        if min_step_id < 0 or max_step_id >= model_vocab_size:
                             printc(f"ERROR: Invalid step IDs item {train_step}, step {step_idx}!", "red")
                             continue 

                        step_policy_log_probs_dist = self.extract_step_values(full_policy_log_probs, prompt_completion_ids, step_input_ids)
                        step_kl_div = self.extract_step_values(full_per_token_kl, prompt_completion_ids, step_input_ids) 

                        if step_policy_log_probs_dist is None or step_policy_log_probs_dist.shape[1] == 0: continue
                        actual_slice_len = step_policy_log_probs_dist.shape[1]
                        if actual_slice_len != step_input_ids.shape[1]:
                            step_input_ids = step_input_ids[:, :actual_slice_len]
                            if step_input_ids.shape[1] == 0: continue 
                        
                        # --- Check requires_grad before gather ---
                        if not step_policy_log_probs_dist.requires_grad:
                             # This should not happen if full_policy_log_probs had grads
                             printc(f"ERROR: step_policy_log_probs_dist does NOT require grad! item {train_step}, step {step_idx}", "red")
                             # Break inner loop for this item
                             all_policy_loss_terms = []; all_kl_loss_terms = []; all_value_losses = []; total_steps_len_tokens = 0
                             break 

                        step_indices_for_gather = step_input_ids.unsqueeze(-1)
                        min_gather_idx, max_gather_idx = step_indices_for_gather.min().item(), step_indices_for_gather.max().item()
                        policy_vocab_size = step_policy_log_probs_dist.shape[-1]
                        if min_gather_idx < 0 or max_gather_idx >= policy_vocab_size:
                              printc(f"ERROR: Invalid policy gather indices item {train_step}, step {step_idx}!", "red")
                              continue 
                        policy_log_probs = torch.gather(step_policy_log_probs_dist, -1, step_indices_for_gather).squeeze(-1)
                        
                        # --- Check requires_grad after gather ---
                        if not policy_log_probs.requires_grad:
                             # This indicates the graph connection was lost during gather or before
                             printc(f"ERROR: policy_log_probs does NOT require grad after gather! item {train_step}, step {step_idx}", "red")
                             all_policy_loss_terms = []; all_kl_loss_terms = []; all_value_losses = []; total_steps_len_tokens = 0
                             break

                        if step_kl_div is None: step_kl_div = torch.zeros_like(policy_log_probs)
                        elif step_kl_div.shape[1] != policy_log_probs.shape[1]: 
                             kl_len, pol_len = step_kl_div.shape[1], policy_log_probs.shape[1]
                             if kl_len > pol_len: step_kl_div = step_kl_div[:, :pol_len]
                             elif kl_len < pol_len: step_kl_div = F.pad(step_kl_div, (0, pol_len - kl_len), value=0.0)
                             if step_kl_div.shape[1] != policy_log_probs.shape[1]: step_kl_div = torch.zeros_like(policy_log_probs)
                        
                        # Value loss (remains disabled/simplified until indexing verified)
                        step_value_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                        # --- Add value calculation here if re-enabled and verified ---
                        
                        # Calculate and store loss terms
                        policy_loss_term_unreduced = -policy_log_probs * step_reward_tensor.unsqueeze(-1)
                        kl_loss_term_unreduced = step_kl_div * self.rft_config.beta
                        
                        # Check for NaN/Inf before appending
                        if torch.isnan(policy_loss_term_unreduced).any() or torch.isinf(policy_loss_term_unreduced).any(): continue
                        if torch.isnan(kl_loss_term_unreduced).any() or torch.isinf(kl_loss_term_unreduced).any(): 
                             kl_loss_term_unreduced = torch.zeros_like(kl_loss_term_unreduced)

                        all_policy_loss_terms.append(policy_loss_term_unreduced)
                        all_kl_loss_terms.append(kl_loss_term_unreduced)
                        all_value_losses.append(step_value_loss * self.rft_config.vf_coef) 
                        total_steps_len_tokens += actual_slice_len
                    except Exception as step_e:
                         printc(f"Error during step processing item {train_step}, step {step_idx}: {step_e}", "red")
                         # Optionally break inner loop if a step fails critically
                         # all_policy_loss_terms = []; all_kl_loss_terms = []; all_value_losses = []; total_steps_len_tokens = 0
                         # break 
                         continue # Skip to next step


                # --- Combine Accumulated Losses ---
                log_total_loss = 0.0; log_policy_loss = 0.0; log_kl_loss = 0.0; log_value_loss = 0.0

                if total_steps_len_tokens > 0 and all_policy_loss_terms: 
                    try:
                        total_policy_loss_terms = torch.cat(all_policy_loss_terms, dim=1) 
                        total_kl_loss_terms = torch.cat(all_kl_loss_terms, dim=1)       
                        avg_policy_loss = total_policy_loss_terms.mean()
                        avg_kl_loss = total_kl_loss_terms.mean()
                        avg_value_loss = torch.stack(all_value_losses).mean() if all_value_losses else torch.tensor(0.0, device=device)

                        # --- Check requires_grad before combining final loss ---
                        if not avg_policy_loss.requires_grad:
                             # KL and Value might not require grad, but policy loss MUST
                             printc(f"ERROR: avg_policy_loss does NOT require grad before final sum! item {train_step}", "red")
                             perform_backward = False
                        # Add checks for avg_kl_loss and avg_value_loss if they *should* have grads (e.g., if value head is active)
                        # elif avg_value_loss is active and not avg_value_loss.requires_grad: ...

                        elif torch.isnan(avg_policy_loss) or torch.isinf(avg_policy_loss) or \
                           torch.isnan(avg_kl_loss) or torch.isinf(avg_kl_loss) or \
                           torch.isnan(avg_value_loss) or torch.isinf(avg_value_loss):
                             printc(f"ERROR: NaN/Inf in final avg losses item {train_step}. Skipping backward.", "red")
                             perform_backward = False 
                        else:
                            total_loss = avg_policy_loss + avg_kl_loss + avg_value_loss
                            
                            # --- Final Check: Does total_loss require grad? ---
                            if total_loss.requires_grad:
                                perform_backward = True 
                                log_policy_loss = avg_policy_loss.item()
                                log_kl_loss = avg_kl_loss.item()
                                log_value_loss = avg_value_loss.item()
                                log_total_loss = total_loss.item()
                            else:
                                 printc(f"ERROR: Final total_loss does NOT require grad! item {train_step}", "red")
                                 printc(f"  avg_policy_loss.requires_grad: {avg_policy_loss.requires_grad}", "red")
                                 printc(f"  avg_kl_loss.requires_grad: {avg_kl_loss.requires_grad}", "red")
                                 printc(f"  avg_value_loss.requires_grad: {avg_value_loss.requires_grad}", "red")
                                 perform_backward = False


                    except Exception as loss_combine_e:
                        printc(f"Error combining losses item {train_step}: {loss_combine_e}", "red")
                        perform_backward = False 

                # --- Perform Backward Pass ---
                if perform_backward and total_loss is not None:
                    loss_to_backward = total_loss / self.rft_config.gradient_accumulation_steps
                    try:
                        # printc(f"Attempting backward for item {train_step} with loss {loss_to_backward.item()}", "green") # Verbose debug
                        self.accelerator.backward(loss_to_backward)
                    except RuntimeError as backward_e:
                         # The error message reported by the user happens here
                         printc(f"Error during backward pass for item {train_step}: {backward_e}", "red")
                         if "element 0 of tensors does not require grad" in str(backward_e):
                              printc("  Backward pass error: Graph disconnected or requires_grad=False.", "red")
                         elif "CUDA out of memory" in str(backward_e): printc("  CUDA OOM during backward pass.", "red")
                         # Reset logs, prevent optimizer step for this cycle if critical
                         log_total_loss = 0.0; log_policy_loss = 0.0; log_kl_loss = 0.0; log_value_loss = 0.0
                         perform_backward = False # Prevent logging/state update based on failed step
                         # No need to zero grads here, happens after optimizer step or cycle end
                    except Exception as backward_e_other: # Catch other potential errors
                         printc(f"Non-RuntimeError during backward pass for item {train_step}: {backward_e_other}", "red")
                         perform_backward = False

                # --- Gradient Update Step ---
                is_accumulation_step_complete = (train_step + 1) % self.rft_config.gradient_accumulation_steps == 0
                
                if is_accumulation_step_complete:
                    # Only step optimizer and scheduler if accumulation cycle is done
                    if self.rft_config.max_grad_norm is not None:
                        if self.accelerator.mixed_precision == "fp16" or self.accelerator.mixed_precision == "bf16":
                           self.accelerator.unscale_gradients(optimizer)
                        self.accelerator.clip_grad_norm_(model.parameters(), self.rft_config.max_grad_norm)
                    
                    try:
                        optimizer.step()
                        # print(f"Optimizer step at global_step {self.state.global_step + 1}") # Verbose debug
                    except Exception as optim_e:
                         printc(f"Error during optimizer.step() at train_step {train_step}: {optim_e}", "red")

                    if lr_scheduler is not None: 
                        with self.accelerator.main_process_first(): 
                             lr_scheduler.step()
                        
                    optimizer.zero_grad() # Zero grads *after* stepping

                    # --- Logging & State Update ---
                    self.state.global_step += 1 # Increment global step ONLY on optimizer step
                    
                    # Log only if the backward pass likely contributed meaningful gradients
                    # (Checking perform_backward ensures we log metrics from the last *successful* backward in the cycle)
                    # However, current logic logs metrics from the *last item*, regardless of backward success.
                    # Let's log if it's a logging step, but metrics might be from a failed-backward item.
                    if self.accelerator.is_main_process and (self.state.global_step % self.rft_config.logging_steps == 0):
                         logs = {"train/loss": log_total_loss, 
                                 "train/policy_loss": log_policy_loss,
                                 "train/kl_loss": log_kl_loss,
                                 "train/value_loss": log_value_loss, 
                                 "step": self.state.global_step,
                                 "epoch": round(epoch + ((train_step + 1) / len(train_dataloader)), 2)}
                         if lr_scheduler is not None:
                             lr_list = lr_scheduler.get_last_lr()
                             if lr_list: logs["train/learning_rate"] = lr_list[0]
                         self.log(logs)

                # --- Cleanup per item ---
                # (Same cleanup code using del as before)
                # ... [del statements] ...
                # Consider periodic gc.collect() / torch.cuda.empty_cache() if memory grows
                # if train_step > 0 and train_step % 50 == 0:
                #     import gc; gc.collect(); torch.cuda.empty_cache()


            # --- End of Epoch ---
            printc(f"Epoch {epoch+1} finished.", "blue")
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

        # --- End of Training ---
        printc("Training finished.", "blue")
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)



# end of file