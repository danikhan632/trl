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

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

from transformers import TrainingArguments


@dataclass
class RFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`RFTTrainer`].

    Only the parameters specific to RFT training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`RFTTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the required columns (`"prompt"`, `"question"`, `"answer"`) in the dataset. If you
            use a custom reward function or custom processing functions that require other columns, set this to `False`.
            Overwrites the default `True` from `TrainingArguments`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `1000`):
            Maximum length of the prompt sequence. If the prompt is longer than this value, it will be truncated (left-padded).
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations to sample per prompt during the data collection or initial sampling phase.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature used for sampling completions during the data collection or initial sampling phase.
            Higher values increase randomness.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum number of tokens to generate for each completion during the data collection or initial sampling phase.

        > Parameters that control generation acceleration powered by vLLM (for initial sampling)

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for accelerating the generation of completions (initial sampling phase). Requires `vllm`
            to be installed (`pip install vllm`). Ensure a GPU is available for vLLM if set to `True`.
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device for vLLM generation (e.g., `"cuda:1"`). If `"auto"`, selects the next available GPU after training
            devices, assuming not all GPUs are used for training.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Proportion (0 to 1) of GPU memory for vLLM's weights, activations, and KV cache on the generation device.
            Higher values improve throughput but risk OOM errors.

        > Parameters that control the RFT training loop (PPO-like)

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for the AdamW optimizer. Overwrites the default from `TrainingArguments`.
        per_device_train_batch_size (`int`, *optional*, defaults to `1`):
            Number of prompts processed per device during training. The effective batch size passed to the model's
            forward pass is `per_device_train_batch_size * num_generations`. Reduced default due to memory usage from multiple generations.
        gradient_accumulation_steps (`int`, *optional*, defaults to `8`):
            Number of update steps to accumulate gradients before performing a backward/update pass. Increased default
            to compensate for the smaller `per_device_train_batch_size`.
        beta (`float`, *optional*, defaults to `0.04`):
            Coefficient for the KL divergence penalty term in the RFT objective, balancing policy updates and deviation
            from the reference model.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clipping parameter for the policy gradient objective (PPO-style surrogate objective). Limits the change in
            the policy probability ratio.
        response_length (`int`, *optional*, defaults to `2000`):
            Maximum total length (prompt + response) for sequences generated by the policy model during training rollouts.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Coefficient for the value function loss term in the PPO objective. Scales the contribution of the value loss
            to the total loss.
        whiten_rewards (`bool`, *optional*, defaults to `True`):
            Whether to normalize rewards (whitening) before using them to compute advantages. Helps stabilize training.

        > Parameters related to adaptive reward scaling based on progress

        window_size (`int`, *optional*, defaults to `3`):
            Number of previous training steps to consider for averaging progress (used in adaptive reward scaling).
        boost_multiplier (`float`, *optional*, defaults to `1.2`):
            Multiplier applied to normalized scores that are above the recent average progress (adaptive reward scaling).
        dampen_multiplier (`float`, *optional*, defaults to `0.8`):
            Multiplier applied to normalized scores that are below the recent average progress (adaptive reward scaling).
        progress_positive_divisor (`int`, *optional*, defaults to `10`):
            Divisor used in scaling the effect of positive progress (scores above average) on rewards.
        progress_negative_base (`int`, *optional*, defaults to `100`):
            Base value used in the formula for scaling the effect of negative progress (scores below average) on rewards.
        progress_negative_divisor (`int`, *optional*, defaults to `10`):
            Divisor used in scaling the effect of negative progress (scores below average) on rewards.
        progress_multiplier_divisor (`int`, *optional*, defaults to `50`):
            Divisor used for amplifying the progress effect based on the raw score magnitude.

        > Parameters defining dataset keys, special tokens, and prompts

        question (`str`, *optional*, defaults to `"question"`):
            The key in the dataset dictionary corresponding to the input prompt or question.
        answer (`str`, *optional*, defaults to `"answer"`):
            The key in the dataset dictionary corresponding to the reference or expected answer (used potentially for reward or logging).
        b_think (`str`, *optional*, defaults to `"<think>"`):
            Special token or tag indicating the beginning of a "thinking" or internal monologue segment in the generated response.
        e_think (`str`, *optional*, defaults to `"</think>"`):
            Special token or tag indicating the end of a "thinking" segment in the generated response.
        system_prompt (`str`, *optional*, defaults to `"Answer the question:\n"`):
            The initial system prompt provided to the model before the user question/prompt.
        answer_default (`str`, *optional*, defaults to `"No answer provided"`):
            Default text used if an answer cannot be extracted from the model's generation (e.g., if thinking tags are present but no final answer).
        delimiter (`str`, *optional*, defaults to `"\n\n"`):
            Delimiter string used for separating different parts of text, potentially in prompts or responses.

        > Parameters for custom functions

        critic_prompt_func (`Callable` or `None`, *optional*, defaults to `None`):
            A function to format the input for a critic model, if one is used externally or implicitly for reward calculation.
            The function should typically take the prompt and/or completion as input.
        process_rewards_func (`Callable` or `None`, *optional*, defaults to `None`):
            A function to post-process the calculated rewards before they are used in the RFT objective. Can be used for
            custom scaling, clipping, or other reward transformations.

        > Parameters for remote reference model inference

        remote_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to use a remote inference endpoint for the reference model (used for KL divergence calculation)
            instead of loading it locally.
        remote_ref_model_uri (`str` or `None`, *optional*, defaults to `""`):
            The URI (e.g., HTTP endpoint) of the remote reference model inference service. Required if `remote_ref_model` is `True`.

        > Parameters controlling policy generation during training/evaluation

        temperature (`float`, *optional*, defaults to `0.7`):
            Temperature used for sampling responses from the policy model during training rollouts and evaluation.
            Controls randomness. Note: This is distinct from the temperature used for initial data sampling.
        top_k (`int`, *optional*, defaults to `0`):
            Filters token probabilities during policy generation, keeping only the top `k` tokens. `0` disables top-k filtering.
        top_p (`float`, *optional*, defaults to `1.0`):
            Filters token probabilities during policy generation using nucleus sampling. Keeps the smallest set of tokens whose
            cumulative probability exceeds `top_p`. `1.0` disables top-p filtering.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Penalty applied to token probabilities during policy generation to discourage repetition. `1.0` means no penalty.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `RFTTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to remove columns not explicitly needed for training (like 'prompt', 'question', 'answer'). "
            "Set to False if custom functions rely on other columns. Overwrites TrainingArguments default."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of completions to generate per prompt during initial data sampling."},
    )
    # This temperature is specifically for the initial sampling phase if applicable
    temperature_sampling: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling completions during initial data generation/sampling phase."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of tokens for generated completions during initial data sampling."},
    )

    # Parameters that control generation acceleration powered by vLLM (for initial sampling)
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use vLLM for accelerating initial completion generation. Requires `pip install vllm` and an available GPU."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device for vLLM generation (e.g., 'cuda:1'). 'auto' selects next available GPU after training devices."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "GPU memory utilization ratio (0-1) for vLLM. Higher values increase throughput but risk OOM errors."
        },
    )

    # Parameters that control the training loop (RFT/PPO)
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Initial learning rate for the AdamW optimizer (overwrites TrainingArguments default)."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Number of prompts per device for training. Effective batch size is this * num_generations."
            " Default lowered due to memory needs."
        },
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Accumulate gradients over this many steps before backpropagation. Default increased for larger effective batch size."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL divergence coefficient (penalty) in the RFT objective."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clipping range for the PPO surrogate objective ratio."},
    )
    response_length: int = field(
        default=2000,
        metadata={
            "help": "Maximum total sequence length (prompt + response) for policy generation during training rollouts."
        },
    )
    max_prompt_length: int = field( # Keep only one max_prompt_length definition
        default=1000,
        metadata={
            "help": "Maximum length of the prompt sequence. Truncated if longer (left-padded)."
        },
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient for the value function loss in the PPO objective."},
    )
    whiten_rewards: bool = field(
        default=True,
        metadata={"help": "Normalize (whiten) rewards before computing advantages."},
    )

    # Adaptive Reward Scaling Parameters
    window_size: int = field(
        default=3,
        metadata={"help": "Window size (number of steps) for averaging progress in adaptive reward scaling."},
    )
    boost_multiplier: float = field(
        default=1.2,
        metadata={"help": "Multiplier for normalized scores above average (adaptive reward scaling)."},
    )
    dampen_multiplier: float = field(
        default=0.8,
        metadata={"help": "Multiplier for normalized scores below average (adaptive reward scaling)."},
    )
    progress_positive_divisor: int = field(
        default=10,
        metadata={"help": "Divisor for scaling positive progress effects on rewards."},
    )
    progress_negative_base: int = field(
        default=100,
        metadata={"help": "Base value for scaling negative progress effects on rewards."},
    )
    progress_negative_divisor: int = field(
        default=10,
        metadata={"help": "Divisor for scaling negative progress effects on rewards."},
    )
    progress_multiplier_divisor: int = field(
        default=50,
        metadata={"help": "Divisor amplifying progress effect based on raw score magnitude."},
    )

    # Dataset Keys, Special Tokens, Prompts
    question: str = field(
        default="question",
        metadata={"help": "Dataset dictionary key for the input prompt/question."},
    )
    answer: str = field(
        default="answer",
        metadata={"help": "Dataset dictionary key for the reference answer."},
    )
    b_think: str = field(
        default="<think>",
        metadata={"help": "Token/tag marking the beginning of a thought segment."},
    )
    e_think: str = field(
        default="</think>",
        metadata={"help": "Token/tag marking the end of a thought segment."},
    )
    system_prompt: str = field(
        default="Answer the question:\n",
        metadata={"help": "System prompt prepended to the input question/prompt."},
    )
    answer_default: str = field(
        default="No answer provided",
        metadata={"help": "Default text used when no answer is extracted from the generation."},
    )
    delimiter: str = field(# Keep only one delimiter definition
        default="\n\n",
        metadata={"help": "Delimiter string used for separating text segments."},
    )

    # Custom Function Hooks
    critic_prompt_func: Optional[Callable] = field( # Correct type hint
        default=None,
        metadata={"help": "Optional function to format input for a critic model."},
    )
    process_rewards_func: Optional[Callable] = field( # Correct type hint
        default=None,
        metadata={"help": "Optional function to post-process rewards before RFT objective calculation."},
    )
    
    evalulate_state_func: Optional[Callable] = field( # Correct type hint
        default=None,
        metadata={"help": "Optional function to compute rewards."},
    )
    
    # Remote Reference Model Parameters
    remote_ref_model: bool = field(
        default=False,
        metadata={"help": "Use a remote inference endpoint for the reference model KL computation."},
    )
    remote_ref_model_uri: str = field(
        default="",
        metadata={"help": "URI of the remote reference model endpoint (required if remote_ref_model is True)."},
    )

    # Generation Parameters (for policy model during training/evaluation)
    temperature: float = field( # This temperature is for policy generation
        default=0.7,
        metadata={"help": "Temperature for policy model generation during training rollouts and evaluation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k filtering parameter for policy generation (`0` disables)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus (top-p) sampling parameter for policy generation (`1.0` disables)."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty parameter for policy generation (`1.0` disables)."},
    )

    output_dir: str = field(default="./rft_output", metadata={"help": "Output directory."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per device."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps to reduce memory usage."})
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    save_steps: int = field(default=500, metadata={"help": "Steps interval to save the model."})
    logging_dir: str = field(default="./logs", metadata={"help": "Logging directory."})
    logging_steps: int = field(default=100, metadata={"help": "Steps interval for logging."})
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy."})
    save_total_limit: int = field(default=1, metadata={"help": "Total limit of saved checkpoints."})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate."})
    debug_log: bool = field(
        default=False,
        metadata={"help": ""},
    )
    force_answer: str = field(
        default="",
        metadata={"help": ""},
    )