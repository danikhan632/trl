import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from trl import XFTConfig, XFTTrainer
import random
from trl.extras.rft_utils import printc
import sys




def generate_r1_prompt(input, tokenizer):
    """Wraps a Countdown sample in an R1-style think/answer prompt."""
    r1_prefix = [
        {"role": "system", "content": """
         You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.
        when you solve the problem you should show your chain of thought. 
        Also each thought in the chain of thought should have a standard delimiter like two new lines.
        """},
        {"role": "user", "content": f"Solve the coding question:\n{input}"},
    ]
    prompt = tokenizer.apply_chat_template(r1_prefix, tokenize=False,add_generation_prompt=True)
    prompt+="Let me solve this step by step.\n\n<think>\n\n"
    # printc(prompt,'red')
    # sys.exit(0)
    return {"prompt": prompt}


def equation_reward_func(completions, **kwargs):
    printc(completions)
    rewards = []
    for out in completions:

        rewards.append(0.0)
    return rewards

# ─── Configuration ────────────────────────────────────────────────────────────
dataset_name     = "Jiayi-Pan/Countdown-Tasks-3to4"
model_name       = "Qwen/Qwen3-0.6B"
output_dir       = "qwen-r1-aha-moment"
max_steps        = 100
batch_size       = 2
num_generations  = 2
learning_rate    = 5e-7
beta             = 0.001
push_to_hub      = False
hub_token        = None
# ──────────────────────────────────────────────────────────────────────────────

# Optional: log in to HF hub if you want to push later
if push_to_hub:
    if not hub_token:
        raise ValueError("Please provide a valid `hub_token` to push to Hugging Face Hub")
    login(token=hub_token, add_to_git_credential=True)

# 1. Load & preprocess
ds = load_dataset("nvidia/OpenCodeReasoning", 'split_0')['split_0']
ds = ds.shuffle(seed=42).select(range(50_000))

tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = ds.map(
    lambda x: generate_r1_prompt(x["input"], tokenizer),
    remove_columns=ds.column_names,
)

split = ds.train_test_split(test_size=0.1)
train_ds, eval_ds = split["train"], split["test"]

# 2. Define model + training configs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).to(device)

tr_args = XFTConfig(
    output_dir=output_dir,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=max_steps,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    max_prompt_length=1024,
    max_completion_length=2048,
    num_generations=num_generations,
    beta=beta,
)

trainer = XFTTrainer(
    model=model,
    reward_funcs=[ equation_reward_func],
    args=tr_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# 3. Train!
trainer.train()

# 4. Save & (optionally) push
trainer.save_model(output_dir)
if push_to_hub:
    trainer.push_to_hub(commit_message="Mini-R1 aha moment checkpoint")
