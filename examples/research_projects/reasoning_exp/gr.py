import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
import random
from trl.extras.rft_utils import printc
def generate_r1_prompt(nums, target, tokenizer):
    """Wraps a Countdown sample in an R1-style think/answer prompt."""
    r1_prefix = [
        {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer."},
        {"role": "user", "content": f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic (+, -, *, /) and each number only once. Show your work in <think></think> tags, then return your final equation in <answer></answer>."},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
    ]
    prompt = tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)

    return {"prompt": prompt, "target": target, "nums": nums}

def format_reward_func(completions, **kwargs):
    """1. Checks for proper <think>… / <answer>…</answer> format."""
    rewards = []
    regex = r"^<think>([\s\S]*?)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    for out in completions:
        text = "<think>" + out  # synthetic prepend
        rewards.append(1.0 if re.search(regex, text, re.DOTALL) else 0.0)
    return rewards

def equation_reward_func(completions, target, nums, **kwargs):
    """2. Verifies equation uses all nums exactly once and equals target."""
    printc(completions)
    rewards = []
    for out, gt, numbers in zip(completions, target, nums):
        text = "<think>" + out
        m = re.search(r"<answer>(.*?)<\/answer>", text)
        printc(out, 'red')
        foo = random.random()
        printc(foo,'green')
        rewards.append(foo)
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
ds = load_dataset(dataset_name, split="train")
ds = ds.shuffle(seed=42).select(range(50_000))

tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = ds.map(
    lambda x: generate_r1_prompt(x["nums"], x["target"], tokenizer),
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

tr_args = GRPOConfig(
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
    max_prompt_length=256,
    max_completion_length=1024,
    num_generations=num_generations,
    beta=beta,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_func, equation_reward_func],
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
