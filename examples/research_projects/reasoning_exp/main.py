import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl.trainer.rft_trainer import RFTTrainer
from trl.trainer.rft_config import RFTConfig  # RFT-specific training config

# Force CPU execution
device = torch.device("cuda")

# Load dataset
dataset = load_dataset("danikhan632/OpenMystery", split="train")


# Model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model on CPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# LoRA Configuration (without quantization)
lora_config = LoraConfig(
    r=8,                      # Rank of LoRA update matrices
    lora_alpha=32,            # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # Modify based on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

config = RFTConfig()
config.b_think='<｜Assistant｜><think>'
config.question = 'start_state'
config.answer = 'solution'
config.system_prompt= 'Solve the mystery:\n'

# Apply LoRA to the model
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()  # Display the number of trainable parameters

# Load the reference model (for KL regularization, needed by RFT)
ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)


# Define optimizer (only trains LoRA parameters)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

# Define learning rate scheduler
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,  # Adjust as needed
    num_training_steps=len(dataset) * config.num_train_epochs,
)




# Initialize the trainer
trainer = RFTTrainer(
    model=model,
    ref_model=ref_model,
    rft_config=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, scheduler),
)

# Train the model
trainer.train()
