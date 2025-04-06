import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler, # Bring back get_scheduler
    TrainingArguments 
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training 
)
from trl.trainer.rft_trainer import RFTTrainer
from trl.trainer.rft_config import RFTConfig 
import os 

# --- Configuration ---
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
compute_dtype = torch.bfloat16 
attn_implementation = "eager" 
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Device setup
if not torch.cuda.is_available():
    raise SystemError("CUDA device not found, but script requires it.")
device = torch.device("cuda") 
print(f"Using device: {device}")
print(f"Using compute dtype: {compute_dtype}")
print(f"Using attention implementation: {attn_implementation}")

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    print("Tokenizer missing pad token, setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 

# --- Load Base Model ---
print(f"Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=compute_dtype, 
    attn_implementation=attn_implementation,
).to(device)

# --- Resize Embeddings ---
print(f"Checking and resizing token embeddings...")
print(f"  Tokenizer vocab size: {len(tokenizer)}")
print(f"  Model vocab size (before resize): {model.config.vocab_size}")
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Model vocab size (after resize): {model.config.vocab_size}")
    if model.config.vocab_size != len(tokenizer):
         raise RuntimeError("Failed to resize token embeddings correctly!")
else:
    print("  Vocab sizes already match.")

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=8,                      
    lora_alpha=32,            
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Apply LoRA/PEFT to the model ---
print("Applying LoRA PEFT adapter...")
# model = prepare_model_for_kbit_training(model) # If using quantization
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

# --- Load Reference Model ---
print(f"Loading reference model: {model_name}")
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=compute_dtype,
    attn_implementation=attn_implementation,
).to(device)

# --- Resize Reference Model Embeddings ---
print(f"Checking and resizing reference model token embeddings...")
print(f"  Reference Model vocab size (before resize): {ref_model.config.vocab_size}")
if ref_model.config.vocab_size != len(tokenizer):
    ref_model.resize_token_embeddings(len(tokenizer))
    print(f"  Reference Model vocab size (after resize): {ref_model.config.vocab_size}")
    if ref_model.config.vocab_size != len(tokenizer):
         raise RuntimeError("Failed to resize reference model token embeddings correctly!")
else:
    print("  Reference model vocab size matches tokenizer.")


# --- RFT Configuration ---
config = RFTConfig(
    output_dir="./rft_mystery_output", 
    num_train_epochs=1, 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    learning_rate=1e-4, # This LR will be used by scheduler, optimizer uses it initially
    logging_steps=10, 
    report_to="none", 
    remove_unused_columns=False, 
)

config.b_think = '<|user|>\n<|assistant|><think>' 
config.e_think = '</think>' 
config.delimiter = '\n' 
config.question = 'start_state'
config.answer = 'solution'
config.system_prompt = 'Solve the mystery:\n'

# --- Load Dataset ---
dataset = load_dataset("danikhan632/OpenMystery", split="train")
# dataset = dataset.shuffle(seed=42).select(range(1000)) # Optional subset

# --- Create Optimizer and Scheduler MANUALLY ---
# Ensure optimizer targets only trainable parameters (PEFT should handle requires_grad correctly)
# AdamW is generally recommended
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), # Filter for trainable params
    lr=config.learning_rate # Use the initial LR from config
    # Add weight_decay etc. if needed, matching Trainer defaults or your preference
    # weight_decay=config.weight_decay 
)

# Calculate total training steps for the scheduler
num_update_steps_per_epoch = len(dataset) // (config.gradient_accumulation_steps * config.per_device_train_batch_size * config.world_size) # world_size for distributed
max_steps = int(config.num_train_epochs * num_update_steps_per_epoch)

# Create scheduler (e.g., linear)
lr_scheduler = get_scheduler(
    name="linear", # Or "cosine", etc.
    optimizer=optimizer,
    num_warmup_steps=0, # Set warmup steps if desired (e.g., config.warmup_steps)
    num_training_steps=max_steps,
)

print(f"Manually created optimizer: {optimizer}")
print(f"Manually created scheduler: {lr_scheduler}")


# --- Initialize Trainer ---
# Pass the manually created optimizer and scheduler
print("Initializing RFTTrainer...")
trainer = RFTTrainer(
    model=model,                
    ref_model=ref_model,        
    rft_config=config,          
    train_dataset=dataset,
    processing_class=tokenizer, 
    peft_config=lora_config,    # Still pass peft_config for awareness if needed by Trainer logic
    optimizers=(optimizer, lr_scheduler), # Pass the created optimizer and scheduler
)

# --- Train the Model ---
print("Starting training...")
trainer.train()

print("Training finished.")
# trainer.save_model("./rft_mystery_output/final_adapter") 