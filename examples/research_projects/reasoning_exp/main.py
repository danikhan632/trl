import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # get_scheduler # No longer needed if Trainer handles optimizer
    TrainingArguments # Base Trainer arguments might be useful if extending RFTConfig later
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training # Useful if quantization were used
)
from trl.trainer.rft_trainer import RFTTrainer
from trl.trainer.rft_config import RFTConfig  # RFT-specific training config
import os # Import os for environment variables if needed

# --- Configuration ---
# Set environment variable for CUDA behavior (optional, for debugging)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

# Choose precision: bfloat16 is recommended for stability on Ampere+ GPUs
# Use float32 if bf16 is not supported or issues persist
# Avoid float16 due to previous numerical instability issues
compute_dtype = torch.bfloat16 
# compute_dtype = torch.float32 

# Choose attention implementation: 'eager' is safer given the previous SWA/SDPA warning
# 'sdpa' might work with bf16/fp32 but 'eager' avoids the potential conflict
attn_implementation = "eager" 
# attn_implementation = "sdpa" 

# Model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Standard way
# Force CUDA as per original script, ensure it's available
if not torch.cuda.is_available():
    raise SystemError("CUDA device not found, but script requires it.")
device = torch.device("cuda") 
print(f"Using device: {device}")
print(f"Using compute dtype: {compute_dtype}")
print(f"Using attention implementation: {attn_implementation}")

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add padding token if missing (many models like Llama don't have one by default)
if tokenizer.pad_token is None:
    print("Tokenizer missing pad token, setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    # Make sure pad_token_id is set in the tokenizer object
    tokenizer.pad_token_id = tokenizer.eos_token_id 

# --- Load Base Model ---
print(f"Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=compute_dtype, 
    attn_implementation=attn_implementation,
    # trust_remote_code=True # Add if required by the model
).to(device)

# --- Resize Embeddings (Critical Fix) ---
print(f"Checking and resizing token embeddings...")
print(f"  Tokenizer vocab size: {len(tokenizer)}")
print(f"  Model vocab size (before resize): {model.config.vocab_size}")
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    # Verify resize
    print(f"  Model vocab size (after resize): {model.config.vocab_size}")
    if model.config.vocab_size != len(tokenizer):
         raise RuntimeError("Failed to resize token embeddings correctly!")
else:
    print("  Vocab sizes already match.")

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=8,                      # Rank of LoRA update matrices
    lora_alpha=32,            # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # Modify based on model architecture analysis (e.g., using print(model))
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Apply LoRA/PEFT to the model ---
print("Applying LoRA PEFT adapter...")
# If using quantization (not currently), uncomment prepare_model_for_kbit_training
# model = prepare_model_for_kbit_training(model) 
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Display the number of trainable parameters

# --- Load Reference Model ---
# Load the reference model WITHOUT PEFT adapter, using the same precision and attn implementation
print(f"Loading reference model: {model_name}")
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=compute_dtype,
    attn_implementation=attn_implementation,
    # trust_remote_code=True # Add if required
).to(device)

# --- Resize Reference Model Embeddings ---
# Important: Ensure ref_model also has the correct vocab size if it's used for logprob calculation
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
# Inherit from TrainingArguments if needed, or set directly
config = RFTConfig(
    output_dir="./rft_mystery_output", # Specify output directory
    num_train_epochs=1, # Example: Set number of epochs
    per_device_train_batch_size=1, # RFT often uses batch size 1 with grad accum
    gradient_accumulation_steps=8, # Accumulate grads over 8 steps
    learning_rate=1e-4, # Example LoRA learning rate
    logging_steps=10, # Log every 10 global steps
    # bf16=True if compute_dtype == torch.bfloat16 else False, # Let Trainer handle precision based on Accelerator
    # fp16=True if compute_dtype == torch.float16 else False, # Avoid fp16
    report_to="tensorboard", # Or "wandb", "none"
    remove_unused_columns=False, # Important for custom datasets in TRL
    # Add any other TrainingArguments needed
)

# RFT-specific settings from original script
config.b_think = '<|user|>\n<|assistant|><think>' # Example marker, adjust based on model's chat template
config.e_think = '</think>' # Example marker
config.delimiter = '\n' # Example step delimiter
config.question = 'start_state'
config.answer = 'solution'
config.system_prompt = 'Solve the mystery:\n'
# config.response_length = 512 # Max new tokens to generate in RFT loop
# config.max_prompt_length = 512 # Max length of the initial prompt

# --- Load Dataset ---
# Load dataset (after tokenizer is ready, if preprocessing depends on it)
dataset = load_dataset("danikhan632/OpenMystery", split="train")
# Optional: Shuffle or select subset
# dataset = dataset.shuffle(seed=42).select(range(1000)) 

# --- Initialize Trainer ---
# Let the Trainer handle optimizer and scheduler creation based on config and PEFT setup
print("Initializing RFTTrainer...")
trainer = RFTTrainer(
    model=model,                # Pass the PEFT-adapted model
    ref_model=ref_model,        # Pass the base reference model
    rft_config=config,          # Pass the RFTConfig object (acts as TrainingArguments)
    train_dataset=dataset,
    processing_class=tokenizer, # Pass the tokenizer instance
    peft_config=lora_config,    # Pass the LoRA config for Trainer awareness
    # optimizers=(optimizer, scheduler), # REMOVED: Let Trainer handle this
)

# --- Train the Model ---
print("Starting training...")
trainer.train()

print("Training finished.")
# Optionally save the final adapter
# trainer.save_model("./rft_mystery_output/final_adapter") 