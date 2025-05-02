import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler
)
from trl.trainer.rft_trainer import RFTTrainer
from trl.trainer.rft_config import RFTConfig


def setup_ddp():
    """
    Initialize distributed process group and set device.
    Launch this script via:
        torchrun --nproc_per_node=<NUM_GPUS> script.py
    If no distributed env vars are found, run in single-GPU mode.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), dist.get_rank()
    else:
        # Single-GPU fallback
        return 0, 1, 0


def main():
    # DDP Setup (or single-GPU)
    local_rank, world_size, rank = setup_ddp()
    device = torch.device("cuda", local_rank if world_size > 1 else 0)

    # Model & Tokenizer setup
    model_name = "Qwen/Qwen3-0.6B"
    compute_dtype = torch.bfloat16
    attn_impl = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl
    ).to(device)

    # Reference model for RFT
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl
    ).to(device)

    # RFT configuration
    rft_config = RFTConfig(
        output_dir="./rft_mystery_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    rft_config.b_think = '<|user|>\n<|assistant|><think>'
    rft_config.e_think = '</think>'
    rft_config.delimiter = '\n'
    rft_config.question = 'start_state'
    rft_config.answer = 'solution'
    rft_config.system_prompt = 'Solve the mystery:\n'

    # Dataset + sampler + dataloader
    dataset = load_dataset("danikhan632/OpenMystery", split="train")
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            dataset,
            batch_size=rft_config.per_device_train_batch_size,
            sampler=sampler,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=rft_config.per_device_train_batch_size,
            shuffle=True,
            drop_last=True
        )

    # Optimizer & LR scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=rft_config.learning_rate
    )
    total_steps = (len(train_loader) // rft_config.gradient_accumulation_steps) * rft_config.num_train_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Wrap model in DDP if needed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0:
            print(f"Training on {world_size} GPUs")
        trainer_model = model.module
    else:
        print("Running on a single GPU or CPU")
        trainer_model = model

    # Initialize and run trainer
    trainer = RFTTrainer(
        model=trainer_model,
        ref_model=ref_model,
        rft_config=rft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train()

    # Save only on rank 0 or single-GPU
    if rank == 0:
        trainer.save_model("./rft_mystery_output/final_model")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()