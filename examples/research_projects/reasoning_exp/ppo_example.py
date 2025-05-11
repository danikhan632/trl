#!/usr/bin/env python3
"""
Fine-tune GPT2 to generate positive IMDB movie reviews via PPO, using a BERT sentiment classifier as reward.
"""
import argparse
import torch
from tqdm import tqdm
import pandas as pd
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GenerationConfig,
    DataCollatorWithPadding
)
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

def build_dataset(
    tokenizer, dataset_name: str, min_length: int, max_length: int
):
    """
    Load and preprocess the IMDB dataset: filter long reviews, tokenize, sample variable input lengths.
    """
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200)

    length_sampler = LengthSampler(min_length, max_length)

    def tokenize_fn(sample):
        ids = tokenizer.encode(sample["review"], add_special_tokens=False)
        cut = ids[: length_sampler()]
        sample["input_ids"] = cut
        sample["query"] = tokenizer.decode(cut)
        return sample

    ds = ds.map(tokenize_fn, remove_columns=["review"])
    ds.set_format(type="torch", columns=["input_ids"])
    return ds


def train(args):
    # 1. Build tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(
        tokenizer, args.dataset_name, args.input_min_length, args.input_max_length
    )

    # 2. Data collator for padding
    collator = DataCollatorWithPadding(tokenizer)

    # 3. Load policy & reference models
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    policy_model.generation_config = GenerationConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    policy_model.base_model_prefix = 'transformer'
    policy_model.transformer = policy_model.pretrained_model
    ref_model.base_model_prefix = 'transformer'


    # 4. Load reward (sentiment) model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name
    )
    
    #print(help(reward_model))

    # 5. Prepare PPO config
    config = PPOConfig()
    config.learning_rate = args.lr
    config.report_to = [args.log_with] if args.log_with else []

    # 6. Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=dataset,
        # optional args (you can omit any you don’t need, they’ll take these defaults)
        value_model=policy_model,
        data_collator=collator,
        eval_dataset=None,
        optimizers=(None, None),
        callbacks=None,
        peft_config=None,
    )



    # 9. Training loop
    ppo_trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT2 against positive sentiment"
    )
    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Base GPT2 model with IMDB fine-tuning",
    )
    parser.add_argument(
        "--reward_model_name", type=str, default="lvwerra/distilbert-imdb",
        help="HuggingFace model for sentiment rewards",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="stanfordnlp/imdb"
    )
    parser.add_argument(
        "--lr", type=float, default=1.41e-5,
        help="Learning rate for PPO optimizer",
    )
    parser.add_argument(
        "--log_with", type=str, default="wandb",
        help="Reporter integration (e.g. wandb, comet_ml)",
    )
    parser.add_argument("--input_min_length", type=int, default=2)
    parser.add_argument("--input_max_length", type=int, default=8)
    parser.add_argument("--output_min_length", type=int, default=4)
    parser.add_argument("--output_max_length", type=int, default=16)
    parser.add_argument("--sent_batch_size", type=int, default=16)
    parser.add_argument(
        "--output_dir", type=str, default="gpt2-imdb-pos-v2"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Whether to push model and tokenizer to HF Hub",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
