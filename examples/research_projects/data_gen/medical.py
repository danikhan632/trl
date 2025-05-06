#!/usr/bin/env python3
import json
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

def make_prompt(question: str, solution: str) -> str:
    return f"""You are an expert medical educator.
Create a concise rubric to evaluate the quality of the following answer.
THE RUBRIC SHOULD HAVE A SCORE RANGE OF -100 TO 100
Question:
{question}

Answer:
{solution}

Rubric:
- """

def generate_rubrics(
    df: pd.DataFrame,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    output_path: str,
):
    # Ensure a clean 0…N-1 integer index
    df = df.reset_index(drop=True)

    with open(output_path, "w") as fout:
        total = len(df)
        for start in range(0, 100, batch_size):
            batch_df = df.iloc[start : start + batch_size]

            # Build prompts
            requests = [
                {
                    "id": int(idx),  # ensure it's an int
                    "prompt": make_prompt(row["Question"], row["Response"])
                }
                for idx, row in batch_df.iterrows()
            ]

            # One-shot generate
            responses = llm.generate(requests, sampling_params=sampling_params)

            for resp in responses:
                # Cast the returned request_id to int for positional indexing
                i = int(resp.request_id)
                raw = resp.outputs[0].text.strip()
                if not raw.startswith("-"):
                    raw = "- " + raw
                bullets = [b.strip() for b in raw.split("\n- ") if b.strip()]

                # Positional lookup via iloc
                question = df.iloc[i]["Question"]

                fout.write(json.dumps({
                    "id": i,
                    "question": question,
                    "rubric": bullets
                }, ensure_ascii=False) + "\n")

            end = min(start + batch_size - 1, total - 1)
            print(f"Processed examples {start}–{end} / {total}")

    print(f"\nAll done: rubrics written to {output_path}")

def main():
    # 1) Load train split
    ds = load_dataset(
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "en",
        split="train"
    )

    # 2) Convert to DataFrame
    df = ds.to_pandas()
    print(f">>> Loaded {len(df)} examples into DataFrame")

    # 3) Initialize vLLM
    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        gpu_memory_utilization=0.8,
        max_num_seqs=8,
    )
    sampling_params = SamplingParams(
        max_tokens=150,
        temperature=0.2,
        top_p=0.9,
    )

    # 4) Generate
    generate_rubrics(
        df=df,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=4,
        output_path="medical_rubrics.jsonl",
    )
    



if __name__ == "__main__":
    main()
