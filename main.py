import requests
import json
from datasets import load_dataset

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


def chat_completion(model: str, messages: list):
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return {"role": "assistant", "content": completion.choices[0].message}


def gen_rubric(entry) -> str:
    return f"""You are an expert medical educator.
Create a concise rubric to evaluate the quality of the following answer.
THE RUBRIC SHOULD HAVE A SCORE RANGE OF -100 TO 100

Question:
{entry['Question']}

Answer:
{entry['Response']}

Rubric:
- """

# load the dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

# list to collect outputs
data = []

for entry in dataset["train"]:
    messages = [
        {"role": "user", "content": gen_rubric(entry)}
    ]
    resp = chat_completion("microsoft/Phi-4-reasoning", messages)
    if resp is None:
        # skip this entry if the call failed
        continue

    # store question, answer, and generated rubric
    data.append({
        "Question": entry["Question"],
        "Response": entry["Response"],
        "Rubric": resp["content"].strip()
    })

# finally, write out to JSON
with open("rubrics.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(data)} rubrics and wrote to rubrics.json")

