"""Fixed GSM8K splits and final-answer reward, shared by all distillation stages."""
import random
import re

from datasets import load_dataset

INSTRUCTION = "Solve the problem step by step. End with a final line '#### <integer>'."


def extract_answer(text):
    """Return the integer on a last line '#### N', or None."""
    lines = text.strip().splitlines() if isinstance(text, str) else []
    match = re.fullmatch(r"####\s*([+-]?\d[\d,]*)\.?", lines[-1].strip()) if lines else None
    return int(match.group(1).replace(",", "")) if match else None


def make_splits():
    dataset = load_dataset("openai/gsm8k", "main")
    splits = {}
    for name, source, take in (("train", "train", slice(0, 1000)), ("validation", "train", slice(1000, 1200)), ("test", "test", slice(0, 500))):
        indices = list(range(len(dataset[source])))
        random.Random(42).shuffle(indices)
        splits[name] = [{
            "id": f"{source}-{i}",
            "prompt": [{"role": "user", "content": f"{dataset[source][i]['question']}\n\n{INSTRUCTION}"}],
            "answer": extract_answer(dataset[source][i]["answer"]),
        } for i in indices[take]]
    return splits


def exact_reward(completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("One answer is required for each completion.")
    rewards = []
    for completion, expected in zip(completions, answer):
        if isinstance(completion, list):
            completion = completion[0].get("content", "") if len(completion) == 1 and isinstance(completion[0], dict) else ""
        rewards.append(float(extract_answer(completion) == expected))
    return rewards
