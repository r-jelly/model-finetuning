"""Fixed synthetic splits and exact-answer reward, shared by SFT and GRPO."""
import random
import re


def make_splits():
    rows = []
    for a in range(10, 100):
        for b in range(a, 100):
            rows.append({
                "id": f"{a}*{b}",
                "prompt": [{"role": "user", "content": f"What is {a} * {b}? Reply with only one integer."}],
                "completion": [{"role": "assistant", "content": str(a * b)}],
                "answer": a * b,
            })
    random.Random(42).shuffle(rows)
    return {"train": rows[:2000], "validation": rows[2000:2500], "test": rows[2500:3000]}


def exact_reward(completions, answer, **kwargs):
    if len(completions) != len(answer):
        raise ValueError("One answer is required for each completion.")
    rewards = []
    for completion, expected in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[0].get("content", "") if len(completion) == 1 and isinstance(completion[0], dict) else ""
        else:
            text = completion
        text = text.strip() if isinstance(text, str) else ""
        valid = re.fullmatch(r"[+-]?[0-9]{1,4}", text)
        rewards.append(float(bool(valid) and int(text) == expected))
    return rewards
