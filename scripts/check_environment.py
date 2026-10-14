"""Check package imports, CUDA and a tiny offline LoRA training step."""
import importlib.metadata
import math
import platform

import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, Trainer
from trl import SFTTrainer


def main():
    print(f"Python: {platform.python_version()}")
    for package in (
        "torch", "torchvision", "torchaudio", "transformers", "datasets",
        "accelerate", "peft", "trl", "huggingface-hub", "bitsandbytes",
    ):
        print(f"{package}: {importlib.metadata.version(package)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch build.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=128, n_positions=32, n_embd=32, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=1, pad_token_id=2,
    ))
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8,
        target_modules=["c_attn"], fan_in_fan_out=True,
    )).cuda()
    model.train()
    dataset = Dataset.from_dict({"input_ids": [[3, 4, 5, 6, 7, 8, 9, 1]]})
    tokens = torch.tensor(dataset[0]["input_ids"], device="cuda").unsqueeze(0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = model(input_ids=tokens, labels=tokens).loss
    assert math.isfinite(loss.item()), "Training loss is not finite."
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable)), "LoRA weights did not update."
    print(f"Offline LoRA training step: OK (loss={loss.item():.4f})")
    layer = bnb.nn.Linear4bit(64, 32, bias=False, quant_type="nf4", compute_dtype=torch.float16).cuda()
    result = layer(torch.randn(2, 64, device="cuda", dtype=torch.float16))
    assert torch.isfinite(result).all(), "NF4 output is not finite."
    print("bitsandbytes NF4 GPU operation: OK")
    print("Environment check: PASS")


if __name__ == "__main__":
    main()
