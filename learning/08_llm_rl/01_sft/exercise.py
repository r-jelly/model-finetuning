"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arithmetic import make_splits, exact_reward
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = ROOT / "checkpoints/08_llm_rl/sft"
SEED = 42
MAX_STEPS = 50

def load_model():
    """TODO: tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요."""
    raise NotImplementedError('tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요.')


def evaluate(model, tokenizer, rows):
    """TODO: model.eval() 후 greedy 생성의 입력 prefix를 제외한 completion을 exact_reward로 평가하세요."""
    raise NotImplementedError('model.eval() 후 greedy 생성의 입력 prefix를 제외한 completion을 exact_reward로 평가하세요.')


def train(model, tokenizer, rows):
    """TODO: prompt/completion 데이터와 SFTConfig를 연결하고 adapter와 tokenizer를 저장하세요."""
    raise NotImplementedError('prompt/completion 데이터와 SFTConfig를 연결하고 adapter와 tokenizer를 저장하세요.')


def check_environment():
    splits = make_splits()
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [2000, 500, 500]
    config = SFTConfig(output_dir=str(CHECKPOINT_DIR), max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=4, max_length=128, completion_only_loss=True, report_to="none", push_to_hub=False, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    assert config.max_length == 128 and config.completion_only_loss
    assert exact_reward(["3", "wrong"], [3, 3]) == [1.0, 0.0]
    assert BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16).load_in_4bit
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    splits = make_splits()
    model, tokenizer = load_model()
    print("original", evaluate(model, tokenizer, splits["validation"]))
    train(model, tokenizer, splits["train"])
    print("SFT", evaluate(model, tokenizer, splits["validation"]))
    # adapter와 tokenizer를 저장하고 같은 greedy 결과로 재로딩을 검증하세요.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="다운로드 없는 환경 확인")
    mode.add_argument("--run", action="store_true", help="직접 작성한 학습 코드 실행")
    args = parser.parse_args()
    if args.check:
        check_environment()
    else:
        main()
