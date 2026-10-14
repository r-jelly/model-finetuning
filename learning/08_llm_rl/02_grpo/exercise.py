"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arithmetic import make_splits, exact_reward
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[3]
SFT_CHECKPOINT = ROOT / "checkpoints/08_llm_rl/sft"
OUTPUT_DIR = ROOT / "outputs/08_llm_rl/grpo"
SEED = 42
MAX_STEPS = 20
NUM_GENERATIONS = 2

def load_sft_model():
    """TODO: 원본 모델에 저장된 SFT adapter를 is_trainable=True로 로드하고 tokenizer도 반환하세요."""
    raise NotImplementedError('원본 모델에 저장된 SFT adapter를 is_trainable=True로 로드하고 tokenizer도 반환하세요.')


def inspect_rewards(model, tokenizer, rows):
    """TODO: 학습 전에 요청별 2개 답변을 생성해 평균 보상과 그룹 내 보상 차이를 확인하세요."""
    raise NotImplementedError('학습 전에 요청별 2개 답변을 생성해 평균 보상과 그룹 내 보상 차이를 확인하세요.')


def train(model, tokenizer, rows):
    """TODO: prompt/answer 데이터와 exact_reward, GRPOConfig를 연결하세요. completion 정답을 입력에 넣지 마세요."""
    raise NotImplementedError('prompt/answer 데이터와 exact_reward, GRPOConfig를 연결하세요. completion 정답을 입력에 넣지 마세요.')


def evaluate(model, tokenizer, rows):
    """TODO: SFT와 같은 greedy 생성으로 미학습 문제의 정답률을 반환하세요."""
    raise NotImplementedError('SFT와 같은 greedy 생성으로 미학습 문제의 정답률을 반환하세요.')


def check_environment():
    config = GRPOConfig(output_dir=str(OUTPUT_DIR), max_steps=2, per_device_train_batch_size=2, gradient_accumulation_steps=2, num_generations=NUM_GENERATIONS, max_completion_length=8, use_vllm=False, beta=0.0, report_to="none", push_to_hub=False, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    assert config.num_generations == 2 and not config.use_vllm
    assert exact_reward([[{"role": "assistant", "content": "3"}]], [3]) == [1.0]
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    if not (SFT_CHECKPOINT / "adapter_config.json").is_file():
        raise FileNotFoundError("01_sft 학습 후 adapter를 checkpoints/08_llm_rl/sft에 저장하세요.")
    splits = make_splits()
    model, tokenizer = load_sft_model()
    inspect_rewards(model, tokenizer, splits["train"][:8])
    print("SFT", evaluate(model, tokenizer, splits["validation"]))
    train(model, tokenizer, splits["train"])
    print("GRPO", evaluate(model, tokenizer, splits["validation"]))


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
