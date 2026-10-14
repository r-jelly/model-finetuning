"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import GPT2Config, GPT2LMHeadModel

SEED = 42
MODEL_ID = "openai-community/gpt2"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
TOKEN_BUDGET = 200_000
PACKING = False
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/06_continued_pretraining"

def prepare_data(tokenizer):
    """TODO: 학습/검증/평가 분할을 유지하고 training blocks와 고정 평가 blocks를 반환하세요."""
    raise NotImplementedError('학습/검증/평가 분할을 유지하고 training blocks와 고정 평가 blocks를 반환하세요.')


def train(model, dataset):
    """TODO: 같은 유효 토큰 예산을 소비하도록 causal LM 학습을 작성하세요."""
    raise NotImplementedError('같은 유효 토큰 예산을 소비하도록 causal LM 학습을 작성하세요.')


def evaluate(model, dataset):
    """TODO: 유효 토큰 수로 가중한 loss, perplexity를 반환하세요."""
    raise NotImplementedError('유효 토큰 수로 가중한 loss, perplexity를 반환하세요.')


def check_environment():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=1, n_head=4, bos_token_id=0, eos_token_id=1, pad_token_id=2))
    tokens = torch.tensor([[2, 3, 4, 5]])
    labels = tokens.clone()
    labels[:, -1] = -100
    loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=labels).loss
    assert torch.isfinite(loss) and math.isfinite(math.exp(loss.item()))
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_data, validation_data, test_data = prepare_data(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    print("before", evaluate(model, validation_data))
    train(model, train_data)
    print("after", evaluate(model, validation_data))
    # PACKING을 바꿔 새 모델로 비교하세요. 최종 test_data는 동일하게 유지합니다.


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
