"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForQuestionAnswering

SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "rajpurkar/squad"
USE_OVERFLOW = False
MAX_LENGTH = 256
DOC_STRIDE = 64
MAX_ANSWER_LENGTH = 30
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/05_question_answering"

def prepare_data():
    """TODO: context 단위의学習/검증/평가 질문と tokenizer를 반환하세요."""
    raise NotImplementedError('context 단위의学習/검증/평가 질문と tokenizer를 반환하세요.')


def tokenize_examples(examples, tokenizer):
    """TODO: 문자 offset을 토큰 start/end로 변환하고 overflow window 정보를 보존하세요."""
    raise NotImplementedError('문자 offset을 토큰 start/end로 변환하고 overflow window 정보를 보존하세요.')


def train(model, tokenizer, dataset):
    """TODO: QA head와 인코더를 학습하고 저장하세요."""
    raise NotImplementedError('QA head와 인코더를 학습하고 저장하세요.')


def evaluate(model, tokenizer, examples):
    """TODO: 질문별 window 결과를 합쳐 EM/F1과 정답 포함 비율을 반환하세요."""
    raise NotImplementedError('질문별 window 결과를 합쳐 EM/F1과 정답 포함 비율을 반환하세요.')


def check_environment():
    config = DistilBertConfig(vocab_size=64, dim=32, hidden_dim=64, n_layers=1, n_heads=4)
    model = DistilBertForQuestionAnswering(config)
    output = model(input_ids=torch.ones(2, 8, dtype=torch.long), start_positions=torch.tensor([1, 2]), end_positions=torch.tensor([2, 3]))
    assert output.start_logits.shape == (2, 8) and torch.isfinite(output.loss)
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    train_examples, validation_examples, test_examples, tokenizer = prepare_data()
    train_features = tokenize_examples(train_examples, tokenizer)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    train(model, tokenizer, train_features)
    print(evaluate(model, tokenizer, validation_examples))
    # USE_OVERFLOW를 바꿔 독립적인 새 모델로 비교하세요. test_examples는 마지막에만 평가합니다.


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
