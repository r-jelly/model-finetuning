"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "stanfordnlp/sst2"
MODE = "frozen"  # frozen -> full -> lora
MAX_LENGTH = 128
BATCH_SIZE = 8
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/02_sentiment"

def prepare_data():
    """TODO: 학습2000/검증500/평가500 데이터셋과 tokenizer를 반환하세요."""
    raise NotImplementedError('학습2000/검증500/평가500 데이터셋과 tokenizer를 반환하세요.')


def build_model(mode):
    """TODO: 분류 모델과 frozen/full/lora의 학습 파라미터 범위를 설정하세요."""
    raise NotImplementedError('분류 모델과 frozen/full/lora의 학습 파라미터 범위를 설정하세요.')


def train(model, tokenizer, dataset):
    """TODO: Trainer를 연결하고 학습된 모델을 저장하세요."""
    raise NotImplementedError('Trainer를 연결하고 학습된 모델을 저장하세요.')


def evaluate(model, tokenizer, dataset):
    """TODO: 정확도와 Macro-F1을 반환하세요."""
    raise NotImplementedError('정확도와 Macro-F1을 반환하세요.')


def check_environment():
    config = DistilBertConfig(vocab_size=64, dim=32, hidden_dim=64, n_layers=1, n_heads=4, num_labels=2)
    model = DistilBertForSequenceClassification(config)
    output = model(input_ids=torch.ones(2, 8, dtype=torch.long), labels=torch.tensor([0, 1]))
    assert output.logits.shape == (2, 2) and torch.isfinite(output.loss)
    assert f1_score([0, 1], [0, 1], average="macro") == 1.0
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    train_data, validation_data, test_data, tokenizer = prepare_data()
    model = build_model(MODE)
    train(model, tokenizer, train_data)
    print(evaluate(model, tokenizer, validation_data))
    # 설정 선택을 마친 뒤 test_data를 최종 평가하세요.


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
