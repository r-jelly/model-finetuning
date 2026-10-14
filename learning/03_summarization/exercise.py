"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5Config, T5ForConditionalGeneration
from rouge_score import rouge_scorer

SEED = 42
MODEL_ID = "google-t5/t5-small"
DATASET_ID = "EdinburghNLP/xsum"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
MAX_STEPS = 100
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/03_summarization"

def prepare_data():
    """TODO: 고정된 학습/검증/평가 데이터셋과 tokenizer를 반환하세요."""
    raise NotImplementedError('고정된 학습/검증/평가 데이터셋과 tokenizer를 반환하세요.')


def evaluate(model, tokenizer, dataset):
    """TODO: greedy 생성 후 ROUGE-L과 원문/정답/예측을 반환하세요."""
    raise NotImplementedError('greedy 생성 후 ROUGE-L과 원문/정답/예측을 반환하세요.')


def train(model, tokenizer, dataset):
    """TODO: target padding을 loss에서 제외하고 요약 학습을 작성하세요."""
    raise NotImplementedError('target padding을 loss에서 제외하고 요약 학습을 작성하세요.')


def check_environment():
    config = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4, decoder_start_token_id=0, pad_token_id=0)
    model = T5ForConditionalGeneration(config)
    loss = model(input_ids=torch.ones(1, 8, dtype=torch.long), labels=torch.tensor([[2, 3, 1, -100]])).loss
    assert torch.isfinite(loss)
    score = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score("a cat runs", "a cat runs")
    assert score["rougeL"].fmeasure == 1.0
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    train_data, validation_data, test_data, tokenizer = prepare_data()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    print("before", evaluate(model, tokenizer, validation_data))
    train(model, tokenizer, train_data)
    print("after", evaluate(model, tokenizer, validation_data))
    # 설정 선택 후 같은 test_data로 기준선과 학습 모델을 최종 평가하세요.


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
