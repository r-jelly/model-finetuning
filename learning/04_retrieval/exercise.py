"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.base.sampler import BatchSamplers

SEED = 42
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_ID = "mteb/scifact"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/04_retrieval"

def prepare_data():
    """TODO: 학습 pairs, 검증 queries, 평가 queries, 고정 corpus, 관련 문서 ID를 반환하세요."""
    raise NotImplementedError('학습 pairs, 검증 queries, 평가 queries, 고정 corpus, 관련 문서 ID를 반환하세요.')


def evaluate(model, queries, corpus, relevant_ids):
    """TODO: cosine 순위를 계산하고 Recall@5/MRR@10을 반환하세요."""
    raise NotImplementedError('cosine 순위를 계산하고 Recall@5/MRR@10을 반환하세요.')


def train(model, pairs):
    """TODO: MultipleNegativesRankingLoss로 학습 query/positive 쌍을 학습하세요."""
    raise NotImplementedError('MultipleNegativesRankingLoss로 학습 query/positive 쌍을 학습하세요.')


def check_environment():
    assert losses.MultipleNegativesRankingLoss is not None
    assert SentenceTransformerTrainer is not None and BatchSamplers.NO_DUPLICATES
    vectors = torch.nn.functional.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=1)
    assert torch.allclose(vectors @ vectors.T, torch.eye(2))
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    pairs, validation_queries, test_queries, corpus, relevant_ids = prepare_data()
    model = SentenceTransformer(MODEL_ID)
    model.max_seq_length = MAX_SEQ_LENGTH
    print("before", evaluate(model, validation_queries, corpus, relevant_ids))
    train(model, pairs)
    print("after", evaluate(model, validation_queries, corpus, relevant_ids))
    # test_queries는 설정 선택이 끝난 뒤 최종 비교에만 사용하세요.


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
