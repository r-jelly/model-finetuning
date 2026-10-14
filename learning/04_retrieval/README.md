# 04 · 질문에 맞는 문서 검색

모델 `sentence-transformers/all-MiniLM-L6-v2`, 데이터 `mteb/scifact`. 첫 세션 예상 5시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/04_retrieval/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. `corpus`(문서 5,183개)·`queries`·`default`(train/test 관련 쌍) config를 불러 query 10개와 정답 문서를 직접 읽습니다.
2. train query 809개를 seed=42로 학습 709·검증 100으로 나누고 ID를 파일로 저장합니다. test query 300개는 최종 평가로 고정합니다. 문서는 title과 text를 이어 붙입니다.
3. 원본 모델의 cosine 검색을 전체 corpus에서 평가합니다. 정답 문서가 여러 개면 모두 relevant로 표시합니다.
4. 학습 query/positive 쌍에 MultipleNegativesRankingLoss를 적용합니다. 동일 positive가 한 batch에 반복되지 않게 `BatchSamplers.NO_DUPLICATES`를 씁니다.
5. 같은 corpus에서 재평가하고 엉뚱한 상위 검색 결과 10개를 분석합니다.

## 평가와 완료 조건

Recall@5 (정답 문서 회수 비율)와 MRR@10 (첫 정답 순위의 역수 평균)을 계산합니다. test query는 학습에 넣지 않습니다.
이 모델은 MS MARCO로 이미 학습됐기 때문에 처음 보는 과학 주장 검색 데이터를 씁니다.

## 실행

```bash
uv run python learning/04_retrieval/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/04_retrieval/`, 체크포인트: `checkpoints/04_retrieval/`, 결과: `outputs/04_retrieval/`.

## 실험 범위

시작 설정: batch=32, max_seq_length=256(모델 기본값), 3 epochs. MNRL은 batch 안의 다른 문서를 negative로 쓰므로 batch가 클수록 negative가 많습니다.

## 공식 자료

[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [SciFact](https://huggingface.co/datasets/mteb/scifact)
