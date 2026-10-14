# 04 · 질문에 맞는 문서 검색

모델 `sentence-transformers/all-MiniLM-L6-v2`, 데이터 `microsoft/ms_marco`, config `v1.1`. 첫 세션 예상 5시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/04_retrieval/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. train에서 is_selected=1인 문서가 있는 서로 다른 query 1,000개를 고정합니다. query_id 기준 학습800/검증100/평가100으로 나누고 passage_text/is_selected를 확인합니다.
2. 정답 passage와 방해 passage로 평가 후보 집합을 최대 1,000개 고정합니다. 문서 텍스트를 중복 제거하고 여러 정답을 모두 relevant로 표시합니다.
3. 원본 모델의 cosine 검색을 평가합니다. 평가 후보 ID와 query ID를 파일로 저장합니다.
4. 학습 query/positive 쌍에 MultipleNegativesRankingLoss를 적용합니다. 동일 positive가 한 batch에 반복되지 않게 합니다.
5. 같은 후보 집합에서 재평가하고 엉뚱한 상위 검색 결과 10개를 분석합니다.

## 평가와 완료 조건

Recall@5 (정답 문서 회수 비율)와 MRR@10 (첫 정답 순위의 역수 평균)을 계산합니다. test query는 학습에 넣지 않습니다.

## 실행

```bash
uv run python learning/04_retrieval/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/04_retrieval/`, 체크포인트: `checkpoints/04_retrieval/`, 결과: `outputs/04_retrieval/`.

## 실험 범위

시작 설정: batch=8, max_seq_length=128, 1 epoch. 큰 corpus는 streaming 또는 필요한 shard만 사용하세요. 작은 후보 집합의 결과는 전체 MS MARCO 성능이 아닙니다.

## 공식 자료

[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco)
