# 05 · 문서에서 정답 구간 찾기

모델 `distilbert/distilbert-base-uncased`, 데이터 `rajpurkar/squad`. 첫 세션 예상 4시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/05_question_answering/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. train에서 서로 다른 context 기준으로 학습2,000·검증200·평가200 질문을 나눕니다. 같은 문서의 질문이 분할을 넘지 않게 합니다.
2. fast tokenizer의 offset_mapping과 sequence_ids를 사용해 문자 정답을 context 토큰 위치로 변환합니다.
3. 단순 truncation과 겹치는 구간(doc_stride=64) 방식을 각각 구현합니다. 정답 없는 window는 CLS 위치를 label로 사용합니다.
4. AutoModelForQuestionAnswering을 학습합니다. 학습하지 않은 QA head는 의미 있는 기준선이 아닙니다.
5. start<=end, 최대 정답 길이30, context 범위만 허용하고 질문별 모든 window에서 정답을 복원합니다.

## 평가와 완료 조건

동일 질문에서 normalized Exact Match, token F1, 정답 포함 비율을 비교합니다. context/question 중 정답 위치가 바뀌는 버그를 먼저 확인합니다.

## 실행

```bash
uv run python learning/05_question_answering/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/05_question_answering/`, 체크포인트: `checkpoints/05_question_answering/`, 결과: `outputs/05_question_answering/`.

## 실험 범위

시작 설정: input length=256, batch=8, lr=3e-5, 1 epoch. 서로 다른 전처리 때문에 실제 학습 window 수가 달라지면 기록합니다.

## 공식 자료

[SQuAD](https://huggingface.co/datasets/rajpurkar/squad) · [공식 QA 안내](https://huggingface.co/docs/transformers/tasks/question_answering)
