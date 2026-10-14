# 03 · 뉴스 한 문장 요약

모델 `google-t5/t5-small`, 데이터 `EdinburghNLP/xsum`. 첫 세션 예상 4시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/03_summarization/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. train 1,000개·validation 100개·test 100개를 seed=42로 고정합니다. document와 summary를 직접 10개 읽습니다.
2. 입력에 summarize: 접두사를 붙입니다. 입력 최대 512토큰, target 최대 64토큰. target padding의 loss는 -100으로 제외합니다.
3. 학습 전 모델과 첫 문장 반환 기준선을 평가합니다.
4. Seq2SeqTrainer 또는 직접 학습 루프를 작성하고 greedy generate로 동일 평가합니다.
5. ROUGE-L과 출력 20개의 사실 오류·누락을 함께 기록합니다.

## 평가와 완료 조건

ROUGE-L, 사실 오류·누락 사례. 모델 저장·재로딩 후 같은 입력의 생성 결과를 확인합니다.

## 실행

```bash
uv run python learning/03_summarization/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/03_summarization/`, 체크포인트: `checkpoints/03_summarization/`, 결과: `outputs/03_summarization/`.

## 실험 범위

시작 설정: batch=2, gradient accumulation=4, lr=5e-5, max_steps=100, max_new_tokens=64. 전체 XSum 벤치마크 점수로 해석하지 않습니다.
T5는 fp16에서 overflow로 loss가 NaN이 될 수 있어 bf16 또는 fp32로 학습합니다. ROUGE는 use_stemmer=True로 계산합니다.

## 공식 자료

[T5-small](https://huggingface.co/google-t5/t5-small) · [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)
