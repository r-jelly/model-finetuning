# 02 · 감정 분류와 LoRA

모델 `distilbert/distilbert-base-uncased`, 데이터 `stanfordnlp/sst2`. 첫 세션 예상 3시간, LoRA 비교는 별도 세션.

## 첫 행동 · 10분

루트에서 `uv run python learning/02_sentiment/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. train을 섞어 3,000개를 고정하고 학습 2,000개·검증 500개·최종 평가 500개로 나눕니다. 공식 test의 label=-1은 사용하지 않습니다.
2. `prepare_data()`에서 AutoTokenizer, 최대 길이 128, DataCollatorWithPadding을 연결합니다.
3. `build_model()`에서 frozen 모드를 먼저 작성합니다. 인코더는 동결하고 pre_classifier/classifier는 학습합니다.
4. `train()`과 `evaluate()`를 연결하고 full 모드를 같은 조건에서 비교합니다.
5. lora 모드에서 target_modules=[q_lin,v_lin], r=8, alpha=16, modules_to_save=[pre_classifier,classifier]를 사용해 비용도 비교합니다.

## 평가와 완료 조건

정확도·Macro-F1, 학습 파라미터 수·최대 GPU 메모리·시간을 비교합니다. 새로 초기화된 분류층은 무학습 기준선이 아니라 학습 대상입니다.

## 실행

```bash
uv run python learning/02_sentiment/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/02_sentiment/`, 체크포인트: `checkpoints/02_sentiment/`, 결과: `outputs/02_sentiment/`.

## 실험 범위

시작 설정: batch=8, 2 epochs. full/lora lr=2e-5, frozen head lr=1e-3. 모드별 학습률 차이를 기록하세요. report_to="none", push_to_hub=False.

## 공식 자료

[모델](https://huggingface.co/distilbert/distilbert-base-uncased) · [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) · [LoRA](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
