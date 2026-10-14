# 01 · MNIST 숫자 분류

랜덤 초기화 MLP와 CNN을 같은 데이터에서 비교합니다. 첫 학습 세션 예상 2시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/01_mnist/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. `prepare_data()`에서 torchvision MNIST와 ToTensor를 사용합니다. 원래 train에서 seed=42로 학습 5,000장·검증 1,000장을 뽑고, 공식 test에서 1,000장을 평가용으로 고정합니다.
2. `build_model()`에서 MLP부터 작성합니다. 입력 [B,1,28,28], 출력 logits [B,10]. CrossEntropyLoss 앞에 softmax를 넣지 않습니다.
3. `train()`을 작성하고 학습 32장만 반복해 암기 가능한지 먼저 확인합니다.
4. `evaluate()`는 eval 모드와 no_grad로 정확도·혼동행렬·틀린 이미지 20장을 반환합니다.
5. CNN으로 모델만 바꿔 같은 데이터·epoch·평가에서 비교하고 results.md를 채웁니다.

## 평가와 완료 조건

정확도·혼동행렬을 비교합니다. 32장 암기 확인, 기준선/비교 모델 저장 후 재로딩 결과 확인까지 완료합니다.

## 실행

```bash
uv run python learning/01_mnist/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/01_mnist/`, 체크포인트: `checkpoints/01_mnist/`, 결과: `outputs/01_mnist/`.

## 실험 범위

시작 설정: batch=64, Adam lr=1e-3, 3 epochs. 검증·평가 이미지에는 학습 증강을 적용하지 않습니다.

## 공식 자료

[MNIST API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
