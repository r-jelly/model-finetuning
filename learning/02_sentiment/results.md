# 실험 기록

상태: 완료

## 질문과 가설

- 해결할 문제: 같은 모델을 같은 데이터셋에 대해 fine-tuning 할 때, Full fine-tuning, 레이어 Freezing, LoRA의 성능 차이는 어떻게 되는가?
- 예상하는 변화와 이유: Full > LoRA > Freeze 순으로 성능 향상이 있을 것. 가장 리소스는 많이 필요하지만 모든 weight을 변동시키는 Full이 제일 성능이 높을 것이고, classification head만을 학습하는 Freeze가 리소스는 제일 적게 필요하나 성능은 제일 낮을 것. LoRA는 그 중간일 것이다.
- 변경하는 변수 하나: 학습 방법론

## 통제 조건

- 데이터 분할·샘플 수·seed: SST-2 데이터셋의 `train`에서 학습 2,000 / 검증 500개, `validation`에서 평가 872개의 데이터 사용
- 모델·학습률·배치·입력 길이: DistillBERT, Trainer Default (AdamW), 각 학습 방법별 다른 lr, batch=8, max_length=128
- 학습 예산 (optimizer step / 환경 step): 500 step (2 epoch)
- 평가 데이터·평가 방식: 고정 test 872개에 대한 정확도, Macro F1 점수.

## 결과

| 실행 | 변경 | 주요 지표 | 평가 n | 95% CI | 시간 (sec) | 최대 GPU 메모리 (MB) |
|---|---|---|---|---|---|---|
| Frozen | lr=0.003 | test acc=0.83, f1=0.83 | 872 | [0.816, 0.854] | 4.32  | 303 |
| LoRA | lr=0.001 | val acc=0.863, f1=0.863  | 872 | [0.839, 0.885] | 9.52 | 386 |
| Full | lr=0.00005 | val acc=0.859, f1=0.859 | 872 | [0.834, 0.880] | 12.0 | 1052 |

정답률의 95% 신뢰구간은 Wilson 구간으로, 다른 지표는 bootstrap으로 구합니다.

두 모델을 비교할 때는 같은 평가 사례로 paired bootstrap을 돌려 점수 차이(비교 − 기준선)의 95% 신뢰구간을 구합니다. 이 구간 전체가 비교 모델이 더 좋은 쪽에 있을 때만 개선이라고 씁니다.

- Frozen vs LoRA: acc +0.033, 95% CI [+0.011, +0.056], F1 +0.033, 95% CI [+0.011, +0.056]
- LoRA vs Full: acc -0.004, 95% CI [-0.023, 0.013], F1 -0.004, 95% CI [-0.023, 0.013]

Frozen보다 LoRA에서는 확실한 개선, LoRA와 Full은 판단 불가

## 실패 사례

입력 / 정답 / 예측 / 원인 후보를 10개 확인하세요. RL은 실패한 episode를 확인하세요.
원인 후보는 관찰과 구분하고, 검증 전에는 결론으로 쓰지 마세요.

- often heartbreaking (정답 0, 예측 1), seamless (정답 1, 예측 0): 짧은 단어로만 이루어져 있는 문장들이 많아, 분류에 어려움을 겪었을 가능성

## 판단

- 가설을 지지하는 관찰: LoRA가 Frozen fine-tuning에 비해 3.3%p의 성능 향상을 보였다. LoRA의 GPU 메모리 사용량이 거의 Frozen과 유사하였다.
- 가설과 다른 관찰: Test 데이터에 대해서 LoRA와 Full의 성능 차이가 거의 없었다. Validation에서는 유의미한 성능 향상이 있었으므로 over-fitting이 발생했다고 볼 수도 있었을 것이다.
- 다음 행동 하나: Full fine-tuning에 regularization을 추가해 over-fitting을 방지하는 전략을 쓸 수 있다.

완료: 기준선 + 변경 하나 + 동일 평가 + 결과 해석. 성능 개선은 필수 조건이 아닙니다.
