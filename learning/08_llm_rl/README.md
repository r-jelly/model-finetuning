# 08 · 작은 LLM의 산술 답변: SFT → GRPO

모델: Qwen/Qwen2.5-0.5B-Instruct. LoRA r=8, alpha=16, target_modules=q_proj/v_proj로 시작합니다.
데이터: 10~99의 두 수 곱셈 4,095개. 교환법칙 중복(a,b)/(b,a)을 제거하고 seed=42로 train2000/validation500/test500을 나눕니다.
`arithmetic.py`에 데이터 분할과 엄격한 정답 보상이 제공돼 있습니다. 외부 데이터나 reward model은 필요 없습니다.

## 순서

1. `01_sft/`: 원본 모델의 greedy 정답률을 측정하고, train의 요청+모범 답변으로 LoRA SFT를 작성합니다.
2. SFT 모델과 adapter를 저장하고 새로 로드한 모델의 예측이 같은지 확인합니다.
3. `02_grpo/`: 저장한 SFT checkpoint에서 시작해 요청당 여러 답변을 생성하고 정답 보상으로 GRPO를 작성합니다.
4. 같은 validation/test 문제와 greedy 생성으로 원본/SFT/GRPO를 비교합니다.

원본 Instruct 모델도 이미 post-training된 모델입니다. 이 과제는 새로운 모델의 전체 사전학습을 재현하지 않습니다.
두 단계에 같은 chat template과 '정수 하나만 답하세요' 형식을 사용합니다.
test는 설정 선택에 사용하지 않습니다. outputs 또는 checkpoint에 Hugging Face 토큰을 저장하지 않습니다.

## 시작 설정과 완료 조건

SFT 첫 세션 약 4시간, GRPO 연결·분석은 별도 약 4시간. 실행 시간 실측: SFT 50 step 약 30초, GRPO 1 step 약 0.4초.
SFT: batch=1, accumulation=4, max_length=128, max_steps=50, lr=1e-4, completion_only_loss=True.
GRPO: batch=8, num_generations=8, accumulation=2, max_completion_length=8, max_steps=200, beta=0, use_vllm=False. lr은 5e-6과 5e-5를 validation으로 비교해 고릅니다.
BF16 지원 GPU에서는 bf16=True, gradient checkpointing=True, report_to='none', push_to_hub=False.
gradient checkpointing을 켠 채 train 모드로 생성하면 출력이 깨집니다. 평가 전에 model.eval()을 호출하세요.
optimizer step=2의 작은 실행부터 시작하고, GPU 최대 메모리를 확인한 뒤 위 예산으로 늘립니다.
같은 요청의 생성 8개가 모두 같은 reward면 상대적 보상 신호가 없습니다. 학습 전에 reward 분포와 그룹 내 차이를 확인하세요.
문제가 너무 쉽다면 피연산자 범위를 늘리고 분할을 처음부터 다시 고정하세요. test 결과를 보고 난 뒤 조정했다면 새 test가 필요합니다.
정답률·평균 reward·그룹 내 reward 차이가 있는 비율·출력 길이·최대 GPU 메모리를 기록합니다.
GRPO 전후에는 pass@8(샘플 8개 중 하나라도 정답인 비율)도 비교합니다.

## 실행

```bash
uv run python learning/08_llm_rl/01_sft/exercise.py --check
uv run python learning/08_llm_rl/01_sft/exercise.py --run
uv run python learning/08_llm_rl/02_grpo/exercise.py --check
uv run python learning/08_llm_rl/02_grpo/exercise.py --run
```

## 공식 자료

- [모델](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [SFT](https://huggingface.co/docs/trl/sft_trainer)
- [GRPO](https://huggingface.co/docs/trl/grpo_trainer)
