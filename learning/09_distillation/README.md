# 09 · GSM8K 풀이 증류: hard label → soft label → on-policy

student: Qwen/Qwen2.5-0.5B-Instruct, teacher: Qwen/Qwen2.5-1.5B-Instruct. 두 모델은 tokenizer와 vocab(151,936)이 같아 logits를 그대로 비교할 수 있습니다.
데이터: `openai/gsm8k` main. 공식 train을 seed=42로 섞어 학습1,000·검증200, 공식 test에서 평가500을 고정합니다.
`gsm8k.py`에 데이터 분할과 엄격한 정답 보상이 제공돼 있습니다. 답의 마지막 줄 `#### 정수`만 채점합니다.

## 계보

| 연도 | 자료 | 방식 | 단계 |
|---|---|---|---|
| 2015 | [Hinton et al.](https://arxiv.org/abs/1503.02531) | temperature soft label | 02 |
| 2016 | [Kim & Rush](https://arxiv.org/abs/1606.07947) | teacher 생성 시퀀스를 정답으로 학습 | 01 |
| 2019 | [DistilBERT](https://arxiv.org/abs/1910.01108) | soft label로 사전학습 모델 압축 | 02 |
| 2023 | [MiniLLM](https://arxiv.org/abs/2306.08543) · [GKD](https://arxiv.org/abs/2306.13649) | student 생성 + reverse KL | 03 |
| 2024 | [Gemma 2](https://arxiv.org/abs/2408.00118) | 사전학습 단계의 soft label | 02 |
| 2025 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | reasoning 풀이 hard label로 소형 모델 학습 | 01 |
| 2025 | [Qwen3](https://arxiv.org/abs/2505.09388) · [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) | off-policy 후 on-policy 증류 | 01–03 |

## 순서

1. `01_hard_label/`: teacher와 student의 검증 정답률을 잰 뒤, teacher 풀이 중 정답만 남겨 SFT합니다.
2. `02_soft_label/`: 01과 같은 teacher 풀이에서 정답 토큰 대신 teacher 분포 전체를 KL로 학습합니다.
3. `03_on_policy/`: student가 직접 생성한 풀이에서 teacher 분포와의 reverse KL을 학습합니다.
4. 같은 validation/test 문제와 greedy 생성으로 student/hard/soft/on-policy/teacher를 비교합니다.

08 GRPO는 답변 전체에 0/1 보상 하나를 받습니다. 03 on-policy는 같은 student 생성에서 토큰마다 teacher 신호를 받습니다.
01과 02의 차이는 target(정답 토큰 하나 vs teacher 분포)뿐이고, 02와 03의 차이는 시퀀스를 누가 생성했는지뿐입니다.

## 공통 비교 규칙

세 단계 모두 원본 student에서 시작하고 LoRA r=8, alpha=16, target_modules=q_proj/v_proj, lr=1e-4, effective batch 8, optimizer step 100을 같게 둡니다.
teacher 정답률은 상한 참고값으로 함께 기록합니다. test는 설정 선택에 사용하지 않습니다.
Qwen 공개 수치로 GSM8K는 0.5B-Instruct 약 50%, 1.5B-Instruct 약 73%입니다. 프롬프트와 채점 형식이 달라 이 환경에서는 검증하지 않았습니다.
teacher가 student보다 뚜렷이 높지 않으면 진행하지 말고 TEACHER_ID를 Qwen/Qwen2.5-3B-Instruct로 바꿉니다. tokenizer와 vocab은 같습니다.

## 시작 설정과 완료 조건

hard label 첫 세션 약 3시간, soft label 약 3시간, on-policy 약 4시간. 실행 시간은 실측하지 않았습니다.
생성: max_new_tokens=256, 학습 시퀀스 max_length=512. BF16 지원 GPU에서는 bf16=True, gradient checkpointing=True, report_to='none', push_to_hub=False.
gradient checkpointing을 켠 채 train 모드로 생성하면 출력이 깨집니다. 생성 전에 model.eval()을 호출하세요.
문제 8개·optimizer step 2의 작은 실행부터 시작하고, 시간과 GPU 최대 메모리를 확인한 뒤 위 예산으로 늘립니다.
정답률·평균 출력 길이·형식 실패(`####` 없음) 비율·시간·최대 GPU 메모리를 기록합니다.
soft label과 reverse KL loss는 구현 후 `GKDTrainer.generalized_jsd_loss`와 같은 텐서에서 값을 비교하는 self-check를 `check_environment()`에 추가합니다.

## 실행

```bash
uv run python learning/09_distillation/01_hard_label/exercise.py --check
uv run python learning/09_distillation/01_hard_label/exercise.py --run
uv run python learning/09_distillation/02_soft_label/exercise.py --run
uv run python learning/09_distillation/03_on_policy/exercise.py --run
```

## 공식 자료

- [student](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [teacher](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [SFT](https://huggingface.co/docs/trl/sft_trainer) · [GKD](https://huggingface.co/docs/trl/gkd_trainer)
