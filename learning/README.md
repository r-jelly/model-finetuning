# 학습 과제 · 현재 시작 전

환경은 루트의 uv .venv 하나를 공유합니다. 아래 폴더가 독립적인 과제입니다.
7번과 8번은 내부에 단계 폴더가 있습니다. 학습 해답은 제공하지 않고 TODO 함수로 남겼습니다.

## 첫 행동 · 10분

루트에서 다음 명령을 실행한 뒤 01_mnist/exercise.py의 prepare_data()를 작성하세요.

```bash
uv run python learning/01_mnist/exercise.py --check
```

## 전체 순서

| 폴더 | 주제 | 모델·데이터 | 주요 비교 |
|---|---|---|---|
| 01_mnist | 처음부터 task 학습 | MLP/CNN · MNIST | MLP vs CNN |
| 02_sentiment | 분류 파인튜닝 | DistilBERT · SST-2 | frozen/full/LoRA |
| 03_summarization | 생성 파인튜닝 | T5-small · XSum | 원본/첫 문장/학습 후 |
| 04_retrieval | 검색 파인튜닝 | MiniLM · MS MARCO 일부 | 원본 vs 대조 학습 |
| 05_question_answering | 정답 구간 추출 | DistilBERT · SQuAD | 자르기 vs 겹치는 구간 |
| 06_continued_pretraining | 추가 원문 학습 | GPT-2 · WikiText-2 | padding vs packing |
| 07_cartpole | 기본 RL | Q-table/MLP · CartPole | random → Q-learning → PPO |
| 08_llm_rl | LLM post-training | Qwen2.5-0.5B-Instruct · 합성 덧셈 | 원본 → SFT → GRPO |

## 한 과제를 끝내는 순서

1. README의 첫 행동을 하고 데이터 10개를 직접 확인합니다.
2. exercise.py에서 TODO 함수를 순서대로 작성하고 작은 배치/짧은 실행을 확인합니다.
3. 기준선을 평가하고 변경 하나를 비교합니다.
4. 검증 데이터로 설정을 선택한 뒤 고정 평가 데이터로 최종 비교합니다.
5. results.md에 결과표·실패 사례·다음 판단을 남깁니다. 개선되지 않아도 완료할 수 있습니다.

## 환경 확인

```bash
uv sync --locked
uv run python scripts/check_environment.py
uv run python scripts/check_learning_templates.py
uv run python -m unittest discover -s tests -v
uv pip check
```

--check는 외부 모델/데이터를 다운로드하지 않고 패키지·작은 모델 forward·설정 API만 확인합니다.
--run은 아직 작성하지 않은 함수에서 NotImplementedError로 멈춥니다.
알고리즘 구현 후 shape·수동 계산·저장/재로딩 self-check를 추가하세요.
전체 데이터셋과 모델 다운로드, 8개 과제의 실제 학습·GPU 메모리 검증은 아직 하지 않았습니다.
RL 지원 코드의 데이터 분할·보상 함수는 tests에서 검증합니다.
데이터/모델은 각 과제를 시작할 때 다운로드하며 Hugging Face 기본 cache를 공유합니다.
결과·체크포인트는 루트 outputs/checkpoints 아래 과제별 폴더에 저장합니다.
report_to='none', push_to_hub=False로 시작하며 외부 로깅·업로드는 사용하지 않습니다.

## 구축 확인 · 2026-10-14

11개 단계의 오프라인 환경/API 확인, 지원 코드 테스트 4개, 패키지 의존성 확인을 통과했습니다.
RTX 4070 Ti의 작은 LoRA 학습 단계와 bitsandbytes NF4 연산도 통과했습니다.
CartPole 랜덤 정책은 validation seed 10개에서 평균 return 23.0을 확인했습니다.
이는 환경 기준선이며 아직 Q-learning/PPO나 실제 모델 학습의 결과는 아닙니다.
