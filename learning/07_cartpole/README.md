# 07 · CartPole: 랜덤 정책 → Q-learning → PPO

환경은 CartPole-v1, 관측은 연속값 4개, 행동은 좌/우 2개입니다. CPU로 충분합니다.
첫 세션 예상: 환경 관찰 30분, Q-learning 3시간, PPO는 2회 × 3시간.

## 시작 · 5분

```bash
uv run python learning/07_cartpole/01_random/exercise.py --run
```

## 단계

1. `01_random/`: 제공된 랜덤 정책을 실행하고 observation/action/reward/terminated/truncated를 확인합니다.
2. `02_q_learning/`: 관측을 구간으로 나누고 Q-table, epsilon-greedy, Bellman update를 직접 작성합니다.
3. `03_ppo/`: Actor–Critic, rollout, GAE, clipped policy loss를 직접 작성합니다.

## 공통 비교 규칙

학습 예산은 각각 환경 step 50,000회로 시작합니다. 성능 개선을 보장하는 예산이 아닙니다.
검증 seed 1000~1009로 설정을 선택합니다. 최종 seed 2000~2019는 마지막 비교에만 사용합니다.
학습 seed 42로 작은 실험을 끝낸 뒤, 차이가 보이면 43/44로 반복합니다.
평가 때는 탐험 없이 같은 방식(각 정책의 greedy action)으로 평균 episode return과 표준편차를 계산합니다.
랜덤 정책은 평가 기준선이며 greedy action 비교 대상이 아닙니다.

실제 실패 terminated에서는 bootstrap을 끄지만, 시간 제한 truncated에서는 마지막 관측으로 bootstrap합니다.
두 경우 모두 환경은 reset합니다. 다음 episode의 관측을 마지막 관측 대신 쓰지 않습니다.
Q-learning/PPO 비교는 알고리즘과 상태 표현이 함께 달라지는 비교입니다.

## 출력

checkpoint: `checkpoints/07_cartpole/`, 결과·학습 곡선: `outputs/07_cartpole/`.
baseline/q_learning/ppo 이름을 구분해 저장하고 `results.md`에 기록하세요.

## 공식 자료

- [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [시간 제한 처리](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
