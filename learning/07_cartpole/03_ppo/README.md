# PPO · 2회 × 약 3시간

## 직접 작성할 순서

1. Actor–Critic을 작성하고 logits/value의 shape, sample action/logprob를 확인합니다.
2. rollout을 수집합니다. old logprob/value는 계산 그래프 없이 저장합니다.
3. GAE를 작성하고 짧은 수동 계산 예제와 비교합니다. advantages/returns는 detach합니다.
4. PPO loss를 작성합니다. old=new이면 ratio=1인지 확인하고 policy/value/entropy를 따로 기록합니다.
5. 총 환경 step 50,000회로 학습한 뒤 Q-learning과 같은 seed·평가 방식으로 비교합니다.

terminated: bootstrap=0, GAE 재귀=0.
truncated: 마지막 실제 관측으로 bootstrap, GAE 재귀=0.
rollout 예산 때문에 자른 지점: 실제 next_value로 bootstrap, 현재 buffer 밖으로 재귀하지 않습니다.
reset 뒤 observation으로 이전 episode의 next_value를 계산하지 않습니다.

시작 설정: 1개 환경, CPU, hidden size=64, clip=0.2, value coefficient=0.5, entropy coefficient=0.01.
run: `uv run python learning/07_cartpole/03_ppo/exercise.py --run`.
[공식 알고리즘 설명](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
