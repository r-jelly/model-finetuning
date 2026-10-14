# REINFORCE · 첫 세션 약 3시간

## 직접 작성할 순서

1. `build_policy()`를 작성하고 logits shape, sample action/logprob를 확인합니다.
2. `run_episode()`로 episode 하나의 logprob와 reward를 모읍니다.
3. `compute_returns()`를 작성하고 reward [1,1,1], gamma=0.5의 return [1.75, 1.5, 1]과 비교합니다.
4. `update_policy()`에서 -(logprob × return)의 평균을 loss로 한 번 업데이트합니다.
5. USE_BASELINE=False/True를 학습 seed 42/43/44에서 비교하고 평균 return의 평균·표준편차를 기록합니다.

baseline은 한 업데이트에 모은 episode들의 return 평균입니다. GRPO가 같은 요청에 대한 답변 그룹의 평균을 빼는 것과 같은 역할입니다.
시작 설정: hidden size=64, lr=1e-3, episode 4개마다 업데이트.
run: `uv run python learning/07_cartpole/02_reinforce/exercise.py --run`.
최종 평가 규칙과 seed는 상위 README를 따릅니다.
