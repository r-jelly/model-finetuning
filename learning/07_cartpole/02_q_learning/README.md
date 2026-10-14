# Q-learning · 첫 세션 약 3시간

## 직접 작성할 순서

1. `discretize()`를 작성하고 하한/상한/범위 밖/경계의 출력이 항상 유효 index인지 assert합니다.
2. Q-table과 epsilon-greedy를 작성합니다. 동점 argmax 편향을 관찰하세요.
3. Bellman update를 작성하고 손으로 계산한 한 transition과 비교합니다.
4. train/evaluate를 연결합니다. epsilon과 환경 step 예산을 기록합니다.
5. N_BINS=6과 10만 비교하고 경계로 잘린 관측 비율·평균 return을 기록합니다.

run: `uv run python learning/07_cartpole/02_q_learning/exercise.py --run`.
terminated와 truncated를 하나로 합쳐 bootstrap을 끄지 마세요.
최종 평가 규칙과 seed는 상위 README를 따릅니다.
