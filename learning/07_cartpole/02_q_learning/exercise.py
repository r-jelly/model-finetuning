"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym

SEED = 42
ENV_ID = "CartPole-v1"
TOTAL_STEPS = 50_000
GAMMA = 0.99
ALPHA = 0.1
N_BINS = 6
# shortcut: 무한 속도 범위를 유한 구간으로 자릅니다. 경계 밖 관측 비율이 높으면 범위를 조정하세요.
BOUNDS = np.array([[-2.4, 2.4], [-3., 3.], [-0.2095, 0.2095], [-3., 3.]])
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/07_cartpole/q_learning"

def discretize(observation):
    """TODO: 관측 4개를 clip하고 각각 0~N_BINS-1 정수 tuple로 바꾸세요."""
    raise NotImplementedError('관측 4개를 clip하고 각각 0~N_BINS-1 정수 tuple로 바꾸세요.')


def choose_action(q_table, state, epsilon, rng):
    """TODO: epsilon 확률의 탐험, 아니면 argmax action을 반환하세요."""
    raise NotImplementedError('epsilon 확률의 탐험, 아니면 argmax action을 반환하세요.')


def update_q(q_table, state, action, reward, next_state, terminated):
    """TODO: reward + gamma*maxQ에서 terminated일 때만 bootstrap을 제거하세요."""
    raise NotImplementedError('reward + gamma*maxQ에서 terminated일 때만 bootstrap을 제거하세요.')


def train(env):
    """TODO: seed, epsilon 감소, 종료/reset을 처리하며 정확히 TOTAL_STEPS 상호작용하세요."""
    raise NotImplementedError('seed, epsilon 감소, 종료/reset을 처리하며 정확히 TOTAL_STEPS 상호작용하세요.')


def evaluate(q_table, seeds):
    """TODO: 탐험 없는 정책의 seed별 episode return을 반환하세요."""
    raise NotImplementedError('탐험 없는 정책의 seed별 episode return을 반환하세요.')


def check_environment():
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(seed=SEED)
        assert observation.shape == (4,) and env.action_space.n == 2
        assert BOUNDS.shape == (4, 2) and np.all(BOUNDS[:, 0] < BOUNDS[:, 1])
        assert np.zeros((N_BINS,) * 4 + (2,)).shape == (6, 6, 6, 6, 2)
    finally:
        env.close()
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    env = gym.make(ENV_ID)
    try:
        q_table = train(env)
    finally:
        env.close()
    print(evaluate(q_table, range(1000, 1010)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="다운로드 없는 환경 확인")
    mode.add_argument("--run", action="store_true", help="직접 작성한 학습 코드 실행")
    args = parser.parse_args()
    if args.check:
        check_environment()
    else:
        main()
