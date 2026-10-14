"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical
import gymnasium as gym

SEED = 42
ENV_ID = "CartPole-v1"
TOTAL_STEPS = 50_000
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPISODES_PER_UPDATE = 4
USE_BASELINE = False  # False 실행 후 True로 바꿔 비교합니다.
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/07_cartpole/reinforce"

def build_policy():
    """TODO: 관측 [B,4]에서 logits [B,2]를 반환하는 작은 MLP를 작성하세요."""
    raise NotImplementedError('관측 [B,4]에서 logits [B,2]를 반환하는 작은 MLP를 작성하세요.')


def run_episode(env, policy, seed):
    """TODO: 정책에서 action을 sample해 episode를 끝까지 실행하고 logprob 목록과 reward 목록을 반환하세요."""
    raise NotImplementedError('정책에서 action을 sample해 episode를 끝까지 실행하고 logprob 목록과 reward 목록을 반환하세요.')


def compute_returns(rewards):
    """TODO: 각 시점의 할인 return G_t = r_t + GAMMA * G_{t+1}을 뒤에서부터 계산하세요."""
    raise NotImplementedError('각 시점의 할인 return G_t = r_t + GAMMA * G_{t+1}을 뒤에서부터 계산하세요.')


def update_policy(optimizer, log_probs, returns):
    """TODO: USE_BASELINE이면 returns에서 평균을 빼고, -(logprob * return)의 평균으로 한 번 업데이트하세요."""
    raise NotImplementedError('USE_BASELINE이면 returns에서 평균을 빼고, -(logprob * return)의 평균으로 한 번 업데이트하세요.')


def train_and_evaluate():
    """TODO: episode를 EPISODES_PER_UPDATE개씩 모아 업데이트하며 TOTAL_STEPS 예산을 지키고 greedy 평가를 수행하세요."""
    raise NotImplementedError('episode를 EPISODES_PER_UPDATE개씩 모아 업데이트하며 TOTAL_STEPS 예산을 지키고 greedy 평가를 수행하세요.')


def check_environment():
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(seed=SEED)
        assert observation.shape == (4,) and env.action_space.n == 2
    finally:
        env.close()
    policy = Categorical(logits=torch.zeros(3, 2))
    actions = policy.sample()
    assert actions.shape == (3,) and policy.log_prob(actions).shape == (3,)
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    train_and_evaluate()


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
