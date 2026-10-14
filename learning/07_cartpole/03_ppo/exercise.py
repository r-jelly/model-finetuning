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
ROLLOUT_STEPS = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/07_cartpole/ppo"

def build_actor_critic():
    """TODO: 관측 [B,4]에서 logits [B,2]와 value [B]를 반환하는 작은 MLP를 작성하세요."""
    raise NotImplementedError('관측 [B,4]에서 logits [B,2]와 value [B]를 반환하는 작은 MLP를 작성하세요.')


def collect_rollout(env, model, steps):
    """TODO: 관측/action/reward/old logprob/value/최종관측 next_value/종료 flags를 모으세요."""
    raise NotImplementedError('관측/action/reward/old logprob/value/최종관측 next_value/종료 flags를 모으세요.')


def compute_gae(rollout):
    """TODO: delta의 bootstrap은 terminated만 끄고 GAE 재귀는 terminated 또는 truncated에서 끊으세요."""
    raise NotImplementedError('delta의 bootstrap은 terminated만 끄고 GAE 재귀는 terminated 또는 truncated에서 끊으세요.')


def update_policy(model, optimizer, rollout, advantages, returns):
    """TODO: old logprob를 detach하고 clipped policy loss, value loss, entropy를 연결하세요."""
    raise NotImplementedError('old logprob를 detach하고 clipped policy loss, value loss, entropy를 연결하세요.')


def train_and_evaluate():
    """TODO: 남은 step 예산을 지키며 rollout/update를 반복하고 greedy 평가를 수행하세요."""
    raise NotImplementedError('남은 step 예산을 지키며 rollout/update를 반복하고 greedy 평가를 수행하세요.')


def check_environment():
    env = gym.make(ENV_ID)
    try:
        observation, _ = env.reset(seed=SEED)
        assert observation.shape == (4,)
    finally:
        env.close()
    policy = Categorical(logits=torch.zeros(2, 2))
    actions = policy.sample()
    assert actions.shape == (2,) and policy.log_prob(actions).shape == (2,)
    assert ROLLOUT_STEPS % MINIBATCH_SIZE == 0
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
