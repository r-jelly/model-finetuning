"""Runnable baseline; no rendering or training is required."""
import argparse
import numpy as np
import gymnasium as gym


def run_episode(env, seed):
    observation, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total = 0.0
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return total


def check_environment():
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=42)
        assert observation.shape == (4,) and env.action_space.n == 2
        observation, reward, terminated, truncated, _ = env.step(0)
        assert np.isfinite(observation).all() and reward == 1.0
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
    finally:
        env.close()
    print("CartPole environment OK")


def main():
    env = gym.make("CartPole-v1")
    try:
        returns = [run_episode(env, seed) for seed in range(1000, 1010)]
    finally:
        env.close()
    print({"validation_returns": returns, "mean": float(np.mean(returns)), "std": float(np.std(returns))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_environment()
    else:
        main()
