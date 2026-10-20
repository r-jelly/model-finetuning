"""Tests for provided support code, not the learner's TODO algorithms."""
import importlib.util
from pathlib import Path
import unittest
import runpy
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("arithmetic", ROOT / "learning/08_llm_rl/arithmetic.py")
arithmetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arithmetic)
spec = importlib.util.spec_from_file_location("gsm8k", ROOT / "learning/09_distillation/gsm8k.py")
gsm8k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gsm8k)


class ArithmeticTests(unittest.TestCase):
    def test_splits_are_repeatable_disjoint_and_commutative_unique(self):
        first = arithmetic.make_splits()
        self.assertEqual(first, arithmetic.make_splits())
        self.assertEqual([len(first[k]) for k in ("train", "validation", "test")], [2000, 500, 500])
        ids = [row["id"] for rows in first.values() for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for rows in first.values():
            for row in rows:
                a, b = map(int, row["id"].split("*"))
                self.assertLessEqual(a, b)
                self.assertEqual(row["answer"], a * b)
                self.assertEqual(row["completion"][0]["content"], str(a * b))

    def test_reward_accepts_only_exact_integer_answers(self):
        texts = ["3", " 3\n", "+3", "03", "4", "3 and 4", "3.0", "The answer is 3", "", "9999999999999999999", "3; import os", "٣"]
        self.assertEqual(arithmetic.exact_reward(texts, [3] * len(texts)), [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(arithmetic.exact_reward([[{"role": "assistant", "content": "3"}], [], [{"content": "3"}, {"content": "4"}], None], [3] * 4), [1., 0., 0., 0.])

    def test_mismatched_reward_batch_fails(self):
        with self.assertRaises(ValueError):
            arithmetic.exact_reward(["3"], [])


class Gsm8kRewardTests(unittest.TestCase):
    def test_reward_reads_only_final_answer_line(self):
        texts = ["2 + 3 = 5\n#### 72", "#### 72.", "####72\n", "#### 7,2", "#### 73", "The answer is 72", "#### 72\nDone.", "#### 72 apples", "#### 72.5", "", None]
        self.assertEqual(gsm8k.exact_reward(texts, [72] * len(texts)), [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(gsm8k.exact_reward(["#### 1,000", "#### -5", [{"role": "assistant", "content": "#### 3"}], []], [1000, -5, 3, 3]), [1., 1., 1., 0.])

    def test_gold_answer_format_parses(self):
        self.assertEqual(gsm8k.extract_answer("Natalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"), 72)

    def test_mismatched_reward_batch_fails(self):
        with self.assertRaises(ValueError):
            gsm8k.exact_reward(["#### 3"], [])


class RandomPolicyTests(unittest.TestCase):
    def test_time_limit_ends_episode(self):
        module = runpy.run_path(str(ROOT / "learning/07_cartpole/01_random/exercise.py"))
        env = gym.make("CartPole-v1", max_episode_steps=1)
        try:
            self.assertEqual(module["run_episode"](env, 42), 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
