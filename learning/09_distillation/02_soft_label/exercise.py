"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import sys
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm8k import make_splits, exact_reward
from trl.experimental.gkd import GKDTrainer

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[3]
TRACES_PATH = ROOT / "data/09_distillation/teacher_traces.jsonl"
OUTPUT_DIR = ROOT / "outputs/09_distillation/soft_label"
SEED = 42
MAX_NEW_TOKENS = 256
MAX_LENGTH = 512
MAX_STEPS = 100
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
TEMPERATURE = 1.0

def load_teacher():
    """TODO: teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요."""
    raise NotImplementedError('teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요.')


def load_student():
    """TODO: student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요."""
    raise NotImplementedError('student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요.')


def evaluate(model, tokenizer, rows):
    """TODO: model.eval() 후 01과 같은 greedy 생성으로 정답률을 반환하세요."""
    raise NotImplementedError('model.eval() 후 01과 같은 greedy 생성으로 정답률을 반환하세요.')


def distill_loss(student_logits, teacher_logits, labels, temperature):
    """TODO: temperature로 나눈 분포의 KL(teacher‖student)을 labels != -100 위치에서 토큰 평균하세요."""
    raise NotImplementedError('temperature로 나눈 분포의 KL(teacher‖student)을 labels != -100 위치에서 토큰 평균하세요.')


def train(student, teacher, tokenizer, traces):
    """TODO: 01과 같은 teacher 풀이에서 teacher logits를 no_grad로 구해 distill_loss로 MAX_STEPS 학습하고 adapter를 저장하세요."""
    raise NotImplementedError('01과 같은 teacher 풀이에서 teacher logits를 no_grad로 구해 distill_loss로 MAX_STEPS 학습하고 adapter를 저장하세요.')


def check_environment():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    logits = model(input_ids=torch.ones(1, 8, dtype=torch.long)).logits
    labels = torch.tensor([[-100] * 4 + [1] * 4])
    reference = GKDTrainer.generalized_jsd_loss(logits, torch.randn_like(logits), labels, beta=0.0, temperature=TEMPERATURE)
    assert torch.isfinite(reference) and reference > 0
    assert exact_reward(["#### -5", "#### 5\nmore"], [-5, 5]) == [1.0, 0.0]
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    if not TRACES_PATH.is_file():
        raise FileNotFoundError("01_hard_label에서 teacher 풀이를 data/09_distillation/teacher_traces.jsonl에 먼저 저장하세요.")
    torch.manual_seed(SEED)
    splits = make_splits()
    teacher, _ = load_teacher()
    student, tokenizer = load_student()
    train(student, teacher, tokenizer, Dataset.from_json(str(TRACES_PATH)))
    print("soft label", evaluate(student, tokenizer, splits["validation"]))
    # adapter를 재로딩해 같은 greedy 결과를 확인하세요. test는 세 단계 설정을 모두 고른 뒤에만 평가합니다.


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
