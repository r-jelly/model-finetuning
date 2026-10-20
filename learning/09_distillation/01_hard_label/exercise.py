"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm8k import make_splits, exact_reward
from trl import SFTConfig, SFTTrainer

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[3]
TRACES_PATH = ROOT / "data/09_distillation/teacher_traces.jsonl"
OUTPUT_DIR = ROOT / "outputs/09_distillation/hard_label"
SEED = 42
MAX_NEW_TOKENS = 256
MAX_LENGTH = 512
MAX_STEPS = 100
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4

def load_teacher():
    """TODO: teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요."""
    raise NotImplementedError('teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요.')


def load_student():
    """TODO: student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요."""
    raise NotImplementedError('student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요.')


def evaluate(model, tokenizer, rows):
    """TODO: model.eval() 후 greedy 생성의 입력 prefix를 제외한 completion을 exact_reward로 평가해 정답률을 반환하세요."""
    raise NotImplementedError('model.eval() 후 greedy 생성의 입력 prefix를 제외한 completion을 exact_reward로 평가해 정답률을 반환하세요.')


def generate_traces(teacher, tokenizer, rows):
    """TODO: teacher greedy 풀이 중 exact_reward가 1인 것만 prompt/completion 형식으로 TRACES_PATH에 저장하세요."""
    raise NotImplementedError('teacher greedy 풀이 중 exact_reward가 1인 것만 prompt/completion 형식으로 TRACES_PATH에 저장하세요.')


def train(model, tokenizer, traces):
    """TODO: teacher 풀이를 SFTConfig와 연결해 completion 토큰만 학습하고 adapter와 tokenizer를 저장하세요."""
    raise NotImplementedError('teacher 풀이를 SFTConfig와 연결해 completion 토큰만 학습하고 adapter와 tokenizer를 저장하세요.')


def check_environment():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    assert model(input_ids=torch.ones(1, 8, dtype=torch.long)).logits.shape == (1, 8, 64)
    config = SFTConfig(output_dir=str(OUTPUT_DIR), max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE, completion_only_loss=True, report_to="none", push_to_hub=False)
    assert config.max_length == MAX_LENGTH and config.completion_only_loss
    assert exact_reward(["3 + 4 = 7\n#### 7", "#### 1,000", "The answer is 7"], [7, 1000, 7]) == [1.0, 1.0, 0.0]
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    splits = make_splits()
    teacher, tokenizer = load_teacher()
    print("teacher", evaluate(teacher, tokenizer, splits["validation"]))
    if not TRACES_PATH.is_file():
        generate_traces(teacher, tokenizer, splits["train"])
    del teacher  # SFT 동안 teacher가 GPU 메모리를 차지하지 않게 합니다.
    student, tokenizer = load_student()
    print("student", evaluate(student, tokenizer, splits["validation"]))
    train(student, tokenizer, Dataset.from_json(str(TRACES_PATH)))
    print("hard label", evaluate(student, tokenizer, splits["validation"]))
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
