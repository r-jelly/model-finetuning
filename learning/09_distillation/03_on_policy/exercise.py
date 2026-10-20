"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm8k import make_splits, exact_reward
from trl.experimental.gkd import GKDTrainer

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/09_distillation/on_policy"
SEED = 42
MAX_NEW_TOKENS = 256
MAX_STEPS = 100
PROMPTS_PER_STEP = 8
LEARNING_RATE = 1e-4

def load_teacher():
    """TODO: teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요."""
    raise NotImplementedError('teacher tokenizer와 causal LM을 bf16·eval 모드로 로드하고 파라미터를 동결해 둘 다 반환하세요.')


def load_student():
    """TODO: student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요."""
    raise NotImplementedError('student tokenizer와 causal LM을 로드하고 LoRA를 설정해 둘 다 반환하세요.')


def evaluate(model, tokenizer, rows):
    """TODO: model.eval() 후 01과 같은 greedy 생성으로 정답률을 반환하세요."""
    raise NotImplementedError('model.eval() 후 01과 같은 greedy 생성으로 정답률을 반환하세요.')


def sample_completions(student, tokenizer, rows):
    """TODO: model.eval() 후 요청마다 답변 하나를 sampling하고 prompt+completion input_ids와 completion만 남긴 labels를 반환하세요."""
    raise NotImplementedError('model.eval() 후 요청마다 답변 하나를 sampling하고 prompt+completion input_ids와 completion만 남긴 labels를 반환하세요.')


def reverse_kl_loss(student_logits, teacher_logits, labels):
    """TODO: labels != -100 위치에서 vocab 전체의 KL(student‖teacher)을 토큰 평균하세요."""
    raise NotImplementedError('labels != -100 위치에서 vocab 전체의 KL(student‖teacher)을 토큰 평균하세요.')


def train(student, teacher, tokenizer, rows):
    """TODO: MAX_STEPS 동안 sample → teacher logits(no_grad) → reverse_kl_loss → update를 반복하고 adapter를 저장하세요."""
    raise NotImplementedError('MAX_STEPS 동안 sample → teacher logits(no_grad) → reverse_kl_loss → update를 반복하고 adapter를 저장하세요.')


def check_environment():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    prompt = torch.ones(1, 4, dtype=torch.long)
    sampled = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=True, max_new_tokens=4, pad_token_id=0)
    logits = model(input_ids=sampled).logits
    labels = torch.full_like(sampled, -100)
    labels[:, 4:] = sampled[:, 4:]
    reference = GKDTrainer.generalized_jsd_loss(logits, torch.randn_like(logits), labels, beta=1.0)
    assert sampled.shape[1] <= 8 and torch.isfinite(reference)
    assert exact_reward([[{"role": "assistant", "content": "#### 12"}]], [12]) == [1.0]
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    splits = make_splits()
    teacher, _ = load_teacher()
    student, tokenizer = load_student()
    train(student, teacher, tokenizer, splits["train"])
    print("on-policy", evaluate(student, tokenizer, splits["validation"]))
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
