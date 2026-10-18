"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/01_mnist"
OUTPUT_DIR = ROOT / "outputs/01_mnist"
MODEL_KIND = "mlp"  # mlp 실행 후 cnn으로 바꿉니다.
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3

def prepare_data():
    """TODO: 학습/검증/평가 DataLoader 세 개를 반환하세요."""
    # raise NotImplementedError('학습/검증/평가 DataLoader 세 개를 반환하세요.')
    train_data_all = datasets.MNIST(
        root=DATA_DIR, 
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    indices = torch.randint(0, 60000, (6000,))
    train_data = Subset(train_data_all, indices=indices[:5000])
    valid_data = Subset(train_data_all, indices=indices[5000:])
    
    


def build_model(kind):
    """TODO: 입력 [B,1,28,28]에서 logits [B,10]을 반환하는 nn.Module을 만드세요."""
    raise NotImplementedError('입력 [B,1,28,28]에서 logits [B,10]을 반환하는 nn.Module을 만드세요.')


def train(model, loader):
    """TODO: optimizer.zero_grad, forward, loss, backward, step을 작성하세요."""
    raise NotImplementedError('optimizer.zero_grad, forward, loss, backward, step을 작성하세요.')


def evaluate(model, loader):
    """TODO: eval/no_grad에서 정확도, 혼동행렬, 오답을 반환하세요."""
    raise NotImplementedError('eval/no_grad에서 정확도, 혼동행렬, 오답을 반환하세요.')


def check_environment():
    assert datasets.MNIST is not None
    x = torch.zeros(2, 1, 28, 28)
    assert x.flatten(1).shape == (2, 784)
    assert nn.CrossEntropyLoss()(torch.zeros(2, 10), torch.tensor([0, 9])).isfinite()
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    train_loader, validation_loader, test_loader = prepare_data()
    model = build_model(MODEL_KIND).to("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader)
    print("validation", evaluate(model, validation_loader))
    # 모델과 설정 선택을 마친 뒤 test_loader를 최종 평가하세요.


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
