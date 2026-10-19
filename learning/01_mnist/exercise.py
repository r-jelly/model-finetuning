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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KIND = "cnn"  # mlp 실행 후 cnn으로 바꿉니다.
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3

def prepare_data():
    """TODO: 학습/검증/평가 DataLoader 세 개를 반환하세요."""
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

    indices = torch.randperm(60000)
    train_data = Subset(train_data_all, indices=indices[:5000])
    valid_data = Subset(train_data_all, indices=indices[5000:6000])
    test_data = Subset(test_data, indices=torch.randperm(10000)[:1000])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, validation_loader, test_loader
    

def build_model(kind):
    """TODO: 입력 [B,1,28,28]에서 logits [B,10]을 반환하는 nn.Module을 만드세요."""
    if kind == "mlp":
        model = nn.Sequential(
            nn.Flatten(), # 입력 [B, 1, 28, 28] -> [B, 784]
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    elif kind == "cnn":
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), # [B,1,28,28] -> [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [B,16,28,28] -> [B,16,14,14]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # [B,16,14,14] -> [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [B,32,14,14] -> [B,32,7,7]
            nn.Flatten(),
            nn.Linear(in_features=32*7*7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
        )
    return model


def train(model, loader, optimizer, loss_fn):
    """TODO: optimizer.zero_grad, forward, loss, backward, step을 작성하세요."""
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(loader):.4f}")


def evaluate(model, loader):
    """TODO: eval/no_grad에서 정확도, 혼동행렬, 오답을 반환하세요."""
    model.eval()
    all_X = []
    prediction = []
    true = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            pred = logits.argmax(dim=-1)
            all_X.append(X)
            prediction.append(pred)
            true.append(y)

    all_X = torch.cat(all_X)
    prediction = torch.cat(prediction)
    true = torch.cat(true)
    accuracy = (prediction == true).sum() / len(prediction)
    conf_matrix = confusion_matrix(true.cpu().numpy(), prediction.cpu().numpy())
    wrong_answer = all_X[prediction != true]
    return accuracy, conf_matrix, wrong_answer


def check_environment():
    assert datasets.MNIST is not None
    x = torch.zeros(2, 1, 28, 28)
    assert x.flatten(1).shape == (2, 784)
    assert nn.CrossEntropyLoss()(torch.zeros(2, 10), torch.tensor([0, 9])).isfinite()
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    train_loader, validation_loader, test_loader = prepare_data()
    model = build_model(MODEL_KIND).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, loss_fn)
    print("validation", evaluate(model, validation_loader))
    # 모델과 설정 선택을 마친 뒤 test_loader를 최종 평가하세요.
    print("test", evaluate(model, test_loader))


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
