"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
import json
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
CHECKPOINT_DIR = ROOT / "checkpoints/01_mnist"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KIND = "cnn"  # mlp 실행 후 cnn으로 바꿉니다.
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-3
RUN_NAME = f"{MODEL_KIND}_seed{SEED}"  # outputs/checkpoints 아래 실행별 폴더 이름

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
    train_losses = []
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
            train_losses.append(loss.item())
        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(loader):.4f}")
    return train_losses


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
    return accuracy, conf_matrix, wrong_answer, prediction.cpu(), true.cpu(), [int(i) for i in loader.dataset.indices]


def save_checkpoint(model):
    """TODO: model.state_dict()를 CHECKPOINT_DIR / RUN_NAME / "model.pt"에 저장하고 그 경로를 반환하세요."""
    save_path = CHECKPOINT_DIR / RUN_NAME
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        save_path / "model.pt"
    )
    return save_path / "model.pt"


def check_reload(model, loader, checkpoint_path):
    """TODO: 새 build_model(MODEL_KIND)에 저장한 state_dict를 불러와 model과 정확도가 같은지 assert하세요."""
    reload_model = build_model(MODEL_KIND).to(DEVICE)
    reload_model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    origin_acc = evaluate(model, loader)[0]
    reload_acc = evaluate(reload_model, loader)[0]
    assert origin_acc == reload_acc


def save_results(train_losses, results):
    """TODO: OUTPUT_DIR / RUN_NAME 아래에 metrics.json과 predictions.jsonl을 저장하세요.

    metrics.json: 설정 상수, split별 accuracy·n·혼동행렬, step별 train loss(학습 곡선용)
    predictions.jsonl: 한 줄에 {"split", "mnist_index", "label", "pred"} 하나.
    이미지 대신 원본 MNIST 인덱스(Subset.indices)를 저장해야 노트북에서 이미지를 다시 찾을 수 있습니다.
    """
    # raise NotImplementedError('OUTPUT_DIR / RUN_NAME 아래에 metrics.json과 predictions.jsonl을 저장하세요.')
    metrics = {
        "config": {"model_kind": MODEL_KIND, "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lerning_rate": LEARNING_RATE},
        "train_losses": train_losses
    }
    predictions = []
    for split, (accuracy, conf_matrix, _, prediction, true, indices) in results.items():
        metrics[split] = {
            'accuracy': accuracy.item(),
            'n': len(true),
            'confusion_matrix': conf_matrix.tolist(),
        }
        for idx, label, pred in zip(indices, true.tolist(), prediction.tolist()):
            predictions.append({
                'split': split,
                'mnist_index': idx,
                'label': label,
                'pred': pred
            })

    save_path = OUTPUT_DIR / RUN_NAME
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(save_path / 'predictions.jsonl', 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False))
            f.write("\n")


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
    train_losses = train(model, train_loader, optimizer, loss_fn)
    validation_result = evaluate(model, validation_loader)
    print("validation accuracy", validation_result[0])
    # 모델과 설정 선택을 마친 뒤 test_loader를 최종 평가하세요.
    test_result = evaluate(model, test_loader)
    print("test accuracy", test_result[0])

    # 저장·재로딩 (learning/README.md 작성 규칙 2·3·5)
    checkpoint_path = save_checkpoint(model)
    check_reload(model, validation_loader, checkpoint_path)
    # TODO: train()이 step별 loss를, evaluate()가 예측·정답을 함께 반환하게 바꾼 뒤 save_results(...)를 호출하세요.
    # 그림은 notebooks/01_mnist.ipynb에서 그립니다.
    save_results(train_losses, {'validation': validation_result, 'test': test_result})


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
