# Model Fine-tuning

uv 기반 PyTorch·Hugging Face 파인튜닝 워크스페이스.

## 시작

```bash
cd ~/Develop/model-finetuning
uv sync --locked
source .venv/bin/activate
```

가상환경을 활성화하지 않아도 `uv run`으로 실행할 수 있습니다.
`pyproject.toml`은 의존성과 CUDA 인덱스를, `uv.lock`은 설치 버전을 관리합니다.
Python 3.12.12를 사용하며 PyTorch는 공식 CUDA 12.8 인덱스에서 설치합니다.

## 설치된 도구

- PyTorch, torchvision, torchaudio
- Transformers, Datasets, Accelerate, PEFT, TRL, huggingface-hub
- bitsandbytes (4비트/8비트 양자화), safetensors, sentencepiece, protobuf
- JupyterLab, ipykernel (기본 dev 의존성)
- Gymnasium classic-control, Sentence Transformers, scikit-learn, rouge-score, matplotlib

## Hugging Face 로그인

```bash
uv run hf auth whoami
uv run hf auth login
```

인증은 위 whoami 명령으로 확인합니다. 로그인은 터미널에서 직접 진행하세요.
토큰은 프로젝트 파일에 저장하지 않습니다.

## 환경 검증

```bash
uv run --locked python scripts/check_environment.py
uv pip check
```

검증 스크립트는 외부 모델을 다운로드하지 않고 CUDA, 패키지 import,
작은 GPT-2 모델의 LoRA 학습 단계와 bitsandbytes NF4 GPU 연산을 확인합니다.
설치 시 RTX 4070 Ti (12GB), BF16 지원, 위 검증 통과를 확인했습니다.
실제 모델 학습은 모델 크기·시퀀스 길이·배치 크기에 따라 메모리 요구량이 달라집니다.

## 노트북

```bash
uv run jupyter lab
```

VS Code에서 노트북을 열 경우 `.venv/bin/python`을 인터프리터로 선택합니다.

## 폴더

- `scripts/`: 학습·평가·환경 확인 코드
- `notebooks/`: 탐색용 노트북
- `data/`: 로컬 데이터
- `checkpoints/`: 학습 체크포인트
- `outputs/`: 평가 결과와 산출물

가상환경, 데이터, 체크포인트, 결과, 인증 정보는 `.gitignore`에서 제외합니다.

## 참고 문서

- [uv의 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Hugging Face CLI와 인증](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

## 직접 작성하는 학습 과제

[학습 시작 안내](learning/README.md)에서 8개 과제와 실행 순서를 확인하세요.
각 과제의 exercise.py는 TODO 템플릿이며, 학습 로직은 직접 구현합니다.

```bash
uv run python learning/01_mnist/exercise.py --check
uv run python scripts/check_learning_templates.py
uv run python -m unittest discover -s tests -v
```

CartPole은 random → REINFORCE → PPO, LLM은 SFT → GRPO 단계 폴더로 나뉩니다.
환경 확인은 다운로드 없이 실행합니다. 실제 모델·데이터는 각 과제를 시작할 때 받습니다.
