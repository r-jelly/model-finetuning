"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
DATASET_ID = "stanfordnlp/sst2"
MODE = "full"  # frozen -> full -> lora
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-5  # 모드별 후보 3개는 README "실험 범위"
EPOCHS = 2
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/02_sentiment"
CHECKPOINT_DIR = ROOT / "checkpoints/02_sentiment"
DATA_DIR = ROOT / "data/02_sentiment"
RUN_NAME = f"{MODE}_lr{LEARNING_RATE:.0e}_seed{SEED}"  # outputs/checkpoints 아래 실행별 폴더 이름 (예: lora_lr5e-04_seed42)

def prepare_data():
    """TODO: train에서 학습2000/검증500, 공식 validation에서 평가872 데이터셋과 tokenizer를 반환하세요."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    collator = DataCollatorWithPadding(tokenizer)

    train_all = load_dataset(
        path=DATASET_ID,
        split='train',
        cache_dir=DATA_DIR
    )
    test_dataset = load_dataset(
        path=DATASET_ID,
        split='validation',
        cache_dir=DATA_DIR
    )
    train_all = train_all.shuffle(seed=SEED)
    train_dataset = train_all.select(range(0, 2000))
    valid_dataset = train_all.select(range(2000, 2500))
    
    def tokenize(example):
            return tokenizer(example['sentence'], truncation=True, max_length=MAX_LENGTH)
    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    return train_dataset, valid_dataset, test_dataset, tokenizer, collator


def build_model(mode):
    """TODO: 분류 모델과 frozen/full/lora의 학습 파라미터 범위를 설정하세요."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    if mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "full":
        pass
    elif mode == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=['q_lin', 'v_lin'],
            modules_to_save=['pre_classifier', 'classifier']
        )
        model = get_peft_model(model, config)
    else:
        raise ValueError()
    return model


def train(model, tokenizer, collator, dataset):
    """TODO: Trainer를 연결하고 학습된 모델을 저장하세요."""
    # TODO: 아래 TrainingArguments를 고치세요.
    #   - learning_rate·num_train_epochs → LEARNING_RATE·EPOCHS 상수 (RUN_NAME과 metrics.json이 이 상수를 씁니다)
    #   - output_dir=CHECKPOINT_DIR / RUN_NAME: 지정하지 않으면 실행한 폴더의 trainer_output/에 씁니다
    #   - save_strategy="no": 기본값은 500 step마다 optimizer 상태까지 저장합니다 (이어서 학습할 때만 필요)
    #   - logging_steps=10: 기본값 500이면 이번 학습(500 step)의 loss가 한 번만 기록됩니다
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        output_dir=CHECKPOINT_DIR / RUN_NAME,
        save_strategy='no',
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    # TODO: trainer.train() 뒤에 trainer.save_model()로 output_dir에 모델만 저장하세요.
    #   lora는 adapter와 modules_to_save만 저장됩니다. full과 폴더 크기를 비교해 보세요 (du -sh).
    trainer.save_model()
    # TODO: trainer.state.log_history(step별 loss 기록)를 반환하세요.
    return trainer.state.log_history


def evaluate(model, tokenizer, collator, dataset):
    """TODO: 정확도와 Macro-F1을 반환하세요."""
    trainer = Trainer(
        model=model,
        data_collator=collator,
        processing_class=tokenizer
    )
    out = trainer.predict(dataset)
    preds = out.predictions.argmax(axis=-1)
    labels = out.label_ids
    score_acc = accuracy_score(labels, preds)
    score_f1 = f1_score(labels, preds, average="macro")
    # TODO: save_results()가 쓸 문장별 결과도 dict에 담아 반환하세요. trainer.predict는 dataset 순서를 유지합니다.
    #   pred·label, prob(예측한 클래스의 softmax 확률), dataset["idx"], dataset["sentence"]
    #   힌트: torch.softmax(torch.from_numpy(out.predictions), dim=-1).max(dim=-1).values
    return {
        'accuracy': score_acc, 
        'macro_f1': score_f1,
        'pred': preds.tolist(),
        'label': labels.tolist(),
        'prob': torch.softmax(torch.from_numpy(out.predictions), dim=-1).max(dim=-1).values.tolist(),
        'idx': dataset['idx'],
        'sentence': dataset['sentence'],
    }


def check_reload(model, tokenizer, collator, dataset, checkpoint_path):
    """TODO: 저장한 폴더에서 모델을 새로 불러와 model과 accuracy가 같은지 assert하세요.

    frozen·full은 AutoModelForSequenceClassification, lora는 peft의 AutoPeftModelForSequenceClassification으로 불러옵니다.
    """
    reload_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint_path) \
                    if MODE != "lora" else AutoPeftModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=checkpoint_path)
    assert evaluate(model, tokenizer, collator, dataset)['accuracy'] == evaluate(reload_model, tokenizer, collator, dataset)['accuracy']


def save_results(log_history, cost, results):
    """TODO: OUTPUT_DIR / RUN_NAME 아래에 metrics.json과 predictions.jsonl을 저장하세요.

    metrics.json: 설정 상수, cost(학습 파라미터 수·시간·최대 GPU 메모리), log_history(학습 곡선용),
                  split별 accuracy·macro_f1·n·혼동행렬(sklearn confusion_matrix)
    predictions.jsonl: 한 줄에 {"split", "idx", "sentence", "label", "pred", "prob"} 하나.
    split 이름은 validation(train에서 뗀 500개)과 test(공식 validation 872개)입니다.
    """
    # raise NotImplementedError("OUTPUT_DIR / RUN_NAME 아래에 metrics.json과 predictions.jsonl을 저장하세요.")
    metrics = {
        "config": {
            "model_kind": MODE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE
        },
        "log_history": log_history,
        "cost": cost,
    }
    predictions = []

    for split, result in results.items():
        metrics[split] = {
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_f1'],
            'n': len(result['pred']),
            'confusion_matrix': confusion_matrix(result['label'], result['pred']).tolist()
        }
        for i, sentence, label, pred, prob in zip(result['idx'], result['sentence'], result['label'], result['pred'], result['prob']):
            predictions.append({
                'split': split,
                'index': i,
                'sentence': sentence,
                'label': label,
                'pred': pred,
                'prob': prob
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
    config = DistilBertConfig(vocab_size=64, dim=32, hidden_dim=64, n_layers=1, n_heads=4, num_labels=2)
    model = DistilBertForSequenceClassification(config)
    output = model(input_ids=torch.ones(2, 8, dtype=torch.long), labels=torch.tensor([0, 1]))
    assert output.logits.shape == (2, 2) and torch.isfinite(output.loss)
    assert f1_score([0, 1], [0, 1], average="macro") == 1.0
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    torch.manual_seed(SEED)
    train_data, validation_data, test_data, tokenizer, collator = prepare_data()
    model = build_model(MODE)
    # TODO: train() 앞뒤로 시간(time.perf_counter)과 최대 GPU 메모리(torch.cuda.reset_peak_memory_stats → max_memory_allocated)를 재고,
    #       학습 파라미터 수(requires_grad인 p.numel()의 합)와 함께 cost dict로 모으세요. → results.md 시간·메모리 칸
    torch.cuda.reset_peak_memory_stats() # 피크 메모리 초기화
    start_time = time.perf_counter() # 실행 시간 측정 시작
    
    train_losses = train(model, tokenizer, collator, train_data)
    
    end_time = time.perf_counter() # 실행 시간 측정 종료
    peak_memory_bytes = torch.cuda.max_memory_allocated() # 피크 메모리 측정
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 학습 가능한 파라미터 수 측정

    cost = {
        'run_time': end_time - start_time,
        'max_gpu_memory': peak_memory_bytes,
        'trainable_params': trainable_params,
    }

    valid_results = evaluate(model, tokenizer, collator, validation_data)
    print(f"Validation Acc: {valid_results['accuracy']:.3f}, Macro F1: {valid_results['macro_f1']:.3f}")
    # 설정 선택을 마친 뒤 test_data를 최종 평가하세요.
    test_results = evaluate(model, tokenizer, collator, test_data)
    print(f"Test Acc: {test_results['accuracy']:.3f}, Macro F1: {test_results['macro_f1']:.3f}")

    # 재로딩·결과 저장 (learning/README.md 작성 규칙 2·3·5)
    check_reload(model, tokenizer, collator, validation_data, CHECKPOINT_DIR / RUN_NAME)  # train()의 save_model()이 저장한 폴더
    # TODO: train()·evaluate()의 반환값을 받아 save_results(log_history, cost, {"validation": ..., "test": ...})를 호출하세요.
    # 그림은 notebooks/02_sentiment.ipynb에서 그립니다.
    save_results(train_losses, cost, {"validation": valid_results, "test": test_results})


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
