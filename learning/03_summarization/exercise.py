"""학습 템플릿. --check는 환경만 확인하며 TODO 구현을 검증하지 않습니다."""
import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5Config, T5ForConditionalGeneration
from rouge_score import rouge_scorer

SEED = 42
MODEL_ID = "google-t5/t5-small"
DATASET_ID = "EdinburghNLP/xsum"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
MAX_STEPS = 500
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-4
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/03_summarization"
CHECKPOINT_DIR = ROOT / "checkpoints/03_summarization"
DATA_DIR = ROOT / "data/03_summarization"
RUN_NAME = f"lr{LEARNING_RATE:.0e}_seed{SEED}_steps{MAX_STEPS}_batch{BATCH_SIZE * ACCUMULATION_STEPS}"

def prepare_data():
    """TODO: 고정된 학습/검증/평가 데이터셋과 tokenizer를 반환하세요."""
    dataset = load_dataset(DATASET_ID, cache_dir=DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    collator = DataCollatorForSeq2Seq(tokenizer)

    def preprocess(examples):
        # 입력에 'summarize: ' 접두사 붙이기
        inputs = ["summarize: " + doc for doc in examples['document']]
        labels = tokenizer(text_target=examples['summary'], truncation=True, max_length=MAX_INPUT_LENGTH) # text_target: encdec 모델에서 decoder tokenize 적용 명시
        data = tokenizer(inputs, truncation=True, max_length=MAX_INPUT_LENGTH)
        data['labels'] = labels['input_ids']
        return data

    train_dataset = dataset['train'].shuffle(seed=SEED).select(range(1000)).map(preprocess, batched=True)
    valid_dataset = dataset['validation'].shuffle(seed=SEED).select(range(100)).map(preprocess, batched=True)
    test_dataset = dataset['test'].shuffle(seed=SEED).select(range(100)).map(preprocess, batched=True)
    
    return train_dataset, valid_dataset, test_dataset, tokenizer, collator


def evaluate(model, tokenizer, collator, dataset):
    """TODO: greedy 생성 후 ROUGE-L과 원문/정답/예측을 반환하세요."""
    args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR / RUN_NAME,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=collator,
    )
    out = trainer.predict(dataset, max_new_tokens=MAX_TARGET_LENGTH, num_beams=1)

    # 배치 간 길이가 다른 경우, 짧은 쪽의 배치 끝에 `-100`을 추가하게 됨
    pred_ids = np.where(out.predictions != -100, out.predictions, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    trues = dataset['summary']

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, true in zip(preds, trues):
        score = scorer.score(true, pred)
        scores.append(score['rougeL'].fmeasure)
    rouge_score = np.mean(scores)

    return {
        'rougeL': rouge_score,
        'id': dataset['id'],
        'origin': dataset['document'],
        'true': dataset['summary'],
        'pred': preds
    }


def train(model, tokenizer, collator, dataset):
    """TODO: target padding을 loss에서 제외하고 요약 학습을 작성하세요."""
    train_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR / RUN_NAME,
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        save_strategy='no',
        logging_steps=10,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model()
    return trainer.state.log_history


def save_results(log_history, cost, results):
    """TODO: OUTPUT_DIR / RUN_NAME 아래에 metrics.json, predictions.jsonl을 저장하세요."""
    metrics = {
        "train_losses": log_history,
        "train_cost": cost
    }
    predictions = []
    for split, result in results.items():
        metrics[split] = {
            'rougeL': result['rougeL'],
            'n': len(result['pred'])
        }
        for i, origin, true, pred in zip(result['id'], result['origin'], result['true'], result['pred']):
            predictions.append({
                'split': split,
                'index': i,
                'origin': origin,
                'true': true,
                'pred': pred
            })

    save_path = OUTPUT_DIR / RUN_NAME
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / 'metrics.json').write_text(json.dumps(metrics, indent=4))
    (save_path / 'predictions.jsonl').write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in predictions)
    )


def check_environment():
    config = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4, decoder_start_token_id=0, pad_token_id=0)
    model = T5ForConditionalGeneration(config)
    loss = model(input_ids=torch.ones(1, 8, dtype=torch.long), labels=torch.tensor([[2, 3, 1, -100]])).loss
    assert torch.isfinite(loss)
    score = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score("a cat runs", "a cat runs")
    assert score["rougeL"].fmeasure == 1.0
    print("Environment OK. TODO 학습 로직은 아직 검증하지 않았습니다.")


def main():
    train_data, validation_data, test_data, tokenizer, collator = prepare_data()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    before_valid = evaluate(model, tokenizer, collator, validation_data)

    torch.cuda.reset_peak_memory_stats() # 피크 메모리 초기화
    start_time = time.perf_counter()

    train_losses = train(model, tokenizer, collator, train_data)
    
    end_time = time.perf_counter()
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cost = {
        'run_time': end_time - start_time,
        'max_gpu_memory': peak_memory_bytes,
        'trainable_params': trainable_params,
    }

    after_valid = evaluate(model, tokenizer, collator, validation_data)
    print(f"Validation Changes: Before={before_valid['rougeL']}, After={after_valid['rougeL']}")
    # 설정 선택 후 같은 test_data로 기준선과 학습 모델을 최종 평가하세요.
    test = evaluate(model, tokenizer, collator, test_data)
    save_results(train_losses, cost, {'valid': after_valid, 'test': test})


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
