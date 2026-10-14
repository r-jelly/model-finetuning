# 06 · 다음 토큰 예측 추가 학습

모델 `openai-community/gpt2`, 데이터 `Salesforce/wikitext`, config `wikitext-2-raw-v1`. 첫 세션 예상 3시간.

## 첫 행동 · 10분

루트에서 `uv run python learning/06_continued_pretraining/exercise.py --check`를 실행하고 첫 TODO를 읽으세요.

## 직접 작성할 순서

1. 공식 train/validation/test 분할을 유지하고 train 약20만 토큰, validation/test 각 약2만 토큰을 고정합니다.
2. 문서 사이에 EOS를 넣고 학습 원문만 tokenizer로 변환합니다. padding과 EOS가 같은 ID여도 attention_mask로 실제 padding만 -100 처리합니다.
3. 문장별 padding과 길이128로 이어 묶기 두 방식을 만듭니다.
4. 동일한 유효 학습 토큰 예산으로 추가 학습합니다. 학습 step만 같다고 같은 데이터 예산은 아닙니다.
5. 고정된 동일 평가 blocks에서 학습 전/후 token-weighted loss와 exp(loss)를 계산합니다.

## 평가와 완료 조건

Loss·perplexity·유효 토큰/초. Loss는 유효 label 수로 가중 평균하고, 생성문 10개도 확인합니다.

## 실행

```bash
uv run python learning/06_continued_pretraining/exercise.py --run
```

TODO 함수는 `NotImplementedError`로 멈춥니다. 구현 후 다시 실행하세요.

데이터: `data/06_continued_pretraining/`, 체크포인트: `checkpoints/06_continued_pretraining/`, 결과: `outputs/06_continued_pretraining/`.

## 실험 범위

시작 설정: block=128, batch=4, lr=5e-5. causal LM label shift는 모델 내부에서 수행하므로 두 번 shift하지 않습니다.

## 공식 자료

[GPT-2](https://huggingface.co/openai-community/gpt2) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) · [Causal LM](https://huggingface.co/docs/transformers/tasks/language_modeling)
