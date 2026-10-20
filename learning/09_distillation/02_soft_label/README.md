# Soft label

첫 행동: --check를 실행하세요. --run은 01이 저장한 teacher 풀이 파일이 필요합니다.
`distill_loss()`를 먼저 작성하고 `GKDTrainer.generalized_jsd_loss(..., beta=0.0)`와 같은 랜덤 텐서에서 값이 같은지 assert로 확인합니다. beta=0이 KL(teacher‖student)입니다.
causal LM의 logits는 다음 토큰 예측입니다. 위치 t의 logits와 t+1의 label이 짝이 되도록 한 칸 밀어 completion 위치만 남깁니다.
teacher와 student는 같은 input_ids를 받으므로 둘의 logits는 밀지 않은 같은 위치끼리 비교합니다.
Hinton은 temperature T를 쓰면 loss에 T²를 곱합니다. TRL 함수는 곱하지 않으므로 비교할 때 맞춥니다. TEMPERATURE=1로 시작합니다.
학습 루프는 직접 작성하거나 Trainer의 compute_loss를 바꿔 씁니다. 01과 같은 effective batch 8, optimizer step 100, lr, LoRA 설정을 지킵니다.
vocab이 15만 개라 logits가 큽니다. batch1에 gradient accumulation을 쓰고 최대 GPU 메모리를 기록하세요.
