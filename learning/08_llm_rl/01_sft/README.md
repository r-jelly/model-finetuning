# SFT

첫 행동: --check를 실행하고 load_model()을 작성하세요. 모델 다운로드는 --run에서 직접 작성한 코드가 실행될 때 발생합니다.
train rows에는 prompt/completion을 전달하고 answer/id는 학습 입력에서 제외하세요.
SFTConfig: batch1, accumulation4, max_length128, lr1e-4, gradient_checkpointing=True, report_to='none', push_to_hub=False.
completion_only_loss=True로 요청 토큰에 대한 loss를 제외합니다. 동일한 chat template을 평가에도 적용합니다.
전체 모델 대신 LoRA adapter를 학습하고 CHECKPOINT_DIR에 저장하세요. 원본 모델을 저장하는 것으로 혼동하지 마세요.
원본/SFT 정답률과 최대 GPU 메모리를 results.md에 기록하세요.
