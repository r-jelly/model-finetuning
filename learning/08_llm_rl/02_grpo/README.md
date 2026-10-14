# GRPO

첫 행동: --check로 API를 확인하세요. --run은 먼저 저장된 SFT adapter가 필요합니다.
학습 dataset에는 prompt와 answer만 남기고 모범 completion은 제거합니다.
GRPOTrainer의 reward_funcs=exact_reward, processing_class=tokenizer를 연결합니다.
이미 SFT adapter를 로드한 모델에 새 LoRA를 중복 생성하지 않습니다.
설정은 상위 README를 따르되 처음에는 max_steps=2로 메모리와 저장/재로딩을 확인하세요.
prompt 길이를 128 이내로 사전 확인하고 completion은 최대8토큰으로 시작합니다.
같은 요청의 두 답변이 같은 reward만 받으면 상대적인 학습 신호가 없습니다.
greedy 정답률과 샘플링 reward는 다른 지표이므로 각각 기록합니다.
