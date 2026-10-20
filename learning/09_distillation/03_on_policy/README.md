# On-policy

첫 행동: --check를 실행하고 계보 표의 GKD 초록과 Thinking Machines 글을 읽으세요. 01의 teacher 풀이는 쓰지 않습니다.
`reverse_kl_loss()`를 먼저 작성하고 `GKDTrainer.generalized_jsd_loss(..., beta=1.0)`와 같은 랜덤 텐서에서 값이 같은지 assert로 확인합니다.
`sample_completions()`는 eval 모드에서 sampling하고 old logprob를 저장하지 않습니다. 생성된 토큰은 고정된 데이터로 보고, 각 위치의 분포만 미분합니다.
생성 뒤 student를 train 모드로 되돌리고 같은 시퀀스에서 student logits(grad 있음)와 teacher logits(no_grad)를 구합니다. label shift는 02와 같습니다.
한 step마다 학습 문제 PROMPTS_PER_STEP=8개를 쓰고 optimizer step 100으로 01·02와 같은 횟수를 맞춥니다. 같은 문제가 반복돼도 됩니다.
step마다 loss, 생성 길이, 생성 답의 exact_reward 평균을 기록합니다. 08 GRPO처럼 보상으로 학습하지는 않으며 보상은 관찰용입니다.
on-policy step은 생성 때문에 01·02 step보다 느립니다. step 2로 시간을 재고 예산을 정하세요.
