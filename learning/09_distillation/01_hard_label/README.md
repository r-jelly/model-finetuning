# Hard label

첫 행동: --check를 실행하고 계보 표의 Kim & Rush 초록을 읽은 뒤 `make_splits()`의 학습 문제 10개를 직접 확인하세요.
`load_teacher()`·`load_student()`·`evaluate()`를 먼저 작성해 teacher와 student의 검증 정답률을 잽니다. 평가에도 같은 chat template을 적용합니다.
`generate_traces()`는 teacher greedy 풀이 중 exact_reward=1인 것만 남깁니다. 저장 비율이 teacher의 학습 정답률이며 results.md에 기록합니다.
teacher 풀이는 `prompt`/`completion` 대화 형식으로 저장하고 answer/id는 학습 입력에서 제외합니다. 이 파일은 02에서 그대로 다시 씁니다.
풀이 파일을 바꾸면 01을 다시 학습해야 02와 비교할 수 있습니다. 처음에는 문제 8개로 형식과 생성 시간부터 확인하세요.
SFTConfig: batch1, accumulation8, max_length512, max_steps100, lr1e-4, completion_only_loss=True, gradient_checkpointing=True, report_to='none', push_to_hub=False.
DeepSeek-R1의 소형 모델도 큰 모델 풀이를 이렇게 SFT했습니다. student가 teacher의 정답 토큰 하나만 보고 다른 후보의 확률은 보지 못한다는 점을 기억하세요.
