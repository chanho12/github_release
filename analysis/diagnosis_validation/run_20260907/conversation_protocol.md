# Conversation protocol

## Step 1
모든 방식에 동일 Evidence Card와 품질정보를 제공한다. A는 지표표+체크리스트, B는 일반 LLM, C는 동일 LLM+진단절차, C0는 규칙만 사용한다.

## Step 2
B와 C의 질문예산은 최대 3개로 맞춘다. 실제 익명 점포답변이 없으면 모두 `모름`이다. 시뮬레이션 답변은 scenario_registry에만 두고 실제 지역 발견에 합치지 않는다.

## Step 3
답변 출처를 STORE_PROVIDED / SIMULATED / UNKNOWN으로 기록하고 조건부 개선 후보를 갱신한다. 숫자·원인·성과를 새로 생성하지 않는다.
