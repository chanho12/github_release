# Neo4j + GraphRAG + OpenAI API 실행 안내

## 동작 구조

멀티라벨 모델이 월×시군구×업종의 수요 위축·객단가 압력·고객 편중·경쟁 압력을 계산한다. Neo4j에서 지역·업종·과거 관찰·점포동학·트렌드·마케팅 사례를 Cypher 관계 검색하고, OpenAI 임베딩 기반 벡터 검색과 합친다. 검색 context와 사업자의 추가 답변을 OpenAI Responses API에 보내 조건부 조언을 JSON으로 생성한다.

결과는 개별 점포의 폐업확률이 아니라 지역×업종 위험 신호다.

## 1. 설치

~~~bash
cd model
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python3 -m pip install -r requirements.txt
~~~

로컬 Neo4j는 [Docker Desktop](https://www.docker.com/get-started/) 설치 후 저장소의 docker-compose.yml을 쓰는 방법이 가장 단순하다. GUI를 원하면 [Neo4j Desktop 공식 설치 안내](https://neo4j.com/docs/desktop/current/installation/)를 따른다. Aura라면 로컬 설치 없이 발급받은 neo4j+s:// URI를 사용할 수 있다.

## 2. API 키와 접속정보

~~~bash
cp .env.example .env
~~~

model/.env에 다음 값을 입력한다.

~~~dotenv
OPENAI_API_KEY=본인의_OpenAI_API_Key
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=직접_정한_비밀번호
REQUIRE_NEO4J=true
~~~

코드는 model/.env를 자동으로 읽는다. 실제 .env는 Git에서 제외된다. 생성 답변에는 [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)를, 사례 벡터에는 [OpenAI Embeddings API](https://developers.openai.com/api/docs/guides/embeddings)를 사용한다.

## 3. Neo4j 시작과 최초 실행

Docker 사용 시:

~~~bash
docker compose up -d
~~~

Neo4j Browser는 http://localhost:7474 에서 연다. Desktop이나 Aura를 쓴다면 Compose를 실행하지 않고 해당 DB의 URI와 자격증명을 .env에 입력한다.

최초 한 번은 다음 명령이 환경변수·연결 확인, 그래프 적재, 사례 임베딩, 벡터 인덱스 생성, FastAPI 실행을 연속 수행한다.

~~~bash
PYTHONPATH=src python3 run_api.py --seed
~~~

이후에는 재적재 없이 실행한다.

~~~bash
PYTHONPATH=src python3 run_api.py
~~~

- 상태: http://localhost:8000/health
- Swagger: http://localhost:8000/docs
- neo4j_available, llm_available, embedding_available이 모두 true여야 완전한 Hybrid GraphRAG 모드다.

## 4. 호출 예시

~~~bash
curl -X POST http://localhost:8000/diagnose-region \
  -H 'Content-Type: application/json' \
  -d '{
    "sido": "서울특별시",
    "sigungu": "강남구",
    "industry": "제과점",
    "business_answers": {
      "recent_repeat_customers": "최근 2개월 감소",
      "margin_change": "배달 할인 후 감소"
    }
  }'
~~~

diagnosis.retrieval_mode이 hybrid_vector_and_cypher이면 관계·벡터 검색이 모두 작동한 것이다. advisor.status가 LLM_CALLED이면 Responses API 답변까지 생성됐다.

## 5. 프롬프트와 근거 통제

프롬프트는 src/risk_assistant/llm.py의 SYSTEM_INSTRUCTIONS에 있다.

- 전달된 evidence와 사업자 답변만 사용
- 관측 사실과 가능한 설명을 분리
- 모르는 원인은 확인 필요로 표시
- 외부 사례 성과를 사용자 예상 성과로 단정하지 않음
- 질문 최대 3개, 추천은 2~4주 실험과 KPI로 제시
- 사례를 사용하면 case_id와 source_url을 evidence_refs에 기록

## 6. 검증

~~~bash
PYTHONPATH=src python3 -m pytest -q
~~~

실제 OpenAI 과금 호출과 사용자의 Neo4j 연결은 .env가 있어야 하므로 저장소 자동 테스트에서는 실행하지 않는다.
