# BC 상권 복합위험 진단 + Hybrid GraphRAG

이 폴더는 발표 노트북과 분리된 **모델·Neo4j·GraphRAG·LLM API** 영역이다. 데이터 기술통계느 `../code/BC카드_업종별_전체분석_발표용.ipynb`에서만 수행한다.

## 전체 구조

```text
최근 3개월 지역×업종 시계열
  → 다음 달 수요위험·객단가압력·고객편중 다중라벨
  → 주요 위험 1개 + 동반 위험
  → Neo4j Cypher 관계검색
  → OpenAI Embedding 벡터검색
  → Reciprocal Rank Fusion
  → 지역·업종 과거·트렌드·실제마케팅 사례 결합
  → OpenAI Responses API 조건부 진단·추가질문·실행안·KPI
```

모델의 출력은 개벼 점포 폐업확률이 아니라 **월×시군구×업종 위험신호**다.

## 폴더 구조

```text
model/
├── README.md
├── .env.example
├── requirements.txt
├── train.py
├── benchmark_multilabel.py
├── benchmark_temporal_multilabel.py
├── build_historical_graph_data.py
├── seed_graph.py
├── demo.py
├── docker-compose.yml
├── neo4j/
│   └── schema.cypher
├── artifacts/
├── tests/
└── src/risk_assistant/
    ├── dataset.py
    ├── temporal.py
    ├── modeling.py
    ├── inference.py
    ├── embeddings.py
    ├── graph.py
    ├── recommender.py
    ├── llm.py
    ├── service.py
    └── api.py
```

## 1. 로컬 데모

API 키·Neo4j 없이도 전체 파이프라인을 검증한다.

```bash
cd 발표용/model
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 train.py
PYTHONPATH=src python3 benchmark_multilabel.py
PYTHONPATH=src python3 benchmark_temporal_multilabel.py
PYTHONPATH=src python3 demo.py
```

이 모드에서는 CSV fallback으로 작동하며, LLM은 호출하지 않고 `LLM_NOT_CALLED`룰 반환한다.

## 2. API 환경변수 설정

`.env.example`을 보고 시키릇을 환경변수로 설정한다. 시키릇은 `.env`·노트북·Git에 저장하지 않는다.

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5-mini'
export OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
export OPENAI_EMBEDDING_DIMENSIONS='1536'

export NEO4J_URI='neo4j://localhost:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='...'
export REQUIRE_NEO4J='true'
```

Neo4j Aura는 `NEO4J_URI`에 `neo4j+s://...`ᆯ 넣는다.

## 3. Neo4j 질행·적재

로컬 Docker Neo4j룰 쓸 경우:

```bash
docker compose up -d
PYTHONPATH=src python3 seed_graph.py
```

`seed_graph.py`가 수행하는 일:

1. Neo4j 제약조건 생성
2. 지역·업종·월별관찰·점포동학·트렌드·사례 적재
3. `Case.embedding` 생성
4. `case_embedding` Neo4j Vector Index 생성

`OPENAI_API_KEY`가 없으면 1~2번만 수행하고, 입베딩·벡터색은 건너뚼다.

## 4. FastAPI 실행

```bash
PYTHONPATH=src uvicorn risk_assistant.api:app --host 0.0.0.0 --port 8000
```

상태 확인:

```bash
curl http://localhost:8000/health
```

진단:

```bash
curl -X POST http://localhost:8000/diagnose-region \
  -H 'Content-Type: application/json' \
  -d '{
    "sido": "서울특별시",
    "sigungu": "강남구",
    "industry": "제과점",
    "business_answers": {
      "recent_repeat_customers": "감소"
    }
  }'
```

응답의 `retrieval_mode`이 `hybrid_vector_and_cypher`면 벡터검색과 관계검색이 모두 사용된 것이다.

## 5. 검증

```bash
PYTHONPATH=src python3 -m pytest -q
```

실제 API 키루 불가능하면 `.env`는 커밋하지 않는다. 지금의 `.gitignore`는 이미 이를 제외한다.

## 6. 해석 경계

- 복합위험은 개벼 점포의 폐업확률이 아니다.
- 벡터검색 점수는 사례 건수에서만 사용한다.
- 2026-01~06 육 개월 데이터로 학습한 기준선이므로 배포성능 주장은 아니다.
- 시전분ᄅ는 관계가 아니라며, 제안된 실행안을 사업자의 성과로 단정하지 않는다.
