# 과거 데이터·GraphRAG·Collaborative Filtering 설계

## 판단

- BC와 동일한 과거 소비 집계는 공개 외부자료로 복원할 수 없다. 주최측의 18~24개월 추가제공이 가장 중요하다.
- LOCALDATA 점포 스냅샷의 인허일·폐업일로 제과점·중국음식 2019-01~2026-06 월별 개업·폐업·영업점포 46,080행을 재구성했다.
- 주민인구·CPI·기상은 공식 시계열을 확장할 수 있다. 국세청·한국부동산원은 과거 게시물 아카이브를 월·분기별로 수동 확보해야 한다.
- 과거 상권 수치는 유사 상권 검색에는 쓸 수 있지만, 그 자체가 Collaborative Filtering 데이터는 아니다.

출처·확보 상태·그래프 역할은 `historical_data_sources.csv`에 정리했다.

## Neo4j 그래프

```text
(:Region)<-[:OBSERVED_IN]-(:Observation)-[:FOR_INDUSTRY]->(:Industry)
(:Region)<-[:OBSERVED_IN]-(:MarketHistory)-[:FOR_INDUSTRY]->(:Industry)
(:Case)-[:ADDRESSES]->(:Problem)
(:Case)-[:EVIDENCES]->(:Trend)
(:Business)-[:LOCATED_IN]->(:Region)
(:Business)-[:IN_INDUSTRY]->(:Industry)
(:Business)-[:TRIED]->(:Intervention)-[:RESULTED_IN]->(:Outcome)
```

GraphRAG은 현재 위험확률을 기준으로 다음을 검색한다.

1. 해당 지역×업종의 과거 개업·폐업·순점포 경로
2. 현재 위험 프로필이 유사한 지역
3. 위험 유형을 다룬 최근·과거 트렌드
4. 트렌드를 실제로 적용한 기업 공식 사례와 주의사항
5. 추후 축적될 유사 사업자의 개선안 성과

## CF 전환 조건

현재는 사업자 반응이 없으므로 `case-based retrieval`을 사용한다. 다음 로그가 최소 100건 이상, 개선안별·업종별로 충분히 쌓인 후에만 CF를 학습한다.

- 익명 `business_id`
- 제안·선택한 `intervention_id`
- 실행 시작·종료일
- 비교집단 대비 CNT·AMT·마진·재방문 증분
- 미수행·중도이탈 여부

초기 추천점수는 `위험 유사도 + 업종·지역 적합도 + 근거의 최근성`으로 구성한다. 피드백 축적 후에는 `유사 사업자의 실제 증분성과`를 추가한다.

## 만들어진 과거 자료

- `발표용/data/외부데이터/재현용_가공패널/과거_점포동학_201901_202606.csv`: 46,080행
- `발표용/data/외부데이터/시장트렌드_2026/과거_트렌드_타임라인.csv`: 2023~2026 공식 사례 10건
- `artifacts/historical_observations.csv`: BC 기반 t→t+1 학습 관측
- `neo4j/schema.cypher`: 최종 그래프 제약·관계
