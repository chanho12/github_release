# 2026 업종별 시장트렌드 자료

기준일: 2026-09-12  
BC 관찰기간: 2026-01~06

## 폴더 구성

| 파일 | 내용 | 사용 여부 |
|---|---|---|
| `농림축산식품부_2026_외식산업_환경분석.pdf` | 2026 외식산업 환경·트렌드 공식 보고서 | 외부 트렌드 가설 설정 |
| `2026_업종별_외부트렌드_근거.csv` | 11개 BC 업종별 트렌드·마케팅 가설·출처 | 숫자 건강도 점수에는 미사용 |
| `실제_마케팅_사례_근거표.csv` | 기업 공식 자료로 확인한 10개 실제 사례·성과·KPI·주의사항 | 상황별 전략 가설 선정 |
| `과거_트렌드_타임라인.csv` | 2023~2026 실제 사례를 Neo4j Trend–Case 관계용으로 정규화 | GraphRAG 시대별 사례 검색 |
| `source_sha256.txt` | 보고서 고정 스냅샷 SHA-256 | 재현 검증 |

## 공식 출처와 재취득

### 농림축산식품부·aT 외식산업 환경분석

- 공식 원문: <https://www.mafra.go.kr/bbs/home/798/594990/download.do>
- 확인한 2026 핵심 트렌드: 서바이벌 다이닝, 진정성 있는 미식, 마이 헬시 다이닝, 가성비·가치비
- 추가 사업 키워드: 경력상품, 집밥경제, 초미세가격, 올데이 올라운더, K-푸드 투어
- 재취득: 위 URL을 브라우저에서 열어 PDF를 저장한다.

### BGF리테일 2026 공식 판매신호

- [2026-01-12 트렌드 메뉴 HMR 협업](https://www.bgfretail.com/press/view/?id=1798)
- [2026-02-03 차별화 베이커리 매출](https://www.bgfretail.com/press/view/?id=1817)

기업 보도자료는 해당 기업의 판매신호이며 전체 시장 표본은 아니다. 따라서 BC 결제흐름을 대체하지 않고, 상품·메뉴 가설을 설정하는 보조 근거로만 사용한다.

### 기업 공식 실제 마케팅 사례

| 구분 | 실제 사례 | 공식 출처 |
|---|---|---|
| 2026 직접 | CU 명장 협업·가성비 베이커리 | <https://www.bgfretail.com/press/view/?id=1817> |
| 2026 직접 | CU 방송·SNS 화제 메뉴의 HMR 상품화 | <https://www.bgfretail.com/press/view/?id=1798> |
| 2026 직접 | 애슐리퀸즈 딸기 시즌 콘텐츠 | <https://www.eland.co.kr/news/viewNews?cate=P&cateIdx=915> |
| 2026 직접 | 애슐리퀸즈 성수 테스트베드 매장 | <https://www.eland.co.kr/news/viewNews?cate=P&cateIdx=911> |
| 2026 직접 | 애슐리퀸즈 월별 제철 콘텐츠 | <https://www.eland.co.kr/mgzn/viewMgzn?categoryIdx=2&mgznIdx=1537> |
| 2026 선행 | CU 배달·픽업 퀵커머스 | <https://www.bgfretail.com/press/view/?id=1752> |
| 2026 선행 | CU 구체 원료 기반 건강 HMR | <https://www.bgfretail.com/press/view/?id=1721> |
| 선행 참고 | CU 품목별 구독할인 | <https://www.bgfretail.com/press/view/?id=1075> |
| 선행 참고 | CU 점포별 CRM·마감세일·타깃쿠폰 | <https://www.bgfretail.com/press/view/?id=442> |
| 선행 참고 | 파리바게뜨 SNS 부정 피드백 반복 품목 개선 | <https://www.spc.co.kr/img/front/sub/esg/esg_management/report/2025%20SPC%EA%B7%B8%EB%A3%B9%20%EC%A7%80%EC%86%8D%EA%B0%80%EB%8A%A5%EA%B2%BD%EC%98%81%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf> |

재취득은 표의 공식 URL에서 원문을 열어 CSV의 `reported_result`와 비교한다. 기업 페이지가 개편되면 사이트 내에서 브랜드·발표일·제품명을 함께 검색한다. 사례표는 `python3 scripts/build_external_marketing_casebook.py`로 다시 생성한다.

## 트렌드 근거 사용 원칙

1. 외부 보고서에서 트렌드 가설을 정한다.
2. BC에서 실질 AMT, CNT, 건당금액, 지역 확산, 연령·외국인 비중이 가설과 같이 움직이는지 검증한다.
3. 상품·메뉴·시간대 정보가 없으면 `검증되지 않은 마케팅 가설`로 남긴다.
4. 외부 트렌드 키워드는 `장사 건강도 점수`에 넣지 않는다.
5. 기업 공식 성과는 해당 기업의 사례이며, 우리 대상에서는 A/B 테스트로 증분효과를 다시 확인한다.
