# 「같은 매출 변화, 다른 처방」 진단 구현·비교검증 보고서

## 1. 검증 결론

동일 Evidence Card에서 규칙 기반 C0는 일반 지표표 A보다 **적용 규칙, 추가 질문, 판단 유보 조건, 답변 분기, 확인 지표를 재현 가능하게 구조화했다.** 그러나 이 구조가 사람의 판단을 실제로 개선하는지는 아직 입증되지 않았다. B·C LLM은 API·모델·예산이 지정되지 않아 `NOT_RUN_API`, 익명 사람평가는 `AWAITING_HUMAN_EVAL`이다.

따라서 현재 주제는 지역×업종 수준의 **점검 지원 시제품**까지 방어 가능하다. 맞춤형 개선효과·폐업예측·개별 점포 원인 확정은 방어할 수 없다.

## 2. 기존 주장 재현·수정

| previous_claim | confirmed_basis | corrected_expression | status | evidence |
|---|---|---|---|---|
| 지역총량·업종 불일치 679 | 679/2327 | 정반대 496, 정체 포함 183 | EXECUTED | region_total_vs_industry.csv |
| 지역총량 상승·업종 감소 | 432 | 432/2327=18.56%; 432/2027=21.31%; 조합 비율 | EXECUTED | region_total_vs_industry.csv |
| 대표 사례 4쌍 | 성능 수치로 사용 불가 | 구조차이 상위 개발사례; Pair 1·3만 세 비교항목이 다르고 Pair 2·4는 모두 같음 | EXECUTED | development_pair_review.csv |
| 원단위/로그 성장착시 | 10/9 | 별도 정의 유지; 다른 플래그와 합산 금지 | EXECUTED | claim_reproduction.csv |
| 3방법 일치 | 강건성 | 동일 6개월 재사용; 독립 검증 아님 | EXECUTED | claim_reproduction.csv |
| within/mix 63.7/36.3 | 기여율 | 지역별 절대기여 비중 중앙값; 전국 총액 아님 | EXECUTED | claim_reproduction.csv |
| 연령 구성 | 숫자연령 | 외국인 포함 전체와 국내개인 민감도 병기; 주민과 동일시 금지 | EXECUTED | BC raw crosstab |

679개 불일치는 정반대 496개와 정체 포함 차이 183개다. 지역총량 상승·업종 감소 432개는 전체 2,327조합의 18.56%, 지역총량 상승 2,027조합의 21.31%이며 지역 수 비율이 아니다. 기존 4쌍은 구조 차이가 큰 개발 사례이지 성능표본이 아니다.

## 3. 데이터와 외부자료 범위

| dataset_or_variable | decision | allowed_use | reason_or_limit |
|---|---|---|---|
| BC AMT/CNT/연령/업종 | CORE | 전국 255개·6개월 핵심 결과 | CNT를 고객수로 해석 금지 |
| 주민등록 연령별 인구 | CONTROL | 소비고령화·인구 대비 맥락 | 주민 1인당 실제지출 아님 |
| CPI | CONTROL | 명목/실질 민감도 | 전국 지수·대부분 근사 매핑 |
| LOCALDATA 중국음식·제과점 | MODULE_ONLY | 인허가 기반 공급 모듈 | BC 가맹점과 모집단 불일치 |
| LOCALDATA 한식·일식·서양식·스넥 | MODULE_ONLY | 민감도/사례 | MEDIUM/LOW 근사 매핑 |
| 국세청 편의점·슈퍼마켓 가동사업자 | MODULE_ONLY | 정확 명칭의 공급 방향 | 물리 점포 아님; 폐업 세부값 대부분 마스킹 |
| 생활인구 1~3월 | MODULE_ONLY | 89개 인구감소지역의 체류수요 맥락 | 3개월·공통계절성 |
| 한국부동산원 임대료·공실 | CASE_ONLY | 시도/대표상권 맥락 | 시군구 귀속 금지 |
| SEMAS 현재 API | HOLD | 현재 공급수준 후보 | 키 없음·월별 과거이력 없음 |
| 한국관광 데이터랩 | HOLD | 방문수요 분리 후보 | 실제 원자료 없음 |
| 시설좌표 기반 KNN 공간지표 | DROP_FROM_ANALYSIS | 이번 핵심주제에 불필요 | 공식 인접관계가 아님 |
| 국세청 시군구 폐업건수 | DROP_FROM_ANALYSIS | 현재 폐업효과 검증 | 96.46% 마스킹 |

BC 242,574행, 255개 시군구쌍, 11개 업종, 6개월과 완전 지역×업종 2,327개를 고정했다. 외국인 70,178행이 숫자 연령코드를 사용하므로 전체 숫자연령과 국내개인(GENDER 1·2) 집중도 민감도를 Evidence Card에 함께 넣었다. 어느 쪽도 주민 구성과 동일 집단으로 해석하지 않는다.

## 4. Evidence Card와 진단 절차

2,327개 카드에는 월별 AMT·CNT·건당 결제금액, 로그 slope와 항등식 오차, 강건성, 전국 동일업종 점유율, 지역총량과 업종 관계, 전체 숫자연령/국내개인 연령집중 민감도, 고정 7업종 집중도, 적용 가능한 공급모듈과 한계, 원자료 키를 저장했다.

규칙은 `F 피크민감도 → C 지역총량/업종 차이 → A CNT → B 건당결제액 → D 구성집중 → E 공급` 순이다. 각 규칙은 관찰, 필요한 점포정보, 질문, 답변분기, 조건부 후보, 확인지표, 유보조건, 금지단정을 갖는다. 원인이 확인되지 않으면 후보 상태는 `PENDING_REQUIRED_ANSWER`다.

## 5. 개발·평가 분리

- 개발: 기존 4쌍 8개 지역×업종.
- 평가: 개발지역을 제외하고 AMT 방향·거래규모·공급모듈 가용성으로 층화한 24개 실제 사례, 모두 서로 다른 시군구.
- 전체 데이터는 과거 EDA에 사용됐으므로 ‘규칙 개발과 분리한 내부 평가’이지 외부검증이 아니다.
- 평가사례를 본 뒤 규칙·±0.25% 임계값을 바꾸지 않았다. 0·0.5%는 민감도만 기록한다.

중립구간 민감도:

| neutral_band_pct_m | n | direction_mismatch_n | exact_opposite_n | neutral_involved_mismatch_n | region_up_industry_down_n | amt_up_cnt_nonup_n |
|---|---|---|---|---|---|---|
| 0.0 | 2327 | 592 | 592 | 0 | 505 | 9 |
| 0.25 | 2327 | 679 | 496 | 183 | 432 | 11 |
| 0.5 | 2327 | 767 | 406 | 361 | 357 | 16 |

±0.25%에서 679개였던 방향 불일치는 중립구간 정의에 따라 달라진다. 따라서 679를 자연적 경계나 위험률로 사용하지 않는다.

## 6. 방식별 실제 실행

| 방식 | 동일 카드 | 상태 | 이번에 확인한 것 |
|---|---:|---|---|
| A 일반 지표표+체크리스트 | 예 | EXECUTED | 합리적 공통 점검목록 제공 |
| B 일반 LLM | 예 | NOT_RUN_API | 프롬프트·어댑터 실행기만 작성 |
| C 진단절차+동일 LLM | 예 | NOT_RUN_API | 프롬프트·어댑터 실행기만 작성 |
| C0 규칙·템플릿 | 예 | EXECUTED | 사례별 우선규칙·질문·유보·분기 생성 |

자동검증:

| test | status | value | meaning |
|---|---|---|---|
| log_identity | PASS | 9.159339953157541e-16 | beta_AMT=beta_CNT+beta_ticket |
| card_count | PASS | 2327 | 완전 지역×업종 카드 수 |
| region_split_leakage | PASS | 0 | 개발·평가 시군구 중복 |
| evaluation_unique_regions | PASS | 24 | 평가사례별 고유 지역 |
| C0_repeat_determinism | PASS | e2264a3c21d59b4aa0c401e07200cd8ae5c22d0b11758609665abd294b7d65a8 | 동일 입력 반복 |
| source_card_binding | PASS | 1 | A/C0가 동일 카드 해시에 결합 |
| C0_numeric_fact_consistency | PASS | 1 | C0 수치가 Evidence Card와 정확히 일치 |
| direction_threshold_consistency | PASS | 1 | ±0.25%/월 방향 규칙 재계산 |
| question_budget | PASS | 3 | A/C0 질문 최대 3개 |
| missing_module_abstention | PASS | 0 | 공급자료 없을 때 공급 규칙 비활성 |
| secret_scan | PASS | 0 | 출력의 비밀키 패턴 |
| unsupported_claim_keyword_review | PASS | 0 | 자동 키워드 검사는 사람평가 대체 아님 |
| LLM_B_execution | NOT_RUN_API | no adapter/model/key/budget authorized | 프롬프트·실행기만 작성 |
| LLM_C_execution | NOT_RUN_API | no adapter/model/key/budget authorized | 프롬프트·실행기만 작성 |
| human_evaluation | AWAITING_HUMAN_EVAL | 0 | 점수·만족도·시간을 생성하지 않음 |

A와 C0 모두 같은 카드 해시에 묶였고 숫자를 새로 만들지 않았다. C0의 반복 출력은 동일했다. 이는 소프트웨어 정합성 증거이지 점검 적절성의 사람 검증이 아니다.

## 7. 차이가 난 사례·같은 사례·유보

개발 4쌍의 분리 판정:

| pair_id | first_check_differs | additional_questions_differ | full_order_differs | interpretation |
|---|---|---|---|---|
| 1 | True | True | True | 차이가 없어도 실패가 아님; 동일 점검이 합리적일 수 있음 |
| 2 | False | False | False | 차이가 없어도 실패가 아님; 동일 점검이 합리적일 수 있음 |
| 3 | True | True | True | 차이가 없어도 실패가 아님; 동일 점검이 합리적일 수 있음 |
| 4 | False | False | False | 차이가 없어도 실패가 아님; 동일 점검이 합리적일 수 있음 |

이번 개발 사례에서 Pair 2·4는 첫 점검뿐 아니라 추가 질문과 전체 순서도 같았다. Pair 1·3은 세 항목이 모두 달랐다. 같은 결과도 합리적일 수 있으므로 ‘4/4 성공’을 계산하지 않았다. 평가 24건에서 A의 첫 항목은 공통 검토이고 C0는 관측 신호별 우선규칙을 냈지만, 어느 순서가 더 적절한지는 사람 평가 전에는 미확인이다.

| case_id | A_first_priority | C0_first_priority | C0_trigger_count | C0_question_count | C0_pending_action_count | first_priority_differs |
|---|---|---|---|---|---|---|
| 경기도\|안산시 단원구\|8301 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경기도\|용인시 기흥구\|4020 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경상남도\|사천시\|4004 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 4 | 3 | 3 | True |
| 경상북도\|경주시\|4004 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 4 | 3 | 3 | True |
| 경상북도\|청송군\|8021 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 2 | 2 | 2 | True |
| 인천광역시\|계양구\|8005 | GENERAL_REVIEW | A_CNT | 3 | 3 | 3 | True |
| 전북특별자치도\|부안군\|8301 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 4 | 3 | 3 | True |
| 충청남도\|홍성군\|8004 | GENERAL_REVIEW | A_CNT | 3 | 3 | 3 | True |
| 부산광역시\|남구\|8021 | GENERAL_REVIEW | D_CONCENTRATION | 1 | 1 | 1 | True |
| 부산광역시\|사하구\|8002 | GENERAL_REVIEW | F_PEAK | 3 | 3 | 3 | True |
| 부산광역시\|서구\|8002 | GENERAL_REVIEW | B_TICKET | 2 | 2 | 2 | True |
| 서울특별시\|은평구\|8001 | GENERAL_REVIEW | F_PEAK | 2 | 2 | 2 | True |
| 울산광역시\|동구\|8005 | GENERAL_REVIEW | F_PEAK | 3 | 3 | 3 | True |
| 전라남도\|곡성군\|8301 | GENERAL_REVIEW | G_GENERAL | 0 | 1 | 0 | True |
| 충청북도\|단양군\|4010 | GENERAL_REVIEW | B_TICKET | 2 | 2 | 2 | True |
| 충청북도\|청주시 서원구\|4020 | GENERAL_REVIEW | B_TICKET | 2 | 2 | 2 | True |
| 강원특별자치도\|강릉시\|8006 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 강원특별자치도\|인제군\|8301 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경기도\|안양시 동안구\|8005 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경상남도\|양산시\|8021 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경상남도\|창녕군\|4004 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 경상북도\|포항시 북구\|8301 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 전라남도\|고흥군\|4004 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |
| 전라남도\|순천시\|4020 | GENERAL_REVIEW | C_TOTAL_INDUSTRY | 3 | 3 | 3 | True |

공급모듈이 없으면 E 규칙은 작동하지 않았고, 점포 답변이 없으면 개선 후보는 대기 상태로 남았다. 시뮬레이션 분기 5개는 코드 경로 시험용이며 실제 지역·점포 발견이 아니다.

## 8. 최소 시제품

`diagnosis_cli.py`는 지역·업종을 선택하면 분석기간과 Evidence Card를 확인하고 A 또는 C0 지역 진단을 출력한다. 점포답변 JSON을 선택적으로 받을 수 있지만 답변이 없으면 지역 진단 모드로 명시한다. 폐업확률·예상 매출증가율은 출력하지 않는다.

## 9. 추가 가치 판정

- **데이터 분해:** 지역총량과 업종 방향, CNT와 건당결제액 상쇄, 전체/국내개인 연령민감도를 한 카드에서 구분했다. 구현 완료.
- **진단 규칙:** 동일 정보에서 적용 gate·질문·유보·분기를 결정적으로 생성했다. 구현 및 자동 재현성 확인.
- **LLM:** B/C를 실행하지 않았으므로 추가 가치 미입증. C0만으로 충분한지 역시 사람평가 전에는 판단 보류.
- **사업효과:** 점주·POS·현장실험이 없어 미입증.

## 10. 다음 검증

1. 최소 2명의 전문가와 점주 평가자가 blind packet을 평가한다.
2. 동일 모델·출력예산·질문예산을 고정한 B/C 호출 권한을 명시적으로 제공한다.
3. 실제 동의받은 익명 점포의 영업일·주문·고유고객·상품/가격 자료를 입력한다.
4. 평가 완료 후 paired 사례 단위 집계를 실행한다. 미측정값을 0점으로 바꾸지 않는다.

## 11. 주요 산출물과 재현 명령

- [Evidence Cards](../evidence_cards.jsonl), [진단 규칙](../diagnosis_rules.csv)
- [내부 평가사례](../evaluation_cases.csv), [블라인드 평가표](../human_ratings_template.csv)
- [A/C0 구조 비교](../outputs/A_vs_C0_structural_comparison.csv), [사례별 기계 검증](../outputs/machine_fact_checks.csv)
- [중립구간 민감도](../outputs/neutral_band_sensitivity.csv), [패키지 검증](../outputs/package_validation_results.csv)
- [재현 README](../README.md), [산출물 manifest](../artifact_manifest.csv)
- [분석 생성 코드](../scripts/build_diagnosis_validation.py), [진단 엔진](../scripts/diagnosis_engine.py), [CLI 시제품](../scripts/diagnosis_cli.py)

```bash
python3 scripts/build_diagnosis_validation.py --out analysis/diagnosis_validation/run_20260907
python3 scripts/run_llm_comparison.py --run-dir analysis/diagnosis_validation/run_20260907 --status-only
python3 scripts/aggregate_human_evaluation.py --run-dir analysis/diagnosis_validation/run_20260907 --ratings analysis/diagnosis_validation/run_20260907/human_ratings_template.csv
python3 scripts/validate_diagnosis_package.py --run-dir analysis/diagnosis_validation/run_20260907
```

분석일: 2026-09-07  
상태: A/C0 및 자동검증 완료, B/C `NOT_RUN_API`, 사람평가 `AWAITING_HUMAN_EVAL`.
