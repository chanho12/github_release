# 코드 실행 안내

모든 명령은 저장소 루트에서 실행한다. 각 스크립트는 `dataset/`을 읽고 `analysis/` 아래에 결과를 만든다.

## 핵심 분석

| 코드 | 역할 |
|---|---|
| `analyze_growth_illusion.py` | 초기 AMT 증가·CNT 감소 가설 검증 |
| `analyze_consumption_patterns.py` | AMT 분해·연령집중·업종구조·군집 |
| `analyze_third_exploration.py` | 추가 소비패턴 탐색 |
| `analyze_deep_exploration.py` | H1~H18 심층 분석·반증 |
| `analyze_fourth_external_integration.py` | 체류·공급·부동산 외부결합 |

## 데이터 구축

| 코드 | 역할 |
|---|---|
| `build_localdata_closure_panel.py` | 음식업 인허가 월 패널 |
| `build_nts_closure_panel.py` | 국세청 월간 지역지표 가공 |
| `prepare_external_data_package.py` | 외부자료 품질표·분석패널 생성 |
| `build_national_closure_dataset.py` | 전국 인허가 7종 품질감사·월 패널 |
| `parse_mois_living_population.py` | 생활인구 PDF 파싱 |
| `parse_reb_commercial_rent.py` | 임대료·공실 원본 파싱 |

## 주제·외부자료 검증

| 코드 | 역할 |
|---|---|
| `audit_same_sales_different_actions.py` | 보유자료와 기존 주장 재현 감사 |
| `validate_final_topic_from_raw.py` | BC 원자료 기반 최종 주제 검증 |
| `validate_external_explanatory_modules.py` | 외부자료의 추가 설명가치 검증 |
| `validate_weather_spike_module.py` | 월별 피크와 기상 연관 검증 |
| `validate_national_closure_outputs.py` | 폐업 패널 기간·키·산식 검증 |

## 진단 시스템

| 코드 | 역할 |
|---|---|
| `diagnosis_engine.py` | Evidence Card 규칙 엔진 |
| `diagnosis_cli.py` | 지역·업종 선택 CLI |
| `build_diagnosis_validation.py` | 개발·평가사례와 블라인드 패킷 생성 |
| `validate_diagnosis_package.py` | 진단 패키지 자동검증 |
| `run_llm_comparison.py` | 동일조건 LLM 비교 실행기 |
| `aggregate_human_evaluation.py` | 사람 평가 결과 집계 |

## 권장 전체 실행 순서

```bash
python3 code/analyze_growth_illusion.py
python3 code/analyze_consumption_patterns.py
python3 code/analyze_third_exploration.py
python3 code/analyze_deep_exploration.py
python3 code/analyze_fourth_external_integration.py

python3 code/validate_final_topic_from_raw.py \
  --out analysis/final_topic_validation/run_local

python3 code/build_national_closure_dataset.py
python3 code/validate_national_closure_outputs.py
```

진단 패키지는 앞 단계 감사 산출물을 필요로 한다. 기존 실행 결과를 포함하지 않는 새 clone에서는 감사 파이프라인을 먼저 실행해야 한다.

## 실행 상태 용어

- `EXECUTED`: 실제 실행 완료
- `NOT_RUN_API`: API 키 또는 모델 설정이 없어 미실행
- `AWAITING_HUMAN_EVAL`: 평가자료는 준비됐으나 사람 평가 미실시
- `NOT_MEASURED`: 점포정보나 실행 전후 성과가 없어 미측정

