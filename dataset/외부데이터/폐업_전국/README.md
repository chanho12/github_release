# 전국 개·폐업 외부데이터 패키지

2026년 1~6월 BC카드 지역×업종 분석에 붙일 수 있는 전국 인허가·영업상태 자료를 수집하고, 실제 매핑 가능 범위를 감사한 패키지다.

## 결론

- 전국 월별 개업·폐업·월초/월말 영업 수는 LOCALDATA 최신 스냅샷의 인허가일·폐업일로 재구성했다.
- BC 11개 업종 중 `HIGH` 매핑은 대형할인점·중국음식·제과점 3개다.
- 2026년 상반기 폐업 비교의 핵심 사용은 실제 사건과 전국 커버리지가 충분한 중국음식·제과점 2개가 가장 안전하다.
- `MEDIUM`까지 넓히면 9개 업종을 연결할 수 있지만, 편의점처럼 인허가 하위집단이거나 일반한식·스넥처럼 분류 범위가 다른 자료다.
- 이 패널의 수치는 BC 가맹점 폐업률이나 특정 점포의 폐업위험이 아니라 행정 인허가 모집단의 개·폐업 맥락이다.

상세 판단은 [national_closure_data_audit.md](reports/national_closure_data_audit.md)를 본다.

## 디렉터리

- `raw/localdata/`: 전국 LOCALDATA 원본 CSV
- `raw/nts/`: 국세청 TASIS 2026년 1~6월 원본 XLSX
- `raw/mfds/`, `raw/seoul_validation/`: API 메타데이터와 미실행 사유
- `processed/`: BC 업종 연결표와 월별 재구성 패널
- `audit/`: 품질·날짜·중복·지역·매핑·교차검증 결과
- `metadata/`: 출처, 체크섬, 실행 요약, 전체 파일 manifest

기존 프로젝트에 있던 대용량 공식 원본은 저장공간 중복을 피하기 위해 hard link로 재사용했다. 원본 내용과 SHA-256은 `metadata/source_metadata.csv`에서 확인할 수 있다.

## 핵심 산출물

- `processed/202601_202606_national_closure_panel_bc_mapping.csv`: HIGH·MEDIUM만 포함한 소스별 월 패널
- `processed/source_specific_all_mapping_confidences.csv`: LOW까지 포함한 비교용 소스별 패널
- `processed/bc_to_closure_industry_mapping.csv`: 매핑 근거와 오분류 가능성
- `audit/raw_dataset_quality.csv`: 원본별 행·컬럼·날짜·지역 품질
- `audit/validation_checks.csv`: 산식·키·기간 자동검증
- `metadata/output_manifest.csv`: 패키지 파일의 크기와 SHA-256

편의점의 두 LOCALDATA proxy는 서로 다른 인허가 하위집단이므로 합산하면 안 된다. 패널은 반드시 `source`까지 키로 사용한다.

## 재현

이미 저장된 원본으로 패널을 다시 만든다.

```bash
PYTHONPYCACHEPREFIX=/tmp/bc_national_closure_pycache \
python3 scripts/build_national_closure_dataset.py

PYTHONPYCACHEPREFIX=/tmp/bc_national_closure_validate_pycache \
python3 scripts/validate_national_closure_outputs.py
```

LOCALDATA 원본을 새로 내려받으려면 아래를 실행한다. 기존 원본을 갱신하므로 실행 시점의 스냅샷으로 결과가 달라질 수 있다.

```bash
bash scripts/download_national_closure_sources.sh
```

식약처·서울시·국세청 사업자상태 API는 키 또는 사업자등록번호가 필요하다. 실제 키는 저장하지 말고 `.env.example`을 복사해 로컬에서만 관리한다.

## 날짜 기준

- 대상 분석기간: 2026-01-01~2026-06-30
- 원본 확보·감사일: 2026-09-08
- 월별 영업 수: 해당 월 말일 기준
- 개업 수: 인허가일이 해당 월에 속하는 행 수
- 폐업 수: 폐업일이 해당 월에 속하는 행 수
- 폐업률 시작분모: `월 폐업 수 / 월초 영업 수`
- 폐업률 평균분모: `월 폐업 수 / ((월초 영업 수 + 월말 영업 수)/2)`

두 폐업률을 모두 남겼으며, 분모가 0이면 결측으로 둔다.
