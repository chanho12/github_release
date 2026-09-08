# 같은 매출 변화, 다른 점검

BC카드의 2026년 1~6월 지역×업종 소비데이터를 분해해, 같은 매출 변화에서도 소상공인이 무엇을 먼저 확인해야 하는지 제시하는 경영 의사결정 지원 프로젝트다.

## 우리가 만드는 것

> **BC카드 소비구조 분해와 점주 확인정보를 결합한 지역·업종 경영 점검 우선순위 및 조건부 개선 선택지**

총매출만으로 성장·정체·감소를 판단하지 않고 다음 구조를 함께 본다.

- AMT 성장의 CNT·건당 결제금액 기여
- 지역 전체·개별업종·전국 동일업종 benchmark 차이
- 연령·성별·업종 소비구성 변화
- 특정 월 피크 의존성과 6개월 추세의 안정성
- 정확히 연결 가능한 업종의 주민인구·사업자·인허가 맥락

그 결과를 `확인된 사실 → 최대 3개 추가 질문 → 점주 답변에 따른 조건부 선택지 → 확인 KPI` 순서로 제공한다.

## 현재 결론

- 같은 AMT 변화라도 CNT와 건당 결제금액의 내부 경로가 다르다는 주장은 강하게 확인됐다.
- 지역 전체와 개별업종의 방향 차이도 반복적으로 존재했다.
- 엄격한 `AMT 증가·CNT 감소` 성장착시는 드물어 중심 주제로 사용하지 않는다.
- 폐업률과 순점포 변화를 예측하는 성능은 매우 낮아 폐업예측 기능을 제공하지 않는다.
- 현재 가능한 개인화는 점포정보 입력 전 `지역×업종 점검`이고, 점주 답변 후에만 개선 후보를 좁힐 수 있다.

상세 결과는 [전체 분석 종합보고서](reports/01_전체분석_종합보고서.md)를 참고한다.

## 저장소 구조

```text
.
├── README.md
├── requirements.txt
├── code/                    # 분석·검증·진단 Python 코드
├── notebooks/               # Jupyter 탐색 노트북
├── dataset/
│   ├── README.md
│   ├── bc_card/              # BC 데이터와 코드북
│   ├── 외부데이터/             # 외부자료·출처·품질·가공패널
│   ├── external_raw/         # 기존 코드 호환용 외부 원본
│   ├── localdata_raw/        # LOCALDATA 대용량 원본
│   ├── 폐업률/                 # 국세청 월간 원본
│   └── processed/            # 기존 가공 산출물
├── reports/                 # 핵심 보고서 5종
├── docs/                    # 프로젝트 범위와 감사 프롬프트
├── analysis/                # 코드를 실행하면 결과가 생성되는 위치
└── references/              # 방법 비교 참고자료
```

## 데이터 공개 원칙

데이터 파일은 로컬 패키지 안에 들어 있지만, `.gitignore`로 다음 원본의 GitHub 업로드를 막았다.

- BC카드 제공 원본 CSV
- LOCALDATA 대용량 전국 원본
- 국세청·기상청·법무부 XLS/XLSX 원본
- 외부기관 PDF·ZIP 및 API 키

BC 원본은 주최 측의 재배포 허용 여부를 확인한 뒤에만 공개해야 한다. 공개 저장소에는 우선 코드, 보고서, 스키마, 출처, 다운로드 방법, 품질표와 재현 가능한 소규모 가공표만 올리는 것을 권장한다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Python 3.10 이상을 권장한다.

## 빠른 실행

저장소 루트에서 실행한다.

```bash
# BC 원자료 기반 최종 주제 검증
python3 code/validate_final_topic_from_raw.py \
  --out analysis/final_topic_validation/run_local

# 전국 개·폐업 인허가 월 패널 재구성·검증
python3 code/build_national_closure_dataset.py
python3 code/validate_national_closure_outputs.py

# 초기 핵심 분석
python3 code/analyze_growth_illusion.py
python3 code/analyze_consumption_patterns.py
```

전체 분석 순서는 [code/README.md](code/README.md)에 정리했다.

## 해석 원칙

1. CNT는 고객 수나 방문자 수가 아니라 결제건수다.
2. AMT/CNT는 가격·마진·이익이 아니라 건당 평균 결제금액이다.
3. 지역×업종 결과를 개별 점포 결과로 동일시하지 않는다.
4. 행정 인허가와 가동사업자는 BC 가맹점 모집단과 다르다.
5. 6개월 자료로 전년동월 계절성을 통제했다고 주장하지 않는다.
6. 점주 답변과 실행 전후 KPI가 없으면 개선효과를 주장하지 않는다.

## 현재 검증 상태

| 단계 | 상태 |
|---|---|
| 원자료 품질감사·핵심 경향 재현 | 완료 |
| 외부자료 출처·기간·매핑 감사 | 완료 |
| Evidence Card·규칙 기반 진단 | 완료 |
| 일반 LLM과 동일조건 비교 | 미실시 |
| 전문가·점주 블라인드 평가 | 대기 |
| 실제 점포 적용·개선효과 측정 | 미실시 |

이 폴더는 GitHub 업로드 전용 사본이다. 독립 Git 저장소의 `main` 브랜치로 초기화했지만 커밋·원격 연결·업로드는 아직 수행하지 않았다. 다음 절차는 [GitHub 업로드 가이드](docs/GITHUB_UPLOAD_GUIDE.md)를 따른다. 원래 프로젝트 폴더와 분석 산출물은 수정하지 않는다.
