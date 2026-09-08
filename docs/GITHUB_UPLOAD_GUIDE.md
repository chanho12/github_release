# GitHub 업로드 가이드

이 폴더는 독립 Git 저장소로 초기화돼 있지만 아직 커밋·원격 연결·업로드는 하지 않았다.

## 1. 공개 전 확인

```bash
git status --short
git status --ignored --short
```

반드시 확인할 사항:

- `dataset/ABP_CONTEST_DATA.csv`가 ignored 상태인지
- `dataset/localdata_raw/`와 `dataset/외부데이터/**/raw/`가 ignored 상태인지
- `.env`, API 키, 서비스 키가 추적 대상이 아닌지
- 외부 PDF와 우수사례 PDF가 추적 대상이 아닌지

현재 `.gitignore` 적용 후 커밋 후보에는 100MB를 넘는 파일이 없다.

## 2. 첫 커밋

사용자가 공개 범위를 최종 확인한 뒤 실행한다.

```bash
git add .
git status --short
git commit -m "Initialize BC consumption diagnosis project"
```

`git add -f`로 BC 원본이나 ignored 파일을 강제로 추가하지 않는다.

## 3. GitHub 원격 연결

GitHub에서 빈 저장소를 만든 뒤 다음을 실행한다.

```bash
git remote add origin <GITHUB_REPOSITORY_URL>
git push -u origin main
```

현재 단계에서는 원격 저장소 URL이 정해지지 않아 실행하지 않았다.

## 4. 데이터가 없는 clone에서 실행하기

공개 저장소 사용자는 다음 파일을 별도로 배치해야 한다.

```text
dataset/ABP_CONTEST_DATA.csv
```

대용량 외부자료는 `dataset/외부데이터/`의 README와 출처표를 따라 내려받고, 이후 구축 스크립트를 실행한다. 공개가 허용되는 소규모 가공 패널은 저장소에 포함할 수 있다.

## 5. 권장 저장소 설명

> BC카드 지역×업종 소비를 거래건수·건당 결제금액·소비층·업종구성·월별 안정성으로 분해하고, 확인된 신호에 따라 소상공인의 점검 순서와 조건부 개선 선택지를 제시하는 경영 의사결정 지원 프로젝트

