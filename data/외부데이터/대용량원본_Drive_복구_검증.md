# 대용량 LOCALDATA 원본: Drive 복구 및 검증

검증일: 2026-09-17. GitHub 저장소의 `data/`에는 BC 원본, 분석 캐시, 외부 원본 대부분이 들어 있다. GitHub 일반 파일 한도를 넘는 아래 세 CSV만 [전용 Google Drive 폴더](https://drive.google.com/drive/folders/1KDC7fZF9aw4fm_3iRnFt49P-mYfeWYY3)에 보관한다. 노트북은 세 원본 대신 GitHub에 포함된 가공 패널을 읽으므로, 발표 분석 실행에는 Drive 다운로드가 필수가 아니다. 원시자료부터 다시 감사하거나 가공 패널을 재생성할 때 복구한다.

Drive 파일은 소유자 계정으로 업로드했다. **GitHub collaborator 권한만으로 Drive 접근 권한이 생기지는 않는다.** 링크가 열리지 않는 팀원은 Drive 파일에 읽기 권한을 별도로 받아야 한다.

## 1. 원본·공식 출처·고정 스냅샷

| 원본의 저장소 내 복구 경로 (`data/외부데이터/폐업_전국/raw/localdata/` 아래) | 원본 크기 | 원본 SHA-256 | 공식 재취득처 | 당시 확보 정보 |
|---|---:|---|---|---|
| `general_restaurants/일반음식점.csv` | 696,401,999 B | `90247868f8a58c931e1ebfc8d89ceb74df74c1ccd4a88589c1a08c6038da3103` | [공공데이터포털 일반음식점](https://www.data.go.kr/data/15045016/fileData.do) | 2026-09-04 기존 원본 재사용; CP949 |
| `rest_cafes/휴게음식점.csv` | 207,233,039 B | `98036f144ae9ad0f54ce6ed397908f8f80cbdea5db2ce5d7a152db216bf2624e` | [공공데이터포털 휴게음식점](https://www.data.go.kr/data/15006730/fileData.do) | 2026-09-04 기존 원본 재사용; CP949 |
| `other_candidates/담배소매업.csv` | 182,868,555 B | `03651a46425722a211647cb0bf73d73fdd444bf1eca46334755a27b6e13a2f9e` | [LOCALDATA 담배소매업 파일서비스](https://file.localdata.go.kr/file/tobacco_retailers/info) | 2026-09-08 취득; CP949 |

공식 페이지의 최신 다운로드는 매일 갱신될 수 있으므로 위 SHA-256과 일치한다는 보장은 없다. **분석 당시 바이트 단위의 동일 원본 복구는 Drive 고정 스냅샷을 이용한다.** 원본별 수집일, 행수, 인코딩, 공식 다운로드 경로는 [`폐업_전국/metadata/source_metadata.csv`](폐업_전국/metadata/source_metadata.csv)에 기록되어 있다.

## 2. Drive 파일과 무결성

| 내려받을 파일 | Drive 링크 | 크기 | SHA-256 |
|---|---|---:|---|
| `general_restaurants.zip.part-aa` | [파트 aa](https://drive.google.com/file/d/17FksL9OnE4F3ays4I1EFT0fyomEvEbeb/view?usp=drivesdk) | 94,371,840 B | `d0c99457e0d59f4c6d760f01355cff860d135bd1877c59ec4d5c66a9c9418e0e` |
| `general_restaurants.zip.part-ab` | [파트 ab](https://drive.google.com/file/d/1vDmUCcniIUDrjHZSuXBEJOOeakNME1MG/view?usp=drivesdk) | 92,446,836 B | `777db28c37c6c1c74592fbf715c195b69d95cf894a75119d2e54a40c2098b619` |
| `rest_cafes.zip` | [휴게음식점 ZIP](https://drive.google.com/file/d/1pWMUu2pxmcTZsPbBnsT0msmXhjqCSrrw/view?usp=drivesdk) | 51,738,511 B | `b08fff83671c907d1121a0a216abcea4f3d381127a05fc15a86e747cf491ff77` |
| `tobacco_retail.zip` | [담배소매업 ZIP](https://drive.google.com/file/d/157gQWapzPRSj_YAy9o6kDcJPs7q9d1We/view?usp=drivesdk) | 44,083,956 B | `79711c8498dab17eb1441c6457ec1cde6f0bb1072fbc909ff5a49e5e5cdeca3d` |

`general_restaurants.zip`의 두 파트를 순서대로 합쳤을 때 SHA-256은 `e406c7444179d177a4d23649d0f81a0a3b3b059c13639174bb1692eba588fb30`이다. 압축파일 내부 이름은 OS별 한글 파일명 깨짐을 피하기 위해 ASCII로 만들었다.

## 3. 복구 순서

1. 위 네 파일을 한 폴더에 다운로드한다. 두 `part-` 파일은 각각 다른 파일이므로 **둘 다** 받아야 한다.
2. macOS/Linux에서는 그 폴더에서 다음 명령으로 두 파트를 합친다.

   ```bash
   cat general_restaurants.zip.part-aa general_restaurants.zip.part-ab > general_restaurants.zip
   shasum -a 256 general_restaurants.zip rest_cafes.zip tobacco_retail.zip
   unzip general_restaurants.zip
   unzip rest_cafes.zip
   unzip tobacco_retail.zip
   ```

   Windows에서는 명령 프롬프트에서 `copy /b general_restaurants.zip.part-aa+general_restaurants.zip.part-ab general_restaurants.zip`을 실행한다. PowerShell의 `Get-FileHash general_restaurants.zip -Algorithm SHA256`으로 합친 파일을 검증한 뒤, 탐색기 또는 `Expand-Archive`로 각 ZIP을 푼다.
3. 압축 해제 후 이름을 다음처럼 바꾸고 저장소 루트 기준 경로로 옮긴다.

   | ZIP 내부 파일명 | 최종 상대경로 |
   |---|---|
   | `general_restaurants.csv` | `data/외부데이터/폐업_전국/raw/localdata/general_restaurants/일반음식점.csv` |
   | `rest_cafes.csv` | `data/외부데이터/폐업_전국/raw/localdata/rest_cafes/휴게음식점.csv` |
   | `tobacco_retail.csv` | `data/외부데이터/폐업_전국/raw/localdata/other_candidates/담배소매업.csv` |

4. 최종 CSV의 SHA-256을 1절과 비교한다. macOS/Linux: `shasum -a 256 <파일경로>`; Windows PowerShell: `Get-FileHash <파일경로> -Algorithm SHA256`. 해시가 다르면 그 파일을 분석에 사용하지 않는다.

## 4. 실제 확인 범위

- 세 원본과 ASCII 이름으로 만든 ZIP 내부 파일의 SHA-256이 일치했다.
- 세 ZIP 각각 압축 무결성 테스트(`unzip -t`)가 통과했다.
- 일반음식점 ZIP의 두 파트를 다시 이어 붙인 바이트가 원래 ZIP과 동일함을 `cmp`로 확인했다.
- Drive 업로드 후 파일 4개의 ID·이름·크기·부모 폴더를 메타데이터로 재조회했다.
- **팀원 계정으로의 다운로드 성공 여부는 아직 확인하지 않았다.** Drive 공유 권한은 별도 확인이 필요하다.

공식 페이지의 존재와 자료 설명은 확인했지만, 공식 사이트에서 2026-09 고정 스냅샷을 다시 내려받아 동일 해시로 검증한 것은 아니다. 고정 스냅샷의 동일성은 Drive 복구와 SHA-256 검증으로 확인한다.
