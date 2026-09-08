"""LOCALDATA 식품 인허 전국 파일에서 2026년 1~6월 개·폐업을 집계한다.

준비할 무료 파일(압축을 풀어도 됨):
    dataset/localdata_raw/일반음식점*.csv
    dataset/localdata_raw/휴게음식점*.csv
    dataset/localdata_raw/제과점*.csv

실행:
    python3 scripts/build_localdata_closure_panel.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "localdata_raw"
OUTPUT_DIR = ROOT / "dataset" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "LOCALDATA_식품업종_개폐업_202601_202606.xlsx"
BC_PATH = ROOT / "dataset" / "ABP_CONTEST_DATA.csv"
MONTHS = list(range(202601, 202607))

SIDO_ALIASES = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}


def find_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    normalized = {re.sub(r"\s+", "", str(column)): column for column in columns}
    for candidate in candidates:
        key = re.sub(r"\s+", "", candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"필요 컬럼을 찾지 못했습니다: {candidates}")
    return None


def csv_encoding(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            pd.read_csv(path, encoding=encoding, nrows=0)
            return encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"인코딩을 판별하지 못했습니다: {path}") from last_error


def read_csv_chunks(path: Path, chunk_size: int = 200_000):
    encoding = csv_encoding(path)
    yield from pd.read_csv(path, encoding=encoding, low_memory=False, chunksize=chunk_size)


def parse_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_datetime(compact.where(compact.str.len().eq(8)), format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(text, errors="coerce"))


def classify(service: str, subtype: pd.Series) -> pd.Series:
    text = subtype.fillna("").astype(str).str.strip()
    result = pd.Series(pd.NA, index=text.index, dtype="string")
    if "제과" in service:
        return result.fillna("제과점")

    result.loc[text.str.contains("한식", na=False)] = "한식통합"
    result.loc[text.str.contains("식육|숯불|갈비", na=False)] = "한식통합"
    result.loc[text.str.contains("일식|회집", na=False)] = "일식회집"
    result.loc[text.str.contains("중국식|중식", na=False)] = "중국음식"
    result.loc[text.str.contains("경양식|서양식", na=False)] = "서양음식"
    result.loc[text.str.contains("분식", na=False)] = "스넥"
    return result


def region_lookup() -> tuple[list[str], dict[str, tuple[str, str]]]:
    bc = pd.read_csv(BC_PATH, encoding="utf-8-sig", usecols=["SIDO_NM", "CCG_NM"])
    pairs = bc.drop_duplicates().itertuples(index=False, name=None)
    lookup: dict[str, tuple[str, str]] = {}
    for sido, ccg in pairs:
        lookup[f"{sido} {ccg}"] = (sido, ccg)
        if sido == "세종특별자치시":
            lookup["세종특별자치시"] = (sido, ccg)
        short = next((key for key, value in SIDO_ALIASES.items() if value == sido), None)
        if short:
            lookup[f"{short} {ccg}"] = (sido, ccg)
    return sorted(lookup, key=len, reverse=True), lookup


def extract_region(addresses: pd.Series, prefixes: list[str], lookup: dict[str, tuple[str, str]]) -> pd.DataFrame:
    old_dong = re.compile(r"(만석동|화수동|송현동|창영동|금곡동|송림동)")

    def one(value: object) -> tuple[object, object]:
        if pd.isna(value):
            return pd.NA, pd.NA
        address = re.sub(r"\s+", " ", str(value)).strip()
        # 2026년 하반기 최신 스냅샷의 개편 명칭을 BC 분석기간(1~6월) 경계로 역매핑한다.
        if address.startswith("전남광주통합특별시 "):
            ccg = address.split(" ", 2)[1]
            sido = "광주광역시" if ccg in {"동구", "서구", "남구", "북구", "광산구"} else "전라남도"
            return sido, ccg
        if address.startswith("인천광역시 영종구 "):
            return "인천광역시", "중구"
        if address.startswith("인천광역시 제물포구 "):
            if old_dong.search(address):
                return "인천광역시", "동구"
            return "인천광역시", "중구"
        if address.startswith("인천광역시 서해구 ") or address.startswith("인천광역시 검단구 "):
            return "인천광역시", "서구"
        for prefix in prefixes:
            if address.startswith(prefix + " ") or address == prefix:
                return lookup[prefix]
        return pd.NA, pd.NA

    values = addresses.map(one)
    return pd.DataFrame(values.tolist(), index=addresses.index, columns=["시도", "시군구"])


def normalize_file(path: Path, prefixes: list[str], lookup: dict[str, tuple[str, str]]):
    service = "제과점" if "제과" in path.name else ("휴게음식점" if "휴게" in path.name else "일반음식점")
    for data in read_csv_chunks(path):
        columns = list(data.columns)
        open_col = find_column(columns, ["인허가일자", "인허일자", "영업신고일자"])
        close_col = find_column(columns, ["폐업일자"], required=False)
        subtype_col = find_column(columns, ["업태구분명", "업종명"], required=False)
        road_col = find_column(columns, ["도로명전체주소", "도로명주소"], required=False)
        lot_col = find_column(columns, ["소재지전체주소", "지번주소"], required=False)
        id_col = find_column(columns, ["관리번호", "인허번호"], required=False)
        x_col = find_column(columns, ["좌표정보(X)", "좌표정보X", "X좌표"], required=False)
        y_col = find_column(columns, ["좌표정보(Y)", "좌표정보Y", "Y좌표"], required=False)

        address = data[road_col] if road_col else pd.Series(pd.NA, index=data.index)
        if lot_col:
            address = address.fillna(data[lot_col])
        subtype = data[subtype_col] if subtype_col else pd.Series("", index=data.index)

        out = pd.DataFrame(
            {
                "시설ID": data[id_col].astype("string") if id_col else path.stem + "_" + data.index.astype(str),
                "서비스": service,
                "업태구분명": subtype.astype("string"),
                "연결업종": classify(service, subtype),
                "인허가일자": parse_date(data[open_col]),
                "폐업일자": parse_date(data[close_col]) if close_col else pd.NaT,
                "주소": address.astype("string"),
                "좌표X": pd.to_numeric(data[x_col], errors="coerce") if x_col else pd.NA,
                "좌표Y": pd.to_numeric(data[y_col], errors="coerce") if y_col else pd.NA,
            }
        )
        region = extract_region(out["주소"], prefixes, lookup)
        yield pd.concat([out, region], axis=1)


def monthly_panel(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[data["연결업종"].notna() & data["시도"].notna()].copy()
    panels: list[pd.DataFrame] = []
    for yyyymm in MONTHS:
        start = pd.Timestamp(str(yyyymm) + "01")
        end = start + pd.offsets.MonthEnd(0)
        previous_end = start - pd.Timedelta(days=1)

        base = data[["시도", "시군구", "연결업종"]].drop_duplicates()
        active_previous = data["인허가일자"].le(previous_end) & (data["폐업일자"].isna() | data["폐업일자"].gt(previous_end))
        active_current = data["인허가일자"].le(end) & (data["폐업일자"].isna() | data["폐업일자"].gt(end))
        opened = data["인허가일자"].between(start, end)
        closed = data["폐업일자"].between(start, end)

        keys = ["시도", "시군구", "연결업종"]
        for name, mask in [
            ("가동_전월말", active_previous),
            ("가동_당월말", active_current),
            ("신규_당월", opened),
            ("폐업_당월", closed),
        ]:
            counts = data.loc[mask].groupby(keys).size().rename(name).reset_index()
            base = base.merge(counts, on=keys, how="left")
        base.insert(0, "기준년월", yyyymm)
        panels.append(base)

    panel = pd.concat(panels, ignore_index=True).fillna(
        {"가동_전월말": 0, "가동_당월말": 0, "신규_당월": 0, "폐업_당월": 0}
    )
    panel["폐업강도_pct"] = panel["폐업_당월"].div(panel["가동_전월말"].replace(0, pd.NA)).mul(100)
    return panel.sort_values(["기준년월", "시도", "시군구", "연결업종"])


def main() -> None:
    files = [
        path
        for path in sorted(RAW_DIR.glob("*.csv"))
        if any(keyword in path.name for keyword in ("일반음식점", "휴게음식점", "제과점"))
    ]
    expected = {"일반": any("일반" in p.name for p in files), "휴게": any("휴게" in p.name for p in files), "제과": any("제과" in p.name for p in files)}
    missing = [name for name, exists in expected.items() if not exists]
    if missing:
        raise FileNotFoundError(
            f"{RAW_DIR}에 다음 CSV가 필요합니다: {', '.join(missing)}. "
            "dataset/localdata_download.md의 무료 다운로드 링크를 확인하세요."
        )

    prefixes, lookup = region_lookup()
    panel_parts: list[pd.DataFrame] = []
    raw_rows = 0
    region_linked = 0
    industry_linked = 0
    for path in files:
        for normalized in normalize_file(path, prefixes, lookup):
            raw_rows += len(normalized)
            region_linked += int(normalized["시도"].notna().sum())
            industry_linked += int(normalized["연결업종"].notna().sum())
            panel_parts.append(monthly_panel(normalized))

    sum_columns = ["가동_전월말", "가동_당월말", "신규_당월", "폐업_당월"]
    group_columns = ["기준년월", "시도", "시군구", "연결업종"]
    panel = pd.concat(panel_parts, ignore_index=True).groupby(group_columns, as_index=False)[sum_columns].sum()
    panel["폐업강도_pct"] = panel["폐업_당월"].div(panel["가동_전월말"].replace(0, pd.NA)).mul(100)
    panel = panel.sort_values(group_columns)
    quality = pd.DataFrame(
        {
            "항목": ["원본 행수", "BC 지역 연결률(%)", "BC 업종 연결률(%)", "월별 패널 행수"],
            "값": [raw_rows, region_linked / raw_rows * 100, industry_linked / raw_rows * 100, len(panel)],
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="시군구_월별_개폐업", index=False)
        quality.to_excel(writer, sheet_name="품질_요약", index=False)
    print(f"생성 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
