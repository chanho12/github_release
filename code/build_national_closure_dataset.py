"""Build and audit a nationwide 2026-H1 opening/closure panel for BC industries.

This script does not model closure risk.  It audits official administrative
records, reconstructs monthly counts, and preserves source-specific population
and mapping limitations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "dataset" / "외부데이터" / "폐업_전국"
RAW = BASE / "raw"
PROCESSED = BASE / "processed"
AUDIT = BASE / "audit"
META = BASE / "metadata"
BC_PATH = ROOT / "dataset" / "ABP_CONTEST_DATA.csv"
MONTHS = list(range(202601, 202607))
RETRIEVAL_DATE = pd.Timestamp("2026-09-08")

SOURCES = [
    {
        "source_key": "local_general",
        "source_name": "LOCALDATA 일반음식점",
        "path": RAW / "localdata/general_restaurants/일반음식점.csv",
        "provider": "행정안전부",
        "official_page": "https://www.data.go.kr/data/15045016/fileData.do",
        "public_data_id": "15045016",
        "download_url": "https://file.localdata.go.kr/file/download/general_restaurants/info",
        "update_cycle": "매일(2일 전 기준 현행화)",
    },
    {
        "source_key": "local_rest_cafe",
        "source_name": "LOCALDATA 휴게음식점",
        "path": RAW / "localdata/rest_cafes/휴게음식점.csv",
        "provider": "행정안전부",
        "official_page": "https://www.data.go.kr/data/15006730/fileData.do",
        "public_data_id": "15006730",
        "download_url": "https://file.localdata.go.kr/file/download/rest_cafes/info",
        "update_cycle": "매일(2일 전 기준 현행화)",
    },
    {
        "source_key": "local_bakery",
        "source_name": "LOCALDATA 제과점영업",
        "path": RAW / "localdata/bakery/제과점.csv",
        "provider": "행정안전부",
        "official_page": "https://www.data.go.kr/data/15006688/fileData.do",
        "public_data_id": "15006688",
        "download_url": "https://file.localdata.go.kr/file/download/bakeries/info",
        "update_cycle": "매일(2일 전 기준 현행화)",
    },
    {
        "source_key": "local_large_retail",
        "source_name": "LOCALDATA 대규모·준대규모점포",
        "path": RAW / "localdata/large_scale_retail/생활_대규모점포.csv",
        "provider": "행정안전부",
        "official_page": "https://www.data.go.kr/data/15114138/standard.do",
        "public_data_id": "15114138 (현재 파일 카탈로그 15045013)",
        "download_url": "https://file.localdata.go.kr/file/download/large_scale_retail_stores/info",
        "update_cycle": "매일(2일 전 기준 현행화)",
    },
    {
        "source_key": "local_other_food_retail",
        "source_name": "LOCALDATA 식품판매업(기타)",
        "path": RAW / "localdata/food_retail/식품판매업_기타.csv",
        "provider": "행정안전부",
        "official_page": "https://www.data.go.kr/data/15044979/fileData.do",
        "public_data_id": "15044979",
        "download_url": "https://file.localdata.go.kr/file/download/other_food_retailers/info",
        "update_cycle": "매일(2일 전 기준 현행화)",
    },
    {
        "source_key": "local_otc",
        "source_name": "LOCALDATA 안전상비의약품판매업소",
        "path": RAW / "localdata/other_candidates/안전상비의약품판매업소.csv",
        "provider": "행정안전부",
        "official_page": "https://file.localdata.go.kr/file/over_the_counter_medicine_stores/info",
        "public_data_id": "LOCALDATA file service",
        "download_url": "https://file.localdata.go.kr/file/download/over_the_counter_medicine_stores/info",
        "update_cycle": "매일",
    },
    {
        "source_key": "local_tobacco",
        "source_name": "LOCALDATA 담배소매업",
        "path": RAW / "localdata/other_candidates/담배소매업.csv",
        "provider": "행정안전부",
        "official_page": "https://file.localdata.go.kr/file/tobacco_retailers/info",
        "public_data_id": "LOCALDATA file service",
        "download_url": "https://file.localdata.go.kr/file/download/tobacco_retailers/info",
        "update_cycle": "매일",
    },
]

BC_CODE = {
    "대형할인점": 4004,
    "편의점": 4010,
    "슈퍼마켓": 4020,
    "일반한식": 8001,
    "갈비전문점": 8002,
    "한정식": 8003,
    "일식회집": 8004,
    "중국음식": 8005,
    "서양음식": 8006,
    "스넥": 8021,
    "제과점": 8301,
}

MAPPING_ROWS = [
    ["중국음식", "LOCALDATA 일반음식점", "일반음식점", "중국식", "업태구분명 정확 일치", "HIGH", "중식 음식점 범위가 비교적 직접 대응", "전국 인허가", "BC 가맹점이 아닌 행정 인허가", "일부 중식 복합업태", "주 분석"],
    ["제과점", "LOCALDATA 제과점영업", "제과점영업", "제과점영업", "서비스·업태 직접 일치", "HIGH", "독립 제과점영업 인허가 모집단", "전국 인허가", "휴게음식점 과자점 제외", "BC 가맹점 모집단 차이", "주 분석"],
    ["대형할인점", "LOCALDATA 대규모·준대규모점포", "대규모점포", "대형마트", "업태구분명=대형마트", "HIGH", "백화점·쇼핑센터 등을 제외한 보수적 연결", "전국 유통산업발전법 등록", "BC 내부 대형할인점 범위 미공개", "등록대상 미만 할인점", "주 분석"],
    ["일반한식", "LOCALDATA 일반음식점", "일반음식점", "한식", "업태구분명=한식", "MEDIUM", "방향은 직접적이나 BC 세부 분류와 범위가 다름", "전국 인허가", "한식 내 갈비·한정식 혼입 가능", "한식 복합업태", "제한 분석"],
    ["일식회집", "LOCALDATA 일반음식점", "일반음식점", "일식|횟집", "두 업태 합산", "MEDIUM", "BC 명칭의 일식·회집을 근사", "전국 인허가", "두 업태의 범위가 BC와 불일치 가능", "기타 업태로 신고된 일식", "제한 분석"],
    ["서양음식", "LOCALDATA 일반음식점", "일반음식점", "경양식|패밀리레스트랑", "보수적 업태 합산", "MEDIUM", "서양식 직접 코드는 없고 경양식 중심 근사", "전국 인허가", "경양식 외 업태 혼입", "기타·외국음식 신고 서양식", "제한 분석"],
    ["스넥", "LOCALDATA 일반음식점", "일반음식점", "분식|김밥(도시락)", "간식·분식 업태 합산", "MEDIUM", "BC 스넥의 공식 정의가 없어 분식 중심 근사", "전국 인허가", "식사형 분식 포함", "휴게음식점 일반조리판매 등 누락", "제한 분석"],
    ["편의점", "LOCALDATA 휴게음식점", "휴게음식점", "편의점", "업태구분명=편의점", "MEDIUM", "명칭 특이도는 높지만 휴게음식 인허가를 가진 편의점만 포함", "전국 인허가 하위집단", "편의점 외 업소 가능성", "조리·휴게 인허가 없는 편의점", "제한 분석"],
    ["편의점", "LOCALDATA 안전상비의약품판매업소", "안전상비의약품판매업소", "편의점 브랜드명", "전체를 편의점 proxy로 사용", "MEDIUM", "사업장명 다수가 편의점이나 의약품 판매등록 편의점만 포착", "전국 등록 하위집단", "비편의 판매소", "의약품 미판매 편의점", "민감도·검증"],
    ["슈퍼마켓", "LOCALDATA 대규모·준대규모점포", "준대규모점포", "점포구분명=준대규모점포", "준대규모점포만 연결", "MEDIUM", "기업형 슈퍼마켓에 가까운 부분집합", "전국 등록 하위집단", "슈퍼 외 준대규모 소매", "독립·소형 슈퍼", "제한 분석"],
    ["슈퍼마켓", "LOCALDATA 식품판매업(기타)", "식품판매업(기타)", "기타식품판매업", "서비스 전체를 supermarket proxy", "LOW", "일정 규모 이상 식품판매소이나 BC 슈퍼와 동일 모집단 아님", "전국 인허가", "대형마트·기타 식품소매 포함", "소형 슈퍼", "사례·민감도만"],
    ["갈비전문점", "LOCALDATA 일반음식점", "일반음식점", "식육(숯불구이)", "업태구분명 근사", "LOW", "갈비 외 육류구이의 과잉포함이 큼", "전국 인허가", "고깃집 전반", "한식으로 신고된 갈비점", "통합패널 제외"],
    ["한정식", "LOCALDATA 일반음식점", "일반음식점", "별도 업태 없음", "직접 식별 불가", "NOT_USABLE", "한식 안에서 한정식을 분리할 필드가 없음", "없음", "한식 전체를 잘못 포함", "한식 업태 내 한정식", "제외"],
    ["편의점", "LOCALDATA 담배소매업", "담배소매업", "업태 없음", "전체 proxy 검토", "NOT_USABLE", "편의점 외 전자담배·잡화·마트 등 과잉포함, 폐업일 누락 구조", "전국 지정", "광범위한 비편의점", "담배 미판매 편의점", "제외"],
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_encoding(path: Path) -> str:
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            pd.read_csv(path, encoding=enc, nrows=5)
            return enc
        except UnicodeDecodeError:
            continue
    raise UnicodeError(path)


def parse_date(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_datetime(compact.where(compact.str.len().eq(8)), format="%Y%m%d", errors="coerce")
    return parsed.fillna(pd.to_datetime(text, errors="coerce"))


def bc_regions() -> tuple[pd.DataFrame, dict[str, tuple[str, str]], re.Pattern[str]]:
    bc = pd.read_csv(BC_PATH, encoding="utf-8-sig")
    actual = list(bc.columns)
    required = ["STRD_YYMM", "SIDO_NM", "CCG_NM", "TP_BUZ_NO", "TP_BUZ_NM", "amt", "cnt"]
    missing = [c for c in required if c not in actual]
    if missing:
        raise ValueError(f"BC 필수 컬럼 누락: {missing}; 실제={actual}")
    regions = bc[["SIDO_NM", "CCG_NM"]].drop_duplicates().sort_values(["SIDO_NM", "CCG_NM"])
    regions["REGION_KEY"] = regions["SIDO_NM"] + "|" + regions["CCG_NM"]
    aliases = {
        "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시",
        "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
        "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
        "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도",
    }
    lookup: dict[str, tuple[str, str]] = {}
    reverse = {v: k for k, v in aliases.items()}
    for sido, ccg in regions[["SIDO_NM", "CCG_NM"]].itertuples(index=False):
        lookup[f"{sido} {ccg}"] = (sido, ccg)
        if sido in reverse:
            lookup[f"{reverse[sido]} {ccg}"] = (sido, ccg)
        if sido == "세종특별자치시":
            lookup["세종특별자치시"] = (sido, ccg)
            lookup["세종"] = (sido, ccg)
    pattern = re.compile(r"^(" + "|".join(re.escape(x) for x in sorted(lookup, key=len, reverse=True)) + r")(?:\s|$)")
    bc_summary = pd.DataFrame([
        {"item": "row_count", "value": len(bc)},
        {"item": "columns", "value": "|".join(actual)},
        {"item": "months", "value": "|".join(map(str, sorted(bc.STRD_YYMM.unique())))},
        {"item": "region_count", "value": len(regions)},
        {"item": "industry_count", "value": bc.TP_BUZ_NO.nunique()},
        {"item": "duplicate_grain", "value": int(bc.duplicated(["STRD_YYMM", "SIDO_NM", "CCG_NM", "GENDER_CD", "AGE_CD", "TP_BUZ_NO"]).sum())},
    ])
    bc_summary.to_csv(AUDIT / "bc_source_check.csv", index=False, encoding="utf-8-sig")
    return regions, lookup, pattern


def extract_region(address: pd.Series, lookup: dict[str, tuple[str, str]], pattern: re.Pattern[str]) -> pd.DataFrame:
    x = address.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Snapshot contains post-H1 administrative names; map them to the BC H1 boundary.
    x = x.str.replace(r"^인천광역시 영종구 ", "인천광역시 중구 ", regex=True)
    x = x.str.replace(r"^인천광역시 (서해구|검단구) ", "인천광역시 서구 ", regex=True)
    j = x.str.startswith("인천광역시 제물포구 ")
    old_dong = x.str.contains(r"만석동|화수동|송현동|창영동|금곡동|송림동", regex=True, na=False)
    x.loc[j & old_dong] = x.loc[j & old_dong].str.replace(r"^인천광역시 제물포구 ", "인천광역시 동구 ", regex=True)
    x.loc[j & ~old_dong] = x.loc[j & ~old_dong].str.replace(r"^인천광역시 제물포구 ", "인천광역시 중구 ", regex=True)
    m = x.str.extract(pattern, expand=False)
    pairs = m.map(lookup).map(lambda value: value if isinstance(value, tuple) else (pd.NA, pd.NA))
    out = pd.DataFrame(pairs.tolist(), index=x.index, columns=["SIDO_NM", "CCG_NM"])
    # The new Jeonnam-Gwangju name requires district-specific back-mapping.
    special = x.str.startswith("전남광주통합특별시 ")
    if special.any():
        ccg = x.loc[special].str.split().str[1]
        out.loc[special, "CCG_NM"] = ccg
        out.loc[special, "SIDO_NM"] = np.where(ccg.isin(["동구", "서구", "남구", "북구", "광산구"]), "광주광역시", "전라남도")
    out["REGION_KEY"] = out["SIDO_NM"].astype("string") + "|" + out["CCG_NM"].astype("string")
    return out


def mapped_industry(source_key: str, subtype: pd.Series, store_kind: pd.Series) -> tuple[pd.Series, pd.Series]:
    sub = subtype.fillna("").astype(str).str.strip()
    kind = store_kind.fillna("").astype(str).str.strip()
    name = pd.Series(pd.NA, index=sub.index, dtype="string")
    conf = pd.Series(pd.NA, index=sub.index, dtype="string")
    rules: list[tuple[pd.Series, str, str]] = []
    if source_key == "local_general":
        rules = [
            (sub.eq("중국식"), "중국음식", "HIGH"),
            (sub.eq("한식"), "일반한식", "MEDIUM"),
            (sub.isin(["일식", "횟집"]), "일식회집", "MEDIUM"),
            (sub.isin(["경양식", "패밀리레스트랑"]), "서양음식", "MEDIUM"),
            (sub.isin(["분식", "김밥(도시락)"]), "스넥", "MEDIUM"),
            (sub.eq("식육(숯불구이)"), "갈비전문점", "LOW"),
        ]
    elif source_key == "local_rest_cafe":
        rules = [(sub.eq("편의점"), "편의점", "MEDIUM")]
    elif source_key == "local_bakery":
        rules = [(sub.eq("제과점영업"), "제과점", "HIGH")]
    elif source_key == "local_large_retail":
        rules = [(sub.eq("대형마트") & kind.ne("준대규모점포"), "대형할인점", "HIGH"), (kind.eq("준대규모점포"), "슈퍼마켓", "MEDIUM")]
    elif source_key == "local_other_food_retail":
        rules = [(sub.eq("기타식품판매업"), "슈퍼마켓", "LOW")]
    elif source_key == "local_otc":
        rules = [(pd.Series(True, index=sub.index), "편의점", "MEDIUM")]
    for mask, industry, confidence in rules:
        available = mask & name.isna()
        name.loc[available] = industry
        conf.loc[available] = confidence
    return name, conf


def hashed_duplicate_summary(parts: list[np.ndarray]) -> tuple[int, int]:
    if not parts:
        return 0, 0
    values = np.concatenate(parts)
    if not len(values):
        return 0, 0
    _, counts = np.unique(values, return_counts=True)
    return int((counts > 1).sum()), int((counts - 1).clip(min=0).sum())


def audit_localdata(regions: pd.DataFrame, lookup: dict[str, tuple[str, str]], pattern: re.Pattern[str]):
    quality_rows, state_rows, anomaly_rows, duplicate_rows, subtype_rows, coverage_rows = [], [], [], [], [], []
    monthly_parts: list[pd.DataFrame] = []
    closure_parts: list[pd.DataFrame] = []
    raw_closure_parts: list[pd.DataFrame] = []
    missing_state_parts: list[pd.DataFrame] = []
    anomaly_sample_rows: list[dict[str, object]] = []
    proxy_rows = []

    for spec in SOURCES:
        path = spec["path"]
        enc = detect_encoding(path)
        header = list(pd.read_csv(path, encoding=enc, nrows=0).columns)
        needed = [c for c in ["관리번호", "인허가번호", "인허가일자", "폐업일자", "영업상태명", "상세영업상태명", "사업장명", "업태구분명", "점포구분명", "도로명주소", "지번주소", "좌표정보(X)", "좌표정보(Y)"] if c in header]
        row_count = 0
        open_present = close_present = strict_closed = strict_closed_missing = 0
        parse_fail_open = parse_fail_close = close_before_open = future_dates = abnormal_year = 0
        subtype_counter, state_counter = Counter(), Counter()
        linked_regions: set[str] = set()
        management_hashes: list[np.ndarray] = []
        permit_hashes: list[np.ndarray] = []
        name_address_hashes: list[np.ndarray] = []
        name_coord_hashes: list[np.ndarray] = []
        brand_counter = Counter()

        for d in pd.read_csv(path, encoding=enc, usecols=needed, chunksize=200_000, low_memory=False):
            row_count += len(d)
            for c in needed:
                if c not in d:
                    d[c] = pd.NA
            open_raw = d.get("인허가일자", pd.Series(pd.NA, index=d.index))
            close_raw = d.get("폐업일자", pd.Series(pd.NA, index=d.index))
            open_dt, close_dt = parse_date(open_raw), parse_date(close_raw)
            has_open_raw = open_raw.notna() & open_raw.astype("string").str.strip().ne("")
            has_close_raw = close_raw.notna() & close_raw.astype("string").str.strip().ne("")
            open_present += int(open_dt.notna().sum())
            close_present += int(close_dt.notna().sum())
            parse_fail_open += int((has_open_raw & open_dt.isna()).sum())
            parse_fail_close += int((has_close_raw & close_dt.isna()).sum())
            close_before_open += int((close_dt.notna() & open_dt.notna() & close_dt.lt(open_dt)).sum())
            future_dates += int((open_dt.gt(RETRIEVAL_DATE) | close_dt.gt(RETRIEVAL_DATE)).sum())
            abnormal_year += int(((open_dt.dt.year.lt(1900) | open_dt.dt.year.gt(2026)) & open_dt.notna()).sum())

            state = d.get("영업상태명", pd.Series("", index=d.index)).fillna("").astype(str)
            detail = d.get("상세영업상태명", pd.Series("", index=d.index)).fillna("").astype(str)
            strict = state.str.contains("폐업", na=False) | detail.str.contains("폐업", na=False)
            strict_closed += int(strict.sum())
            strict_closed_missing += int((strict & close_dt.isna()).sum())
            state_counter.update(state.where(state.ne(""), "<NA>"))

            subtype = d.get("업태구분명", pd.Series(pd.NA, index=d.index))
            store_kind = d.get("점포구분명", pd.Series(pd.NA, index=d.index))
            subtype_counter.update(subtype.fillna("<NA>").astype(str))
            address = d.get("도로명주소", pd.Series(pd.NA, index=d.index))
            if "지번주소" in d:
                address = address.fillna(d["지번주소"])
            geo = extract_region(address, lookup, pattern)
            linked_regions.update(geo["REGION_KEY"].dropna().astype(str).unique())
            industry, confidence = mapped_industry(spec["source_key"], subtype, store_kind)

            raw_closed = close_dt.between("2026-01-01", "2026-06-30")
            if raw_closed.any():
                rc = pd.DataFrame({
                    "source": spec["source_name"], "SIDO_NM": geo["SIDO_NM"], "CCG_NM": geo["CCG_NM"],
                    "REGION_KEY": geo["REGION_KEY"], "external_subtype": subtype.fillna("<NA>").astype(str),
                    "close_date": close_dt,
                }).loc[raw_closed]
                rc["STRD_YYMM"] = rc.close_date.dt.strftime("%Y%m").astype(int)
                raw_closure_parts.append(rc)
            miss = pd.DataFrame({
                "source": spec["source_name"], "SIDO_NM": geo["SIDO_NM"],
                "external_subtype": subtype.fillna("<NA>").astype(str),
                "closed_state": strict.astype(int), "closed_state_missing_date": (strict & close_dt.isna()).astype(int),
            })
            missing_state_parts.append(miss.groupby(["source", "SIDO_NM", "external_subtype"], dropna=False, as_index=False)[["closed_state", "closed_state_missing_date"]].sum())
            anomaly_mask = (has_open_raw & open_dt.isna()) | (has_close_raw & close_dt.isna()) | (close_dt.notna() & open_dt.notna() & close_dt.lt(open_dt)) | open_dt.gt(RETRIEVAL_DATE) | close_dt.gt(RETRIEVAL_DATE)
            if anomaly_mask.any() and sum(x["source"] == spec["source_name"] for x in anomaly_sample_rows) < 100:
                sample = pd.DataFrame({
                    "source": spec["source_name"], "management_id": d.get("관리번호", pd.Series(pd.NA, index=d.index)),
                    "business_name": d.get("사업장명", pd.Series(pd.NA, index=d.index)), "open_date_raw": open_raw,
                    "close_date_raw": close_raw, "state": state, "detail_state": detail,
                    "issue": np.select([
                        has_open_raw & open_dt.isna(), has_close_raw & close_dt.isna(),
                        close_dt.notna() & open_dt.notna() & close_dt.lt(open_dt),
                        open_dt.gt(RETRIEVAL_DATE) | close_dt.gt(RETRIEVAL_DATE),
                    ], ["OPEN_PARSE_FAIL", "CLOSE_PARSE_FAIL", "CLOSE_BEFORE_OPEN", "FUTURE_DATE"], default="OTHER"),
                }).loc[anomaly_mask].head(100)
                anomaly_sample_rows.extend(sample.to_dict("records"))

            if spec["source_key"] in {"local_otc", "local_tobacco", "local_other_food_retail"}:
                names = d.get("사업장명", pd.Series("", index=d.index)).fillna("").astype(str).str.upper()
                patterns = {
                    "CU/BGF": r"\bCU\b|BGF|비지에프|씨유",
                    "GS25": r"GS\s*25|지에스\s*25",
                    "세븐일레븐": r"세븐일레븐|7\s*ELEVEN",
                    "이마트24": r"이마트\s*24|EMART\s*24",
                    "미니스톱": r"미니스톱|MINISTOP",
                    "마트·슈퍼": r"마트|슈퍼|MARKET",
                }
                for label, rx in patterns.items():
                    brand_counter[label] += int(names.str.contains(rx, regex=True, na=False).sum())

            # Duplicate diagnostics use stable uint64 hashes, preserving rather than deduplicating rows.
            if "관리번호" in d:
                x = d["관리번호"].astype("string")
                management_hashes.append(pd.util.hash_pandas_object(x[x.notna()], index=False).to_numpy(dtype="uint64"))
            if "인허가번호" in d:
                x = d["인허가번호"].astype("string")
                permit_hashes.append(pd.util.hash_pandas_object(x[x.notna()], index=False).to_numpy(dtype="uint64"))
            names = d.get("사업장명", pd.Series(pd.NA, index=d.index)).astype("string").fillna("")
            addr = address.astype("string").fillna("")
            valid = names.ne("") & addr.ne("")
            name_address_hashes.append(pd.util.hash_pandas_object(names[valid] + "|" + addr[valid], index=False).to_numpy(dtype="uint64"))
            if "좌표정보(X)" in d and "좌표정보(Y)" in d:
                coord = d["좌표정보(X)"].astype("string").fillna("") + "|" + d["좌표정보(Y)"].astype("string").fillna("")
                valid_coord = names.ne("") & ~coord.isin(["|", "nan|nan", "<NA>|<NA>"])
                name_coord_hashes.append(pd.util.hash_pandas_object(names[valid_coord] + "|" + coord[valid_coord], index=False).to_numpy(dtype="uint64"))

            base = pd.DataFrame({
                "SIDO_NM": geo["SIDO_NM"], "CCG_NM": geo["CCG_NM"], "REGION_KEY": geo["REGION_KEY"],
                "TPBUZ_NM": industry, "mapping_confidence": confidence,
                "open_date": open_dt, "close_date": close_dt,
            })
            base = base[base.REGION_KEY.notna() & base.TPBUZ_NM.notna()]
            if not base.empty:
                base["source"] = spec["source_name"]
                for yyyymm in MONTHS:
                    start = pd.Timestamp(str(yyyymm) + "01")
                    end = start + pd.offsets.MonthEnd(0)
                    active_start = base.open_date.lt(start) & (base.close_date.isna() | base.close_date.ge(start))
                    active_end = base.open_date.le(end) & (base.close_date.isna() | base.close_date.gt(end))
                    active_any = base.open_date.le(end) & (base.close_date.isna() | base.close_date.gt(start))
                    opened = base.open_date.between(start, end)
                    closed = base.close_date.between(start, end)
                    keys = ["SIDO_NM", "CCG_NM", "REGION_KEY", "TPBUZ_NM", "mapping_confidence", "source"]
                    temp = base[keys].copy()
                    temp["STRD_YYMM"] = yyyymm
                    temp["active_store_count"] = active_any.astype(int)
                    temp["active_count_at_start"] = active_start.astype(int)
                    temp["active_count_at_end"] = active_end.astype(int)
                    temp["opening_store_count"] = opened.astype(int)
                    temp["closure_store_count"] = closed.astype(int)
                    monthly_parts.append(temp.groupby(keys + ["STRD_YYMM"], as_index=False).sum())
                c = base.loc[base.close_date.between("2026-01-01", "2026-06-30"), ["SIDO_NM", "CCG_NM", "REGION_KEY", "TPBUZ_NM", "mapping_confidence", "source", "close_date"]].copy()
                if not c.empty:
                    c["STRD_YYMM"] = c.close_date.dt.strftime("%Y%m").astype(int)
                    closure_parts.append(c)

        for subtype_value, count in subtype_counter.most_common():
            subtype_rows.append({"source": spec["source_name"], "external_subtype": subtype_value, "row_count": count})
        for state_value, count in state_counter.most_common():
            state_rows.append({"source": spec["source_name"], "state": state_value, "row_count": count})
        for label, count in brand_counter.items():
            proxy_rows.append({"source": spec["source_name"], "name_pattern": label, "matched_rows": count, "all_rows": row_count, "share_pct": count / row_count * 100})

        quality_rows.append({
            "source": spec["source_name"], "file": str(path.relative_to(ROOT)), "encoding": enc,
            "row_count": row_count, "column_count": len(header), "has_open_date_field": "인허가일자" in header,
            "has_close_date_field": "폐업일자" in header, "has_business_state": "영업상태명" in header,
            "has_detailed_state": "상세영업상태명" in header, "open_date_present_pct": open_present / row_count * 100,
            "close_date_present_pct": close_present / row_count * 100, "bc_matched_regions": len(set(regions.REGION_KEY) & linked_regions),
            "bc_region_match_pct": len(set(regions.REGION_KEY) & linked_regions) / len(regions) * 100,
        })
        state_missing_pct = strict_closed_missing / strict_closed * 100 if strict_closed else np.nan
        anomaly_rows.append({
            "source": spec["source_name"], "open_date_parse_fail": parse_fail_open, "close_date_parse_fail": parse_fail_close,
            "close_before_open": close_before_open, "future_date": future_dates, "abnormal_open_year": abnormal_year,
        })
        coverage_rows.append({
            "source": spec["source_name"], "bc_regions": len(regions), "matched_regions": len(set(regions.REGION_KEY) & linked_regions),
            "unmatched_regions": len(set(regions.REGION_KEY) - linked_regions), "external_only_region_keys": len(linked_regions - set(regions.REGION_KEY)),
        })
        quality_rows[-1].update({"strict_closed_rows": strict_closed, "strict_closed_missing_close_date": strict_closed_missing, "strict_closed_missing_pct": state_missing_pct})
        for key_name, parts, field_present in [("관리번호", management_hashes, "관리번호" in header), ("인허가번호", permit_hashes, "인허가번호" in header), ("사업장명+주소", name_address_hashes, "사업장명" in header and ("도로명주소" in header or "지번주소" in header)), ("사업장명+좌표", name_coord_hashes, "사업장명" in header and "좌표정보(X)" in header and "좌표정보(Y)" in header)]:
            duplicated_keys, excess_rows = hashed_duplicate_summary(parts)
            duplicate_rows.append({"source": spec["source_name"], "key": key_name, "field_present": field_present, "duplicated_key_count": duplicated_keys, "excess_duplicate_rows": excess_rows, "action": "보고만 하고 자동 dedup하지 않음"})

    monthly = pd.concat(monthly_parts, ignore_index=True)
    keys = ["SIDO_NM", "CCG_NM", "REGION_KEY", "TPBUZ_NM", "mapping_confidence", "source", "STRD_YYMM"]
    monthly = monthly.groupby(keys, as_index=False)[["active_store_count", "active_count_at_start", "active_count_at_end", "opening_store_count", "closure_store_count"]].sum()
    monthly["net_store_change"] = monthly.opening_store_count - monthly.closure_store_count
    monthly["average_active_count"] = (monthly.active_count_at_start + monthly.active_count_at_end) / 2
    monthly["closure_rate_start"] = monthly.closure_store_count.div(monthly.active_count_at_start.replace(0, np.nan))
    monthly["closure_rate_avg"] = monthly.closure_store_count.div(monthly.average_active_count.replace(0, np.nan))
    monthly["TPBUZ_NO"] = monthly.TPBUZ_NM.map(BC_CODE)
    monthly = monthly[["STRD_YYMM", "SIDO_NM", "CCG_NM", "REGION_KEY", "TPBUZ_NO", "TPBUZ_NM", "active_store_count", "active_count_at_start", "active_count_at_end", "opening_store_count", "closure_store_count", "net_store_change", "closure_rate_start", "closure_rate_avg", "source", "mapping_confidence"]].sort_values(["source", "TPBUZ_NO", "REGION_KEY", "STRD_YYMM"])
    monthly.to_csv(PROCESSED / "source_specific_all_mapping_confidences.csv", index=False, encoding="utf-8-sig")
    monthly = monthly[monthly.mapping_confidence.isin(["HIGH", "MEDIUM"])].copy()
    monthly.to_csv(PROCESSED / "202601_202606_national_closure_panel_bc_mapping.csv", index=False, encoding="utf-8-sig")

    if closure_parts:
        closures = pd.concat(closure_parts, ignore_index=True)
        closures.groupby(["source", "STRD_YYMM"], as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "mapped_2026H1_closure_counts_by_month.csv", index=False, encoding="utf-8-sig")
        closures.groupby(["source", "SIDO_NM"], as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "mapped_2026H1_closure_counts_by_sido.csv", index=False, encoding="utf-8-sig")
        closures.groupby(["source", "REGION_KEY"], as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "mapped_2026H1_closure_counts_by_region.csv", index=False, encoding="utf-8-sig")
        closures.groupby(["source", "TPBUZ_NM", "mapping_confidence"], as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "mapped_2026H1_closure_counts_by_bc_industry.csv", index=False, encoding="utf-8-sig")
    if raw_closure_parts:
        raw_closures = pd.concat(raw_closure_parts, ignore_index=True)
        raw_closures.groupby(["source", "STRD_YYMM"], as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "raw_2026H1_closure_counts_by_month.csv", index=False, encoding="utf-8-sig")
        raw_closures.groupby(["source", "SIDO_NM"], dropna=False, as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "raw_2026H1_closure_counts_by_sido.csv", index=False, encoding="utf-8-sig")
        raw_closures.groupby(["source", "REGION_KEY"], dropna=False, as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "raw_2026H1_closure_counts_by_region.csv", index=False, encoding="utf-8-sig")
        raw_closures.groupby(["source", "external_subtype"], dropna=False, as_index=False).size().rename(columns={"size": "closure_count"}).to_csv(AUDIT / "raw_2026H1_closure_counts_by_subtype.csv", index=False, encoding="utf-8-sig")
    if missing_state_parts:
        ms = pd.concat(missing_state_parts, ignore_index=True).groupby(["source", "SIDO_NM", "external_subtype"], dropna=False, as_index=False)[["closed_state", "closed_state_missing_date"]].sum()
        ms["missing_pct"] = ms.closed_state_missing_date.div(ms.closed_state.replace(0, np.nan)).mul(100)
        ms.groupby(["source", "SIDO_NM"], dropna=False, as_index=False)[["closed_state", "closed_state_missing_date"]].sum().assign(missing_pct=lambda x: x.closed_state_missing_date.div(x.closed_state.replace(0, np.nan)).mul(100)).to_csv(AUDIT / "closure_state_date_missing_by_sido.csv", index=False, encoding="utf-8-sig")
        ms.groupby(["source", "external_subtype"], dropna=False, as_index=False)[["closed_state", "closed_state_missing_date"]].sum().assign(missing_pct=lambda x: x.closed_state_missing_date.div(x.closed_state.replace(0, np.nan)).mul(100)).to_csv(AUDIT / "closure_state_date_missing_by_subtype.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(quality_rows).to_csv(AUDIT / "raw_dataset_quality.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(state_rows).to_csv(AUDIT / "business_state_values.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(anomaly_rows).to_csv(AUDIT / "date_anomalies.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(anomaly_sample_rows).to_csv(AUDIT / "date_anomaly_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_rows).to_csv(AUDIT / "duplicate_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subtype_rows).to_csv(AUDIT / "external_subtype_values.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(AUDIT / "region_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(proxy_rows).to_csv(AUDIT / "proxy_business_name_profile.csv", index=False, encoding="utf-8-sig")

    checks = monthly.assign(accounting_expected=monthly.active_count_at_start + monthly.opening_store_count - monthly.closure_store_count)
    checks["accounting_difference"] = checks.active_count_at_end - checks.accounting_expected
    checks.groupby("source", as_index=False).agg(rows=("STRD_YYMM", "size"), mismatched_rows=("accounting_difference", lambda x: int(x.ne(0).sum())), max_abs_difference=("accounting_difference", lambda x: x.abs().max())).to_csv(AUDIT / "panel_accounting_checks.csv", index=False, encoding="utf-8-sig")
    return monthly, pd.DataFrame(quality_rows)


def write_mapping() -> pd.DataFrame:
    columns = ["BC_TPBUZ_NM", "external_source", "external_industry", "external_subtype", "mapping_rule", "mapping_confidence", "reason", "coverage", "known_false_positive", "known_false_negative", "recommended_use"]
    mapping = pd.DataFrame(MAPPING_ROWS, columns=columns)
    mapping.insert(0, "BC_TPBUZ_NO", mapping.BC_TPBUZ_NM.map(BC_CODE))
    mapping.to_csv(PROCESSED / "bc_to_closure_industry_mapping.csv", index=False, encoding="utf-8-sig")
    return mapping


def audit_nts_and_validation(local_panel: pd.DataFrame) -> None:
    nts_path = ROOT / "dataset/외부데이터/국세청_가동사업자/202601_202606_BC정확매핑_가동사업자.csv"
    if not nts_path.exists():
        return
    nts = pd.read_csv(nts_path)
    nts["REGION_KEY"] = nts.SIDO_NM.astype(str) + "|" + nts.CCG_NM.astype(str)
    summary = nts.groupby(["TP_BUZ_NM_CLEAN"], as_index=False).agg(rows=("STRD_YYMM", "size"), months=("STRD_YYMM", "nunique"), regions=("REGION_KEY", "nunique"), value_missing_pct=("active_business_count", lambda x: x.isna().mean() * 100), masked_pct=("active_business_masked", lambda x: x.fillna(False).mean() * 100))
    summary["role"] = "활성 사업자 분모·추세 validation; 직접 폐업 라벨 아님"
    summary.to_csv(AUDIT / "nts_active_business_audit.csv", index=False, encoding="utf-8-sig")

    # Same-population validation is unavailable.  Compare only direction/scale association for proxy sources.
    candidates = local_panel[local_panel.TPBUZ_NM.isin(["편의점", "슈퍼마켓"])].copy()
    agg = candidates.groupby(["STRD_YYMM", "SIDO_NM", "CCG_NM", "TPBUZ_NM", "source"], as_index=False).active_count_at_end.sum()
    merged = agg.merge(nts, left_on=["STRD_YYMM", "SIDO_NM", "CCG_NM", "TPBUZ_NM"], right_on=["STRD_YYMM", "SIDO_NM", "CCG_NM", "TP_BUZ_NM_CLEAN"], how="inner")
    rows = []
    for (industry, source), g in merged.groupby(["TPBUZ_NM", "source"]):
        valid = g[["active_count_at_end", "active_business_count"]].dropna()
        rows.append({
            "BC_industry": industry, "localdata_source": source, "n": len(valid),
            "pearson": valid.active_count_at_end.corr(valid.active_business_count, method="pearson") if len(valid) > 2 else np.nan,
            "spearman": valid.active_count_at_end.corr(valid.active_business_count, method="spearman") if len(valid) > 2 else np.nan,
            "interpretation": "서로 다른 모집단의 규모 sanity check; 일치도나 폐업 검증으로 해석 금지",
        })
    pd.DataFrame(rows).to_csv(AUDIT / "localdata_vs_nts_active_count_validation.csv", index=False, encoding="utf-8-sig")

    convenience = local_panel[(local_panel.TPBUZ_NM == "편의점") & local_panel.source.isin(["LOCALDATA 안전상비의약품판매업소", "LOCALDATA 휴게음식점"])].copy()
    wide = convenience.pivot_table(index=["STRD_YYMM", "REGION_KEY"], columns="source", values=["active_count_at_end", "opening_store_count", "closure_store_count"], aggfunc="sum")
    cross_rows = []
    for metric in ["active_count_at_end", "opening_store_count", "closure_store_count"]:
        z = wide[metric].dropna()
        if z.shape[1] == 2:
            cross_rows.append({
                "metric": metric, "n": len(z), "pearson": z.iloc[:, 0].corr(z.iloc[:, 1], method="pearson"),
                "spearman": z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman"),
                "interpretation": "같은 편의점의 서로 다른 인허가 하위집단; 두 값을 합산하지 않음",
            })
    pd.DataFrame(cross_rows).to_csv(AUDIT / "convenience_proxy_cross_validation.csv", index=False, encoding="utf-8-sig")


def write_metadata(quality: pd.DataFrame) -> None:
    q = quality.set_index("source").to_dict("index")
    rows = []
    for spec in SOURCES:
        path = spec["path"]
        info = q.get(spec["source_name"], {})
        rows.append({
            "source_name": spec["source_name"], "provider": spec["provider"], "official_page": spec["official_page"],
            "public_data_id": spec["public_data_id"], "download_url_or_api_endpoint": spec["download_url"],
            "retrieval_date": "2026-09-08" if path.stat().st_mtime >= pd.Timestamp("2026-09-08").timestamp() else "2026-09-04 (기존 원본 재사용)",
            "file_name": str(path.relative_to(ROOT)), "file_size": path.stat().st_size, "encoding": info.get("encoding"),
            "row_count": info.get("row_count"), "column_count": info.get("column_count"), "update_cycle": spec["update_cycle"],
            "spatial_scope": "전국 개별 인허가/등록", "temporal_scope": "최신 스냅샷+과거 인허가·폐업일; 2026H1 재구성",
            "license": "이용허락범위 제한 없음", "api_key_required": False, "download_success": True,
            "failure_reason": "", "checksum_sha256": sha256(path),
        })
    for p in sorted((RAW / "nts").glob("*.xlsx")):
        rows.append({
            "source_name": "국세청 월간 지역 경제지표", "provider": "국세청", "official_page": "https://tasis.nts.go.kr/",
            "public_data_id": "15061118 관련 월간 원본", "download_url_or_api_endpoint": "TASIS 공식 다운로드",
            "retrieval_date": "2026-09-02 (기존 원본 재사용)", "file_name": str(p.relative_to(ROOT)), "file_size": p.stat().st_size,
            "encoding": "xlsx", "row_count": np.nan, "column_count": np.nan, "update_cycle": "월",
            "spatial_scope": "전국 시군구×생활업종", "temporal_scope": p.stem, "license": "공공자료",
            "api_key_required": False, "download_success": True, "failure_reason": "", "checksum_sha256": sha256(p),
        })
    blocked = [
        ["식약처 식품판매업 폐업정보", "식품의약품안전처", "https://www.data.go.kr/data/15077647/openapi.do", "15077647", "LINK API", "API 활용신청·인증 필요; 이번 실행에 서비스키 없음"],
        ["식약처 즉석판매제조가공업 폐업정보", "식품의약품안전처", "https://www.data.go.kr/data/15077132/openapi.do", "15077132", "LINK API", "API 활용신청·인증 필요; BC 직접 매핑도 낮음"],
        ["국세청 사업자등록 상태조회", "국세청", "https://www.data.go.kr/data/15081808/openapi.do", "15081808", "OpenAPI", "LOCALDATA에 사업자등록번호가 없고 서비스키도 없어 NOT_USABLE_WITHOUT_BUSINESS_ID"],
        ["서울 상권분석서비스 점포_자치구", "서울특별시", "https://data.seoul.go.kr/dataList/OA-22173/S/1/datasetView.do", "15147215 / OA-22173", "서울 OpenAPI", "전국 주자료가 아니며 API 키 없이 원자료 다운로드 불가; validation 보류"],
    ]
    for name, provider, page, data_id, endpoint, reason in blocked:
        rows.append({
            "source_name": name, "provider": provider, "official_page": page, "public_data_id": data_id,
            "download_url_or_api_endpoint": endpoint, "retrieval_date": "2026-09-08", "file_name": "", "file_size": 0,
            "encoding": "", "row_count": np.nan, "column_count": np.nan, "update_cycle": "자료별 상이",
            "spatial_scope": "자료별 상이", "temporal_scope": "확인 불가", "license": "무료/공공누리(페이지 기준)",
            "api_key_required": True, "download_success": False, "failure_reason": reason, "checksum_sha256": "",
        })
    pd.DataFrame(rows).to_csv(META / "source_metadata.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for p in (PROCESSED, AUDIT, META):
        p.mkdir(parents=True, exist_ok=True)
    regions, lookup, pattern = bc_regions()
    mapping = write_mapping()
    panel, quality = audit_localdata(regions, lookup, pattern)
    audit_nts_and_validation(panel)
    write_metadata(quality)
    status = {
        "run_date": "2026-09-08",
        "bc_regions": int(len(regions)),
        "panel_rows": int(len(panel)),
        "panel_sources": sorted(panel.source.unique().tolist()),
        "high_bc_industries": sorted(mapping.loc[mapping.mapping_confidence.eq("HIGH"), "BC_TPBUZ_NM"].unique().tolist()),
        "high_or_medium_bc_industries": sorted(mapping.loc[mapping.mapping_confidence.isin(["HIGH", "MEDIUM"]), "BC_TPBUZ_NM"].unique().tolist()),
        "no_modeling_performed": True,
    }
    (META / "run_summary.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
