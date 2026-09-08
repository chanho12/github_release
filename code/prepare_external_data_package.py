#!/usr/bin/env python3
"""Build a documented, analysis-ready external-data package for the BC project.

The script never fabricates missing months or expands a station/snapshot to regions
that it does not directly represent. Existing official downloads are normalized into
small CSV panels under dataset/외부데이터, with coverage and checksum records.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BC_PATH = ROOT / "dataset/ABP_CONTEST_DATA.csv"
OUT_DEFAULT = ROOT / "dataset/외부데이터"
MONTHS = list(range(202601, 202607))
RKEY = ["SIDO_NM", "CCG_NM"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False)
                         .replace({"-": np.nan, "nan": np.nan}), errors="coerce")


def load_bc() -> tuple[pd.DataFrame, pd.DataFrame]:
    bc = pd.read_csv(BC_PATH, dtype={"GENDER_CD": str, "AGE_CD": str})
    bc["TP_BUZ_NM_CLEAN"] = bc.TP_BUZ_NM.str.replace(" ", "", regex=False)
    regions = bc[RKEY].drop_duplicates().sort_values(RKEY).reset_index(drop=True)
    return bc, regions


def build_population(out: Path, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    src = ROOT / "dataset/external_raw/mois/202601_202606_age_population.csv"
    target_dir = out / "행정안전부_주민등록인구"
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = target_dir / "202601_202606_주민등록_연령인구_원본.csv"
    shutil.copy2(src, raw_copy)

    raw = pd.read_csv(src, encoding="cp949", dtype=str)
    raw["full"] = (raw["행정구역"].str.replace(r"\s*\(\d+\)", "", regex=True)
                   .str.strip().str.replace(r"\s+", " ", regex=True))
    raw = raw.drop_duplicates("full", keep="last")
    keys = regions.copy()
    keys["full"] = np.where(keys.SIDO_NM.eq("세종특별자치시"), keys.SIDO_NM,
                            keys.SIDO_NM + " " + keys.CCG_NM)
    joined = keys.merge(raw, on="full", how="left", validate="one_to_one")
    age_columns = {
        "1": ["0~9세", "10~19세"], "2": ["20~29세"], "3": ["30~39세"],
        "4": ["40~49세"], "5": ["50~59세"],
        "6": ["60~69세", "70~79세", "80~89세", "90~99세", "100세 이상"],
    }
    rows = []
    for _, r in joined.iterrows():
        for month in MONTHS:
            prefix = f"{str(month)[:4]}년{str(month)[4:]}월_계_"
            total = clean_number(pd.Series([r.get(prefix + "총인구수")])).iloc[0]
            for age_cd, cols in age_columns.items():
                vals = [clean_number(pd.Series([r.get(prefix + col)])).iloc[0] for col in cols]
                value = float(np.sum(vals)) if np.all(pd.notna(vals)) else np.nan
                rows.append({"SIDO_NM": r.SIDO_NM, "CCG_NM": r.CCG_NM,
                             "STRD_YYMM": month, "AGE_CD": age_cd,
                             "population_age": value, "population_total": total})
    panel = pd.DataFrame(rows)
    panel.to_csv(target_dir / "202601_202606_주민등록_연령인구_분석패널.csv",
                 index=False, encoding="utf-8-sig")
    valid = panel.groupby(RKEY).population_age.apply(lambda x: x.notna().all())
    return panel, {
        "dataset": "주민등록 월별 시군구×연령 인구", "period": "2026.01~06",
        "months": panel.STRD_YYMM.nunique(), "matched_regions": int(valid.sum()),
        "bc_regions": len(regions), "region_match_pct": valid.mean() * 100,
        "value_missing_pct": panel.population_age.isna().mean() * 100,
        "industry_mapping": "해당 없음", "decision": "핵심 사용",
        "limit": "주민등록인구이며 외국인은 제외; 화성 신설 4구 결측",
    }


def normalize_nts_region(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["시군구"] = x["시군구"].astype(str).str.strip()
    merged = x["시도"].eq("전남광주통합특별시")
    gwangju_gu = {"동구", "서구", "남구", "북구", "광산구"}
    x.loc[merged & x.시군구.isin(gwangju_gu), "시도"] = "광주광역시"
    x.loc[merged & ~x.시군구.isin(gwangju_gu), "시도"] = "전라남도"
    return x.rename(columns={"기준년월": "STRD_YYMM", "시도": "SIDO_NM", "시군구": "CCG_NM"})


def build_nts(out: Path, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    src = ROOT / "dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx"
    target_dir = out / "국세청_가동사업자"
    target_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(src, sheet_name="국세청_시군구_101업종")
    d = normalize_nts_region(d)
    mapping = {"편의점": (4010, "편의점"), "슈퍼마켓": (4020, "슈퍼마켓")}
    frames = []
    for ext_name, (code, bc_name) in mapping.items():
        q = d[d.국세청업종.eq(ext_name)].copy()
        q["TP_BUZ_NO"], q["TP_BUZ_NM_CLEAN"] = code, bc_name
        q = q.rename(columns={"가동_당월": "active_business_count",
                              "가동_당월_마스킹": "active_business_masked"})
        frames.append(q[RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN",
                                "국세청업종", "active_business_count", "active_business_masked"]])
    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(target_dir / "202601_202606_BC정확매핑_가동사업자.csv",
                 index=False, encoding="utf-8-sig")
    map_df = pd.DataFrame([
        {"TP_BUZ_NO": 4010, "BC업종": "편의점", "외부업종": "편의점", "mapping_level": "정확"},
        {"TP_BUZ_NO": 4020, "BC업종": "슈퍼마켓", "외부업종": "슈퍼마켓", "mapping_level": "정확"},
    ])
    map_df.to_csv(target_dir / "BC_국세청_정확매핑표.csv", index=False, encoding="utf-8-sig")
    raw_rows = []
    for p in sorted((ROOT / "dataset/폐업률").glob("월간 지역 경제지표(2026년 *월).xlsx")):
        raw_rows.append({"raw_file": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size,
                         "sha256": sha256(p), "copied_into_package": False,
                         "reason": "기존 공식 원본 위치를 보존하고 분석용 정확매핑 패널만 패키지에 수록"})
    pd.DataFrame(raw_rows).to_csv(target_dir / "국세청_원본파일_위치_해시.csv",
                                  index=False, encoding="utf-8-sig")
    valid = (panel.dropna(subset=["active_business_count"])
             .groupby(RKEY + ["TP_BUZ_NO"]).STRD_YYMM.nunique().eq(6))
    matched_regions = len(set(map(tuple, panel[RKEY].drop_duplicates().to_numpy())) &
                          set(map(tuple, regions.to_numpy())))
    return panel, {
        "dataset": "국세청 월별 시군구×생활업종 가동사업자", "period": "2026.01~06",
        "months": panel.STRD_YYMM.nunique(), "matched_regions": matched_regions,
        "bc_regions": len(regions), "region_match_pct": matched_regions / len(regions) * 100,
        "value_missing_pct": panel.active_business_count.isna().mean() * 100,
        "industry_mapping": "정확 2/11(편의점·슈퍼마켓)", "decision": "제한적 사용",
        "limit": f"가동사업자는 물리적 점포와 다름; 완전 지역×업종 패널 {int(valid.sum())}개",
    }


def build_localdata(out: Path, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    src = ROOT / "dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx"
    target_dir = out / "LOCALDATA_영업점"
    target_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(src, sheet_name="시군구_월별_개폐업")
    d = d.rename(columns={"기준년월": "STRD_YYMM", "시도": "SIDO_NM", "시군구": "CCG_NM",
                          "가동_전월말": "stores_previous_month_end",
                          "가동_당월말": "stores_current_month_end",
                          "신규_당월": "openings_month", "폐업_당월": "closures_month"})
    mapping = {"중국음식": (8005, "중국음식"), "제과점": (8301, "제과점")}
    frames = []
    for ext_name, (code, bc_name) in mapping.items():
        q = d[d.연결업종.eq(ext_name)].copy()
        q["TP_BUZ_NO"], q["TP_BUZ_NM_CLEAN"] = code, bc_name
        frames.append(q[RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "연결업종",
                                "stores_previous_month_end", "stores_current_month_end",
                                "openings_month", "closures_month"]])
    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(target_dir / "202601_202606_BC정확매핑_영업점.csv",
                 index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"TP_BUZ_NO": 8005, "BC업종": "중국음식", "외부업종": "중국음식", "mapping_level": "HIGH"},
        {"TP_BUZ_NO": 8301, "BC업종": "제과점", "외부업종": "제과점", "mapping_level": "HIGH"},
    ]).to_csv(target_dir / "BC_LOCALDATA_HIGH매핑표.csv", index=False, encoding="utf-8-sig")
    raw_rows = []
    for p in sorted((ROOT / "dataset/localdata_raw").glob("*.csv")):
        raw_rows.append({"raw_file": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size,
                         "sha256": sha256(p), "copied_into_package": False,
                         "reason": "총 883MB 원본 중복을 피하고 기존 위치·해시를 보존"})
    pd.DataFrame(raw_rows).to_csv(target_dir / "LOCALDATA_원본파일_위치_해시.csv",
                                  index=False, encoding="utf-8-sig")
    valid = (panel.groupby(RKEY + ["TP_BUZ_NO"]).STRD_YYMM.nunique().eq(6))
    matched_regions = len(set(map(tuple, panel[RKEY].drop_duplicates().to_numpy())) &
                          set(map(tuple, regions.to_numpy())))
    return panel, {
        "dataset": "LOCALDATA 음식업 인허가·영업상태", "period": "2026.01~06 재구성",
        "months": panel.STRD_YYMM.nunique(), "matched_regions": matched_regions,
        "bc_regions": len(regions), "region_match_pct": matched_regions / len(regions) * 100,
        "value_missing_pct": panel.stores_current_month_end.isna().mean() * 100,
        "industry_mapping": "HIGH 2/11(중국음식·제과점)", "decision": "제한적 사용",
        "limit": f"인허가 모집단은 BC 가맹점과 다름; 완전 지역×업종 패널 {int(valid.sum())}개",
    }


def parse_weather_month(path: Path) -> pd.DataFrame:
    a = pd.read_excel(path, sheet_name="월요약자료1", header=None, skiprows=3)
    b = pd.read_excel(path, sheet_name="월요약자료2", header=None, skiprows=3)
    acols = ["station_id", "station_name", "avg_local_pressure_hpa", "avg_sea_pressure_hpa",
             "avg_temp_c", "temp_normal_diff_c", "avg_high_temp_c", "max_temp_c",
             "max_temp_day", "avg_low_temp_c", "min_temp_c", "min_temp_day",
             "days_max_ge_25c", "days_min_lt_0c", "avg_surface_min_temp_c",
             "avg_humidity_pct", "avg_cloud_tenths", "ground_temp_0_5m_c",
             "sunshine_hours", "sunshine_pct"]
    bcols = ["station_id", "station_name", "precipitation_total_mm", "precip_normal_diff_mm",
             "max_daily_precip_mm", "max_daily_precip_day", "days_precip_ge_0_1mm",
             "days_precip_ge_1mm", "days_precip_ge_10mm", "days_precip_ge_30mm",
             "total_evaporation_mm", "avg_wind_ms", "max_wind_ms", "max_wind_direction",
             "max_wind_day", "days_cloud_lt_2_5", "days_cloud_ge_7_5", "phenomenon_dust_days",
             "phenomenon_fog_days", "phenomenon_storm_days", "phenomenon_thunder_days",
             "phenomenon_snow_days", "phenomenon_frost_days", "phenomenon_ice_days"]
    a = a.iloc[:, :len(acols)].copy(); a.columns = acols
    b = b.iloc[:, :len(bcols)].copy(); b.columns = bcols
    for q in [a, b]:
        q["station_id"] = pd.to_numeric(q.station_id, errors="coerce")
        q["station_name"] = q.station_name.astype(str).str.replace(r"\s+", "", regex=True)
        q.dropna(subset=["station_id"], inplace=True)
        q["station_id"] = q.station_id.astype(int)
    # One duplicated presentation row can occur in the workbook; keep the first.
    a = a.drop_duplicates("station_id", keep="first")
    b = b.drop_duplicates("station_id", keep="first")
    d = a.merge(b.drop(columns="station_name"), on="station_id", how="outer", validate="one_to_one")
    d["STRD_YYMM"] = int(path.name[:6])
    numeric = [c for c in d.columns if c not in {"station_name", "max_wind_direction"}]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def weather_region_map(panel: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    # Only administratively unambiguous station-name matches are retained.
    variants = {"북춘천": "춘천", "북강릉": "강릉", "서청주": "청주", "북창원": "창원",
                "북부산": "부산", "정선군": "정선", "고창군": "고창", "영광군": "영광",
                "김해시": "김해", "순창군": "순창", "양산시": "양산", "보성군": "보성",
                "강진군": "강진", "의령군": "의령", "함양군": "함양", "광양시": "광양",
                "진도군": "진도", "청송군": "청송", "경주시": "경주"}
    # Known station locations whose station label is not the municipality name.
    special = {"대관령": ("강원특별자치도", "평창군"), "백령도": ("인천광역시", "옹진군"),
               "울릉도": ("경상북도", "울릉군"), "추풍령": ("충청북도", "영동군"),
               "흑산도": ("전라남도", "신안군"), "고산": ("제주특별자치도", "제주시"),
               "성산": ("제주특별자치도", "서귀포시")}
    region_lookup: dict[str, list[tuple[str, str]]] = {}
    for r in regions.itertuples(index=False):
        base = re.sub(r"(특별자치시|시|군|구)$", "", r.CCG_NM)
        if " " not in r.CCG_NM:  # district cities cannot be represented by one city station.
            region_lookup.setdefault(base, []).append((r.SIDO_NM, r.CCG_NM))
    rows = []
    for sid, name in panel[["station_id", "station_name"]].drop_duplicates().itertuples(index=False):
        normalized = variants.get(name, name)
        if name in special:
            target = special[name]; method = "공식 지점명 기반 수동 위치연결"
        else:
            candidates = region_lookup.get(normalized, [])
            if len(candidates) != 1:
                continue
            target = candidates[0]; method = "지점명-시군구명 유일 직접연결"
        if tuple(target) not in set(map(tuple, regions.to_numpy())):
            continue
        rows.append({"station_id": sid, "station_name": name, "SIDO_NM": target[0],
                     "CCG_NM": target[1], "mapping_method": method,
                     "scope_warning": "관측지점 값은 시군구 전체 평균이 아님"})
    m = pd.DataFrame(rows).drop_duplicates(["station_id", "SIDO_NM", "CCG_NM"])
    # If multiple stations represent a region, preserve all stations; downstream users aggregate explicitly.
    return m


def build_weather(out: Path, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_dir = out / "기상청_기상"
    files = sorted(target_dir.glob("2026??_기상월보.xls"))
    if len(files) != 6:
        return pd.DataFrame(), {"dataset": "기상청 월별 ASOS 기상", "period": "불완전",
            "months": len(files), "matched_regions": 0, "bc_regions": len(regions),
            "region_match_pct": 0.0, "value_missing_pct": 100.0,
            "industry_mapping": "해당 없음", "decision": "제외", "limit": "공식 월보 6개 미확보"}
    panel = pd.concat([parse_weather_month(p) for p in files], ignore_index=True)
    panel.to_csv(target_dir / "202601_202606_ASOS_월별관측지점.csv", index=False, encoding="utf-8-sig")
    mapping = weather_region_map(panel, regions)
    mapping.to_csv(target_dir / "ASOS_BC지역_보수적매핑.csv", index=False, encoding="utf-8-sig")
    regional = panel.merge(mapping, on=["station_id", "station_name"], how="inner")
    regional.to_csv(target_dir / "202601_202606_ASOS_BC지역_연결패널.csv", index=False, encoding="utf-8-sig")
    complete_stations = panel.groupby("station_id").STRD_YYMM.nunique().eq(6)
    mapped_regions = mapping[RKEY].drop_duplicates().shape[0]
    key_metrics = ["avg_temp_c", "precipitation_total_mm", "sunshine_hours", "avg_wind_ms"]
    return regional, {
        "dataset": "기상청 월별 ASOS 기상", "period": "2026.01~06",
        "months": panel.STRD_YYMM.nunique(), "matched_regions": mapped_regions,
        "bc_regions": len(regions), "region_match_pct": mapped_regions / len(regions) * 100,
        "value_missing_pct": panel[key_metrics].isna().mean().mean() * 100,
        "industry_mapping": "업종 없음", "decision": "제한적 사용" if mapped_regions >= 100 else "제외",
        "limit": f"관측소 {panel.station_id.nunique()}개(6개월 완전 {int(complete_stations.sum())}); 지점은 시군구 평균이 아니며 광역시 구 단위 대표 불가",
    }


def build_foreign(out: Path, regions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_dir = out / "법무부_등록외국인"
    files = sorted(target_dir.glob("2026??_등록외국인_지역_국적.xlsx"))
    rows = []
    for path in files:
        d = pd.read_excel(path, header=2)
        d = d.rename(columns={d.columns[0]: "SIDO_NM", d.columns[1]: "CCG_NM",
                              d.columns[2]: "gender", d.columns[3]: "registered_foreigners"})
        d["SIDO_NM"] = d.SIDO_NM.astype(str).str.strip()
        d["CCG_NM"] = d.CCG_NM.astype(str).str.strip()
        d["gender"] = d.gender.astype(str).str.strip()
        d["registered_foreigners"] = pd.to_numeric(d.registered_foreigners, errors="coerce")
        d = d[d.gender.isin(["총계", "총합계"]) & ~d.CCG_NM.isin(["총계", "총합계"])]
        d = d[~d.SIDO_NM.isin(["총합계", "nan"])]
        d["STRD_YYMM"] = int(path.name[:6])
        rows.append(d[RKEY + ["STRD_YYMM", "registered_foreigners"]])
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=RKEY + ["STRD_YYMM", "registered_foreigners"])
    panel = panel.drop_duplicates(RKEY + ["STRD_YYMM"], keep="last")
    panel.to_csv(target_dir / "202603_202606_등록외국인_시군구_분기스냅샷.csv",
                 index=False, encoding="utf-8-sig")
    bc_set = set(map(tuple, regions.to_numpy()))
    month_matches = []
    for month, q in panel.groupby("STRD_YYMM"):
        matched = len(bc_set & set(map(tuple, q[RKEY].drop_duplicates().to_numpy())))
        month_matches.append({"STRD_YYMM": month, "matched_regions": matched,
                              "bc_regions": len(regions), "match_pct": matched / len(regions) * 100})
    pd.DataFrame(month_matches).to_csv(target_dir / "BC지역_매칭품질.csv", index=False, encoding="utf-8-sig")
    matched_union = len(bc_set & set(map(tuple, panel[RKEY].drop_duplicates().to_numpy())))
    return panel, {
        "dataset": "법무부 등록외국인 시군구×국적", "period": "2026.03·06 스냅샷",
        "months": panel.STRD_YYMM.nunique(), "matched_regions": matched_union,
        "bc_regions": len(regions), "region_match_pct": matched_union / len(regions) * 100,
        "value_missing_pct": panel.registered_foreigners.isna().mean() * 100 if len(panel) else 100.0,
        "industry_mapping": "업종 없음", "decision": "제외",
        "limit": "시군구 상세는 3월·6월만 확인되어 1~6월 월별 추세·스파이크 검증 불가",
    }


def build_integrated_panel(out: Path, bc: pd.DataFrame, pop: pd.DataFrame,
                           nts: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    target_dir = out / "통합패널"
    target_dir.mkdir(parents=True, exist_ok=True)
    base = (bc.groupby(RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], as_index=False)
            [["amt", "cnt"]].sum())
    supply = pd.concat([
        nts.rename(columns={"active_business_count": "supply_count"})
           [RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "supply_count"]]
           .assign(external_source="국세청", source_scope="가동사업자"),
        local.rename(columns={"stores_current_month_end": "supply_count"})
             [RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "supply_count"]]
             .assign(external_source="LOCALDATA", source_scope="식품인허가 영업점"),
    ], ignore_index=True)
    integrated = base.merge(supply, on=RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN"],
                            how="inner", validate="one_to_one")
    integrated["amt_per_supply_proxy"] = integrated.amt / integrated.supply_count
    integrated["cnt_per_supply_proxy"] = integrated.cnt / integrated.supply_count
    integrated.to_csv(target_dir / "202601_202606_BC_정확업종_수요공급패널.csv",
                      index=False, encoding="utf-8-sig")
    pop_total = pop[RKEY + ["STRD_YYMM", "population_total"]].drop_duplicates()
    region_bc = bc.groupby(RKEY + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    region_pop = region_bc.merge(pop_total, on=RKEY + ["STRD_YYMM"], how="left", validate="one_to_one")
    region_pop["amt_per_registered_resident"] = region_pop.amt / region_pop.population_total
    region_pop["cnt_per_registered_resident"] = region_pop.cnt / region_pop.population_total
    region_pop.to_csv(target_dir / "202601_202606_BC_주민인구_지역패널.csv",
                      index=False, encoding="utf-8-sig")
    return integrated


def write_metadata(out: Path, audits: list[dict]) -> None:
    target_dir = out / "통합패널"
    audit = pd.DataFrame(audits)
    audit.to_csv(target_dir / "external_data_quality_audit.csv", index=False, encoding="utf-8-sig")
    sources = pd.DataFrame([
        ["행정안전부 주민등록 연령인구", "https://jumin.mois.go.kr/ageStatMonth.do", "기존 공식 다운로드 재사용", "2026.01~06"],
        ["국세청 월간 지역 경제지표", "https://tasis.nts.go.kr/", "기존 공식 다운로드·가공본 재사용", "2026.01~06"],
        ["LOCALDATA 일반음식점", "https://www.data.go.kr/data/15096283/standard.do", "기존 공식 전국파일 재사용", "인허가 이력으로 2026.01~06 재구성"],
        ["기상청 기상월보", "https://data.kma.go.kr/data/publication/publicationAsosList.do", "공식 엑셀 직접 다운로드", "2026.01~06"],
        ["법무부 등록외국인 상세통계 2026.03", "https://www.immigration.go.kr/bbs/immigration/227/605617/artclView.do", "공식 첨부 엑셀 직접 다운로드", "2026.03.31"],
        ["법무부 등록외국인 상세통계 2026.06", "https://www.immigration.go.kr/bbs/immigration/227/608715/artclView.do", "공식 첨부 엑셀 직접 다운로드", "2026.06.30"],
    ], columns=["dataset", "official_url", "acquisition_method", "period"])
    sources.to_csv(target_dir / "source_metadata.csv", index=False, encoding="utf-8-sig")
    records = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "checksums_sha256.csv":
            records.append({"relative_path": str(p.relative_to(out)), "size_bytes": p.stat().st_size,
                            "sha256": sha256(p)})
    pd.DataFrame(records).to_csv(target_dir / "checksums_sha256.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    bc, regions = load_bc()
    pop, a_pop = build_population(out, regions)
    nts, a_nts = build_nts(out, regions)
    local, a_local = build_localdata(out, regions)
    _, a_weather = build_weather(out, regions)
    _, a_foreign = build_foreign(out, regions)
    integrated = build_integrated_panel(out, bc, pop, nts, local)
    write_metadata(out, [a_pop, a_nts, a_local, a_weather, a_foreign])
    print(pd.DataFrame([a_pop, a_nts, a_local, a_weather, a_foreign]).to_string(index=False))
    print(f"integrated_supply_rows={len(integrated)} out={out}")


if __name__ == "__main__":
    main()
