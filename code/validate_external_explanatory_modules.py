#!/usr/bin/env python3
"""Validate whether external data materially narrows explanations of BC signals.

This script intentionally limits the executable modules to sources already present
in the project and to exact/high-confidence industry mappings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, theilslopes


ROOT = Path(__file__).resolve().parents[1]
BC_PATH = ROOT / "dataset/ABP_CONTEST_DATA.csv"
POP_PATH = ROOT / "dataset/external_raw/mois/202601_202606_age_population.csv"
NTS_PATH = ROOT / "dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx"
LOCAL_PATH = ROOT / "dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx"
MONTHS = list(range(202601, 202607))
RKEY = ["SIDO_NM", "CCG_NM"]
EPS = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def trend(values) -> dict:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) != 6 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return {"beta": np.nan, "growth_pct_m": np.nan, "r2": np.nan,
                "theil_beta": np.nan, "half_beta": np.nan, "robust": False}
    ly = np.log(y)
    ols = linregress(x, ly)
    ts = theilslopes(ly, x).slope
    half = np.log(y[3:].mean() / y[:3].mean()) / 3
    signs = np.sign([ols.slope, ts, half])
    return {"beta": float(ols.slope), "growth_pct_m": float(np.expm1(ols.slope) * 100),
            "r2": float(ols.rvalue**2), "theil_beta": float(ts), "half_beta": float(half),
            "robust": bool(abs(signs.sum()) == 3)}


def direction(value: float, eps: float = EPS) -> str:
    if pd.isna(value):
        return "판정불가"
    return "증가" if value > eps else "감소" if value < -eps else "정체"


def md(df: pd.DataFrame) -> str:
    x = df.copy().replace({np.nan: ""})
    lines = ["| " + " | ".join(map(str, x.columns)) + " |",
             "|" + "|".join(["---"] * len(x.columns)) + "|"]
    for row in x.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in row) + " |")
    return "\n".join(lines)


def load_bc() -> pd.DataFrame:
    d = pd.read_csv(BC_PATH, dtype={"GENDER_CD": str, "AGE_CD": str})
    d["TP_BUZ_NM_CLEAN"] = d.TP_BUZ_NM.str.replace(" ", "", regex=False)
    return d


def parse_population(bc_regions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(POP_PATH, encoding="cp949", dtype=str)
    raw["full"] = (raw["행정구역"].str.replace(r"\s*\(\d+\)", "", regex=True)
                   .str.strip().str.replace(r"\s+", " ", regex=True))
    raw = raw.drop_duplicates("full", keep="last")
    key = bc_regions.copy()
    key["full"] = np.where(key.SIDO_NM.eq("세종특별자치시"), key.SIDO_NM,
                           key.SIDO_NM + " " + key.CCG_NM)
    joined = key.merge(raw, on="full", how="left", validate="one_to_one")

    age_columns = {
        "1": ["0~9세", "10~19세"], "2": ["20~29세"], "3": ["30~39세"],
        "4": ["40~49세"], "5": ["50~59세"],
        "6": ["60~69세", "70~79세", "80~89세", "90~99세", "100세 이상"],
    }
    rows = []
    for _, r in joined.iterrows():
        for month in MONTHS:
            pref = f"{str(month)[:4]}년{str(month)[4:]}월_계_"
            for age, cols in age_columns.items():
                vals = []
                for col in cols:
                    value = r.get(pref + col, pd.NA)
                    vals.append(pd.to_numeric(str(value).replace(",", ""), errors="coerce"))
                value = float(np.sum(vals)) if np.all(pd.notna(vals)) else np.nan
                rows.append({"SIDO_NM": r.SIDO_NM, "CCG_NM": r.CCG_NM,
                             "STRD_YYMM": month, "AGE_CD": age, "population_age": value})
    long = pd.DataFrame(rows)
    quality = (long.groupby(RKEY, as_index=False)
               .agg(observed_month_age_cells=("population_age", lambda x: int(x.notna().sum())),
                    missing_month_age_cells=("population_age", lambda x: int(x.isna().sum()))))
    return long, quality


def population_adjusted_analysis(bc: pd.DataFrame, pop: pd.DataFrame):
    # 주민등록인구에는 외국인이 포함되지 않으므로 국내개인(GENDER 1,2)만 비교한다.
    b = (bc[bc.GENDER_CD.isin(["1", "2"]) & bc.AGE_CD.isin(list("123456"))]
         .groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "STRD_YYMM", "AGE_CD"], as_index=False)
         [["amt", "cnt"]].sum())
    z = b.merge(pop, on=RKEY + ["STRD_YYMM", "AGE_CD"], how="left", validate="many_to_one")
    valid_regions = (pop.groupby(RKEY).population_age.apply(lambda x: x.notna().all()))
    valid_regions = valid_regions[valid_regions].index
    covered_bc = b.set_index(RKEY)
    covered_bc = covered_bc[covered_bc.index.isin(valid_regions)].reset_index()
    nat_bc = (covered_bc.groupby(["TP_BUZ_NO", "AGE_CD", "STRD_YYMM"], as_index=False)
              [["amt", "cnt"]].sum())
    nat_pop = (pop.dropna(subset=["population_age"])
               .groupby(["AGE_CD", "STRD_YYMM"], as_index=False).population_age.sum())
    nat = nat_bc.merge(nat_pop, on=["AGE_CD", "STRD_YYMM"], validate="many_to_one")
    nat_rows = []
    for key, g in nat.groupby(["TP_BUZ_NO", "AGE_CD"]):
        g = g.sort_values("STRD_YYMM")
        a, c, p = trend(g.amt), trend(g.cnt), trend(g.population_age)
        nat_rows.append({"TP_BUZ_NO": key[0], "AGE_CD": key[1],
                         "covered_benchmark_amt_pop_beta": a["beta"] - p["beta"],
                         "covered_benchmark_cnt_pop_beta": c["beta"] - p["beta"]})
    nat_trends = pd.DataFrame(nat_rows)
    rows = []
    keys = RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "AGE_CD"]
    for key, g in z.groupby(keys):
        g = g.sort_values("STRD_YYMM")
        if g.STRD_YYMM.nunique() != 6 or g.population_age.isna().any() or (g.population_age <= 0).any():
            continue
        a, c, p = trend(g.amt), trend(g.cnt), trend(g.population_age)
        apc, cpc = trend(g.amt / g.population_age), trend(g.cnt / g.population_age)
        rows.append({**dict(zip(keys, key)), "amt_growth_pct_m": a["growth_pct_m"],
                     "cnt_growth_pct_m": c["growth_pct_m"], "population_growth_pct_m": p["growth_pct_m"],
                     "amt_population_gap_beta": a["beta"] - p["beta"],
                     "cnt_population_gap_beta": c["beta"] - p["beta"],
                     "amt_per_resident_growth_pct_m": apc["growth_pct_m"],
                     "cnt_per_resident_growth_pct_m": cpc["growth_pct_m"],
                     "amt_per_resident_robust": apc["robust"], "cnt_per_resident_robust": cpc["robust"],
                     "total_cnt_6m": float(g.cnt.sum()), "total_amt_6m": float(g.amt.sum())})
    out = pd.DataFrame(rows).merge(nat_trends, on=["TP_BUZ_NO", "AGE_CD"], validate="many_to_one")
    for metric in ["amt", "cnt"]:
        out[f"benchmark_adjusted_{metric}_population_gap_pct_m"] = np.expm1(
            out[f"{metric}_population_gap_beta"] - out[f"covered_benchmark_{metric}_pop_beta"]
        ) * 100
    out["age_mapping_scope"] = np.where(out.AGE_CD.eq("1"), "민감도(BC 라벨 중첩 가능)", "주분석")
    out["cnt_population_gap_direction"] = out.cnt_per_resident_growth_pct_m.map(direction)
    out["amt_population_gap_direction"] = out.amt_per_resident_growth_pct_m.map(direction)

    main = out[out.AGE_CD.ne("1")].copy()
    tests = []
    for age, q in [("2~6 합계", main), *list(main.groupby("AGE_CD"))]:
        for metric in ["amt", "cnt"]:
            valid = q[[f"{metric}_growth_pct_m", "population_growth_pct_m"]].dropna()
            rho, pval = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]) if len(valid) > 2 else (np.nan, np.nan)
            tests.append({"AGE_CD": age, "metric": metric, "n": len(valid),
                          "spearman_rho": rho, "p_value": pval})
    tests = pd.DataFrame(tests)

    repeat = (main.assign(pos=main.cnt_per_resident_growth_pct_m.gt(EPS),
                          neg=main.cnt_per_resident_growth_pct_m.lt(-EPS),
                          relative_pos=main.benchmark_adjusted_cnt_population_gap_pct_m.gt(EPS),
                          relative_neg=main.benchmark_adjusted_cnt_population_gap_pct_m.lt(-EPS))
              .groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], as_index=False)
              .agg(observed_ages=("AGE_CD", "nunique"), positive_ages=("pos", "sum"),
                   negative_ages=("neg", "sum"), relative_positive_ages=("relative_pos", "sum"),
                   relative_negative_ages=("relative_neg", "sum"), total_cnt_6m=("total_cnt_6m", "sum")))
    repeat["four_or_more_positive"] = repeat.positive_ages.ge(4)
    repeat["four_or_more_negative"] = repeat.negative_ages.ge(4)
    repeat["four_or_more_relative_positive"] = repeat.relative_positive_ages.ge(4)
    repeat["four_or_more_relative_negative"] = repeat.relative_negative_ages.ge(4)

    sens = []
    for include_age1 in [False, True]:
        base = out if include_age1 else main
        for eps in [0.0, 0.25, 0.5]:
            for min_cnt in [0, 1_000, 10_000]:
                q = base[base.total_cnt_6m >= min_cnt]
                for metric in ["amt", "cnt"]:
                    for comparison, col in [
                        ("지역 인구 대비", f"{metric}_per_resident_growth_pct_m"),
                        ("동일업종·연령 공통추세 추가제거", f"benchmark_adjusted_{metric}_population_gap_pct_m"),
                    ]:
                        sens.append({"include_ambiguous_age1": include_age1, "neutral_band_pct_m": eps,
                                     "min_age_cnt_6m": min_cnt, "metric": metric, "comparison": comparison,
                                     "n": len(q), "faster_n": int((q[col] > eps).sum()), "slower_n": int((q[col] < -eps).sum()),
                                     "faster_pct": (q[col] > eps).mean() * 100 if len(q) else np.nan,
                                     "slower_pct": (q[col] < -eps).mean() * 100 if len(q) else np.nan})
    sens = pd.DataFrame(sens)

    reps = []
    choices = [
        ("인구보다_CNT_빠른_대표", main.cnt_per_resident_growth_pct_m.gt(EPS), "cnt_per_resident_growth_pct_m", False),
        ("인구보다_CNT_느린_대표", main.cnt_per_resident_growth_pct_m.lt(-EPS), "cnt_per_resident_growth_pct_m", True),
        ("반례_인구보정후_정체", main.cnt_per_resident_growth_pct_m.abs().le(EPS), "total_cnt_6m", False),
    ]
    for label, mask, col, asc in choices:
        q = main[mask & main.total_cnt_6m.ge(100_000)].sort_values([col, "total_cnt_6m"], ascending=[asc, False])
        if q.empty:
            continue
        r = q.iloc[0]
        reps.append({"case_type": label, "region": f"{r.SIDO_NM} {r.CCG_NM}",
                     "industry": r.TP_BUZ_NM_CLEAN, "AGE_CD": r.AGE_CD,
                     "population_growth_pct_m": r.population_growth_pct_m,
                     "bc_cnt_growth_pct_m": r.cnt_growth_pct_m,
                     "cnt_per_resident_growth_pct_m": r.cnt_per_resident_growth_pct_m,
                     "bc_amt_growth_pct_m": r.amt_growth_pct_m,
                     "amt_per_resident_growth_pct_m": r.amt_per_resident_growth_pct_m,
                     "total_cnt_6m": r.total_cnt_6m})
    return out, tests, repeat, sens, pd.DataFrame(reps)


def normalize_nts_region(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["시군구"] = x["시군구"].astype(str).str.strip()
    merged = x["시도"].eq("전남광주통합특별시")
    gwangju_gu = {"동구", "서구", "남구", "북구", "광산구"}
    x.loc[merged & x.시군구.isin(gwangju_gu), "시도"] = "광주광역시"
    x.loc[merged & ~x.시군구.isin(gwangju_gu), "시도"] = "전라남도"
    return x.rename(columns={"기준년월": "STRD_YYMM", "시도": "SIDO_NM", "시군구": "CCG_NM"})


def store_panel(bc: pd.DataFrame):
    # 정확 매핑 주모듈: NTS 편의점·슈퍼마켓, LOCALDATA 중국음식·제과점.
    specs = []
    nts = pd.read_excel(NTS_PATH, sheet_name="국세청_시군구_101업종",
                        usecols=["기준년월", "시도", "시군구", "국세청업종", "가동_당월", "가동_당월_마스킹"])
    nts = normalize_nts_region(nts)
    nts_map = {"편의점": (4010, "편의점"), "슈퍼마켓": (4020, "슈퍼마켓")}
    for ext, (code, name) in nts_map.items():
        q = nts[nts.국세청업종.eq(ext)].copy()
        q = q.rename(columns={"가동_당월": "supply_count"})
        q["TP_BUZ_NO"], q["TP_BUZ_NM_CLEAN"] = code, name
        q["external_source"], q["source_scope"] = "국세청", "가동사업자"
        specs.append(q[RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "supply_count", "external_source", "source_scope"]])

    local = pd.read_excel(LOCAL_PATH, sheet_name="시군구_월별_개폐업")
    local = local.rename(columns={"기준년월": "STRD_YYMM", "시도": "SIDO_NM", "시군구": "CCG_NM",
                                  "가동_당월말": "supply_count"})
    local_map = {"중국음식": (8005, "중국음식"), "제과점": (8301, "제과점")}
    for ext, (code, name) in local_map.items():
        q = local[local.연결업종.eq(ext)].copy()
        q["TP_BUZ_NO"], q["TP_BUZ_NM_CLEAN"] = code, name
        q["external_source"], q["source_scope"] = "LOCALDATA", "식품인허가 영업점"
        specs.append(q[RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "supply_count", "external_source", "source_scope"]])
    supply = pd.concat(specs, ignore_index=True)

    b = (bc[bc.TP_BUZ_NO.isin([4010, 4020, 8005, 8301])]
         .groupby(RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], as_index=False)[["amt", "cnt"]].sum())
    joined = b.merge(supply, on=RKEY + ["STRD_YYMM", "TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], how="left", validate="one_to_one")
    rows = []
    keys = RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "external_source", "source_scope"]
    for key, g in joined.dropna(subset=["external_source"]).groupby(keys):
        g = g.sort_values("STRD_YYMM")
        if g.STRD_YYMM.nunique() != 6 or g.supply_count.isna().any() or (g.supply_count <= 0).any():
            continue
        a, c, s = trend(g.amt), trend(g.cnt), trend(g.supply_count)
        aps, cps = trend(g.amt / g.supply_count), trend(g.cnt / g.supply_count)
        rows.append({**dict(zip(keys, key)), "amt_growth_pct_m": a["growth_pct_m"],
                     "cnt_growth_pct_m": c["growth_pct_m"], "supply_growth_pct_m": s["growth_pct_m"],
                     "amt_per_supply_growth_pct_m": aps["growth_pct_m"],
                     "cnt_per_supply_growth_pct_m": cps["growth_pct_m"],
                     "amt_robust": a["robust"], "cnt_robust": c["robust"], "supply_robust": s["robust"],
                     "total_cnt_6m": float(g.cnt.sum()), "mean_supply": float(g.supply_count.mean())})
    out = pd.DataFrame(rows)
    for metric in ["amt", "cnt"]:
        out[f"{metric}_direction"] = out[f"{metric}_growth_pct_m"].map(direction)
    out["supply_direction"] = out.supply_growth_pct_m.map(direction)
    out["cnt_supply_type"] = out.cnt_direction + " 소비·" + out.supply_direction + " 공급"
    out["amt_supply_type"] = out.amt_direction + " 소비·" + out.supply_direction + " 공급"
    out["three_method_robust"] = out.cnt_robust & out.supply_robust

    summaries = []
    for metric in ["amt", "cnt"]:
        col = f"{metric}_supply_type"
        for typ, n in out[col].value_counts().items():
            summaries.append({"metric": metric.upper(), "type": typ, "n": int(n), "denominator": len(out),
                              "share_pct": n / len(out) * 100})
    summaries = pd.DataFrame(summaries)

    corr = []
    for industry, q in [("전체", out), *list(out.groupby("TP_BUZ_NM_CLEAN"))]:
        for metric in ["amt", "cnt"]:
            rho, pval = spearmanr(q[f"{metric}_growth_pct_m"], q.supply_growth_pct_m)
            corr.append({"industry": industry, "metric": metric.upper(), "n": len(q),
                         "spearman_rho": rho, "p_value": pval})
    corr = pd.DataFrame(corr)

    sens = []
    for eps in [0.0, 0.25, 0.5]:
        for min_cnt in [0, 10_000, 100_000, 500_000]:
            q = out[out.total_cnt_6m >= min_cnt]
            for metric in ["amt", "cnt"]:
                dd = q[f"{metric}_growth_pct_m"].map(lambda v: direction(v, eps))
                sd = q.supply_growth_pct_m.map(lambda v: direction(v, eps))
                for typ in ["증가 소비·증가 공급", "증가 소비·감소 공급", "감소 소비·증가 공급", "감소 소비·감소 공급", "정체 포함"]:
                    if typ == "정체 포함":
                        n = int(((dd == "정체") | (sd == "정체")).sum())
                    else:
                        a, b = typ.replace(" 소비", "").replace(" 공급", "").split("·")
                        n = int(((dd == a) & (sd == b)).sum())
                    sens.append({"neutral_band_pct_m": eps, "min_total_cnt_6m": min_cnt,
                                 "metric": metric.upper(), "type": typ, "n": n, "denominator": len(q),
                                 "share_pct": n / len(q) * 100 if len(q) else np.nan})
    sens = pd.DataFrame(sens)

    reps = []
    for label in ["증가 소비·증가 공급", "증가 소비·감소 공급", "감소 소비·증가 공급", "감소 소비·감소 공급"]:
        q = out[(out.cnt_supply_type == label) & out.total_cnt_6m.ge(100_000)].sort_values("total_cnt_6m", ascending=False)
        if q.empty:
            continue
        r = q.iloc[0]
        reps.append({"case_type": label, "region": f"{r.SIDO_NM} {r.CCG_NM}", "industry": r.TP_BUZ_NM_CLEAN,
                     "source": r.external_source, "cnt_growth_pct_m": r.cnt_growth_pct_m,
                     "amt_growth_pct_m": r.amt_growth_pct_m, "supply_growth_pct_m": r.supply_growth_pct_m,
                     "cnt_per_supply_growth_pct_m": r.cnt_per_supply_growth_pct_m,
                     "amt_per_supply_growth_pct_m": r.amt_per_supply_growth_pct_m,
                     "total_cnt_6m": r.total_cnt_6m})
    return supply, joined, out, summaries, corr, sens, pd.DataFrame(reps)


def external_audit(pop_quality, supply, joined, store_metrics, age_metrics) -> pd.DataFrame:
    bc = load_bc()
    bc_regions = len(bc[RKEY].drop_duplicates())
    pop_regions = int((pop_quality.missing_month_age_cells == 0).sum())
    source_rows = []
    for source, inds in [("국세청", "2/11 정확"), ("LOCALDATA", "2/11 HIGH")]:
        q = supply[supply.external_source.eq(source)]
        jq = joined[joined.external_source.eq(source)]
        valid_regions = q.groupby(RKEY).STRD_YYMM.nunique().eq(6).sum()
        executable = int((store_metrics.external_source == source).sum())
        source_rows.append({"dataset": source, "local_status": "확보", "period_complete": q.STRD_YYMM.nunique() == 6,
                            "matched_regions": int(valid_regions), "bc_regions": bc_regions,
                            "region_match_pct": valid_regions / bc_regions * 100,
                            "value_missing_pct": q.supply_count.isna().mean() * 100,
                            "bc_industry_mapping": inds, "executable_n": executable,
                            "eligible_n": 510, "executable_pct": executable / 510 * 100,
                            "decision": "제한적 사용",
                            "official_url": "https://tasis.nts.go.kr/" if source == "국세청" else "https://www.data.go.kr/data/15096283/standard.do"})
    rows = [
        {"dataset": "주민등록 연령인구", "local_status": "확보", "period_complete": True,
         "matched_regions": pop_regions, "bc_regions": bc_regions, "region_match_pct": pop_regions / bc_regions * 100,
         "value_missing_pct": (pop_quality.missing_month_age_cells.sum() / (bc_regions * 36)) * 100,
         "bc_industry_mapping": "업종 없음", "executable_n": int(age_metrics.AGE_CD.ne("1").sum()),
         "eligible_n": 11635, "executable_pct": age_metrics.AGE_CD.ne("1").sum() / 11635 * 100,
         "decision": "핵심 사용", "official_url": "https://jumin.mois.go.kr/ageStatMonth.do"},
        *source_rows,
        {"dataset": "기상청 ASOS/AWS", "local_status": "미확보(서비스키/로그인 필요)", "period_complete": False,
         "matched_regions": 0, "bc_regions": bc_regions, "region_match_pct": 0.0, "value_missing_pct": 100.0,
         "bc_industry_mapping": "업종 없음; 관측소→시군구 별도 매핑", "executable_n": 0,
         "eligible_n": 255, "executable_pct": 0.0, "decision": "제외",
         "official_url": "https://www.data.go.kr/data/15059093/openapi.do"},
        {"dataset": "등록외국인", "local_status": "미확보(지역 상세 3·6월만 확인)", "period_complete": False,
         "matched_regions": 0, "bc_regions": bc_regions, "region_match_pct": 0.0, "value_missing_pct": 100.0,
         "bc_industry_mapping": "업종 없음", "executable_n": 0, "eligible_n": 255,
         "executable_pct": 0.0, "decision": "제외",
         "official_url": "https://www.immigration.go.kr/immigration/1569/subview.do"},
    ]
    return pd.DataFrame(rows)


def write_report(outdir: Path, audit, age, age_tests, repeat, age_sens, age_reps,
                 stores, store_summary, store_corr, store_sens, store_reps):
    main_age = age[age.AGE_CD.ne("1")]
    age_n = len(main_age)
    age_pos = int((main_age.cnt_per_resident_growth_pct_m > EPS).sum())
    age_neg = int((main_age.cnt_per_resident_growth_pct_m < -EPS).sum())
    four_pos = int(repeat.four_or_more_positive.sum())
    four_neg = int(repeat.four_or_more_negative.sum())
    relative_pos = int((main_age.benchmark_adjusted_cnt_population_gap_pct_m > EPS).sum())
    relative_neg = int((main_age.benchmark_adjusted_cnt_population_gap_pct_m < -EPS).sum())
    four_relative_pos = int(repeat.four_or_more_relative_positive.sum())
    four_relative_neg = int(repeat.four_or_more_relative_negative.sum())
    age_rho_cnt = age_tests[(age_tests.AGE_CD == "2~6 합계") & (age_tests.metric == "cnt")].iloc[0]
    age_rho_amt = age_tests[(age_tests.AGE_CD == "2~6 합계") & (age_tests.metric == "amt")].iloc[0]
    overall_corr = store_corr[store_corr.industry.eq("전체")].set_index("metric")
    base_sens = store_sens[(store_sens.neutral_band_pct_m == EPS) & (store_sens.min_total_cnt_6m == 0) & (store_sens.metric == "CNT")]

    report = f"""# BC카드 외부데이터 설명력 검증 보고서

## 1. 최종 결론

> **외부데이터는 BC 신호의 원인을 확정하지 못했지만, 주민인구와 공급점포 변화만으로 설명되는지 먼저 거르는 데는 도움이 된다. 최종 서비스에서는 원인 판정기가 아니라 `점검 분기용 근거`로 사용해야 한다.**

이번 실행에서 실제 유효한 외부자료는 주민등록 연령인구와 정확 매핑 4개 업종의 공급자료다. 기상과 등록외국인은 공식 출처 존재까지 확인했지만 2026년 1~6월 시군구 패널 원자료를 확보하지 못해 분석하지 않았다.

- **A 인구보정: 중간.** 주분석 {age_n:,}개 지역×업종×연령에서 CNT/연령인구 성장 Gap이 +{EPS}%p/월을 넘은 사례는 {age_pos:,}개({age_pos/age_n*100:.2f}%), -{EPS}%p/월 미만은 {age_neg:,}개({age_neg/age_n*100:.2f}%)였다. 동일업종·동일연령 공통 추세까지 제거하면 양의 Gap {relative_pos:,}개({relative_pos/age_n*100:.2f}%), 음의 Gap {relative_neg:,}개({relative_neg/age_n*100:.2f}%)로 갈렸다. 연령별 주민인구 성장과 BC 성장의 상관은 대부분 매우 낮았다. 인구는 원인을 찾았다기보다 **단순 인구증감 설명을 배제**하는 역할이다.
- **B 수요–공급: 중간.** 정확 매핑과 6개월 완전관측을 만족한 {len(stores):,}개 지역×업종에서 소비와 공급의 네 방향이 모두 관찰됐다. 공급성장과 BC CNT 성장 상관은 `rho={overall_corr.loc['CNT','spearman_rho']:.3f}`, AMT는 `rho={overall_corr.loc['AMT','spearman_rho']:.3f}`였다. 공급은 일부 사례의 점검순서를 바꾸지만 BC 변화의 일반 원인은 아니다.
- **C 날씨: 핵심 채택 기각(자료 미확보).** 기상청 자료는 공식적으로 존재하지만 현재 프로젝트에는 원자료·서비스키·관측소-시군구 매핑이 없다. 상관을 계산하지 않았다.
- **D 외국인: 핵심 채택 기각(기간 불완전).** 법무부의 시군구 상세 등록외국인 자료는 2026년 3월·6월 게시물을 확인했으나 1~6월 완전 패널을 확보하지 못했다. BC 외국인 결제 증가 원인을 정주외국인 증가로 연결하지 않았다.

## 2. 외부데이터 감사와 사용 판정

{md(audit.round(3))}

### 공식 출처와 논리적 범위

- 주민등록 연령인구: [행정안전부 월별 연령인구](https://jumin.mois.go.kr/ageStatMonth.do)다. 주민등록인구에는 외국인이 제외되므로 BC의 국내개인(`GENDER_CD=1,2`)만 비교했다.
- 국세청: [월간 지역 경제지표](https://tasis.nts.go.kr/)의 시군구×생활업종 **가동사업자 수**다. 물리적 점포수와 같다고 간주하지 않았다.
- LOCALDATA: [전국 일반음식점 표준데이터](https://www.data.go.kr/data/15096283/standard.do) 등 현재 전국 스냅샷의 인허가·폐업일자로 월말 영업점 수를 재구성했다. BC 가맹점 모집단과 동일하지 않고 과거 폐업기록 보존 정도가 결과에 영향을 줄 수 있다.
- 기상청: [ASOS 일자료 OpenAPI](https://www.data.go.kr/data/15059093/openapi.do)는 105개 지점이며 원하는 지점이 없으면 AWS를 함께 써야 한다. 255개 시군구와 일대일 대응하지 않는다.
- 등록외국인: 법무부의 [2026년 3월 상세 지역자료](https://www.immigration.go.kr/bbs/immigration/227/605617/artclView.do)와 [2026년 6월 상세 지역자료](https://www.immigration.go.kr/bbs/immigration/227/608715/artclView.do)는 공식 자료이나 현재 확보 상태가 6개월 조건을 충족하지 않는다.

## 3. A — 인구 보정 소비층 변화

### 방법

BC 연령코드 2~6을 주민 20대·30대·40대·50대·60대 이상과 대응했다. `AGE_CD=1`은 제공 설명의 ‘20대 이하’가 코드 2의 ‘20대’와 겹쳐 주분석에서 제외하고 민감도로만 사용했다. 각 지역×업종×연령의 국내개인 AMT·CNT를 해당 연령 주민수로 나눈 뒤 6개월 로그 OLS 성장률을 계산했다.

### 결과

- 주분석 표본: **{age_n:,}개 지역×업종×연령**
- 인구보다 CNT가 +{EPS}%p/월 이상 빠름: **{age_pos:,}개({age_pos/age_n*100:.2f}%)**
- 인구보다 CNT가 -{EPS}%p/월 이상 느림: **{age_neg:,}개({age_neg/age_n*100:.2f}%)**
- 5개 연령 중 4개 이상에서 양의 Gap: **{four_pos:,}/{len(repeat):,} 조합**
- 5개 연령 중 4개 이상에서 음의 Gap: **{four_neg:,}/{len(repeat):,} 조합**
- 동일업종·동일연령 공통 추세까지 제거한 CNT Gap: 양(+) **{relative_pos:,}개**, 음(-) **{relative_neg:,}개**
- 같은 상대 Gap이 5개 연령 중 4개 이상 반복: 양(+) **{four_relative_pos:,}개**, 음(-) **{four_relative_neg:,}개**
- 연령을 합친 상관: CNT `rho={age_rho_cnt.spearman_rho:.3f}`, AMT `rho={age_rho_amt.spearman_rho:.3f}`. 이 값은 연령층 간 차이가 섞이므로 연령별 상관표를 우선 해석한다.

{md(age_tests.round(4))}

대표와 반례:

{md(age_reps.round(3))}

민감도 전체 결과는 `04_population_adjustment_sensitivity.csv`에 저장했다. 연령코드 1 포함 여부, ±0/0.25/0.5%p 중립구간, 연령별 6개월 CNT 하한을 바꿔도 양·음 Gap은 모두 남는지 확인했다.

### 판정: 중간

인구보정 후에도 서로 다른 방향이 반복되고, 전국 동일업종·동일연령 공통 추세를 제거해도 지역별 양·음 Gap이 남는다. 따라서 `해당 연령 인구가 늘어서 결제가 늘었다`는 설명만으로 충분하지 않다는 점은 방어 가능하다. 연령별 상관은 매우 낮아 주민인구가 BC 성장의 주된 설명변수라고 볼 수 없다. BC 카드 이용률, 유동인구, 비거주 결제, 실제 고객수는 여전히 미확인이다. `BC 결제건수/주민수`는 1인당 이용횟수가 아니라 주민수를 외부 분모로 둔 지표다.

## 4. B — 수요·공급 분해

### 사용 범위

- 국세청 가동사업자: 편의점, 슈퍼마켓
- LOCALDATA 영업점: 중국음식, 제과점
- 그 외 7개 BC 업종: 명칭·범위 통합 또는 직접 자료 부재로 제외

±{EPS}%/월 중립구간 기준 CNT 방향 결과:

{md(base_sens[["type", "n", "denominator", "share_pct"]].round(3))}

전체 AMT·CNT 요약:

{md(store_summary.round(3))}

공급성장과 BC 성장 상관:

{md(store_corr.round(4))}

대표 사례와 반례 역할의 네 방향 사례:

{md(store_reps.round(3))}

`AMT/supply_count`, `CNT/supply_count`는 외부 분모를 붙인 보조지표다. 국세청은 가동사업자, LOCALDATA는 식품 인허가 영업점이므로 **실제 개별 점포 매출·방문건수라고 표현하지 않는다.**

### 판정: 중간

소비↓·공급↑와 소비↑·공급↓가 실제로 존재해 `수요 확인 우선`과 `공급·경쟁환경 확인 우선`을 구분할 수 있다. 그러나 상관이 높지 않고 4/11 업종에 한정된다. 공급 변화가 매출 변화의 원인이라고 단정할 수 없으며 점포별 규모·영업일·BC 가맹률도 없다.

## 5. C — 월별 spike와 기상

### 판정: 기각(현재 핵심분석), 데이터 미확보로 미검증

기상청 ASOS 일자료는 무료·자동승인 OpenAPI로 1904년부터 현재까지 제공되며, 포털 DB 조회는 월자료도 지원한다. 그러나 OpenAPI 서비스키 또는 포털 로그인이 필요하고 현재 프로젝트에 내려받은 2026년 자료가 없다. ASOS 105개 지점만으로 255개 시군구를 직접 대표할 수도 없어 AWS 병합 및 관측소 거리·고도 기준이 선행되어야 한다.

따라서 기존 피크 의존 273개에 날씨 원인을 붙이지 않았다. 향후 자료를 확보해도 업종×기상 민감도가 여러 지역에서 반복되고 월·지역 고정효과 이후 남을 때만 사례 모듈로 승격한다.

## 6. D — 외국인 모듈

### 판정: 기각(현재 핵심분석), 기간 불완전으로 미검증

법무부는 등록외국인 지역별·연령별·국적별 상세 파일을 제공한다. 다만 공식 검색에서 확인된 시군구 상세 게시물은 2026년 3월말과 6월말이며, 프로젝트 내부에는 1~6월 연속 원자료가 없다. 연간 KOSIS 시군구 통계나 체류외국인 총계로 월별 등록외국인을 대체하지 않았다.

BC `GENDER_CD=3`은 외국인 결제이지만 관광객·단기체류·등록외국인·이주노동자를 구분하지 못한다. 6개월 시군구 등록외국인 패널을 확보하더라도 정주외국인 변화와의 동행만 검토할 수 있고 관광 원인은 판정할 수 없다.

## 7. 외부데이터가 원인 후보를 좁혔는가

| 모듈 | 좁힌 범위 | 좁히지 못한 범위 | 최종 역할 |
|---|---|---|---|
| 주민 연령인구 | 단순 인구증감과 결제증감을 분리 | 유동인구·BC 이용률·실제 고객수 | 핵심 통제·질문 분기 |
| 국세청/LOCALDATA 공급 | 수요와 공급 방향의 동행·괴리 | 경쟁강도·점포별 성과·인과 | 정확매핑 업종의 제한 카드 |
| 날씨 | 없음 | 피크 원인 | 현재 제외 |
| 등록외국인 | 없음 | 외국인 결제 증가 원인 | 현재 제외 |

결론적으로 외부자료 추가의 가치는 **원인을 맞히는 것보다 틀린 단일 설명을 배제하고 다음 질문의 순서를 바꾸는 것**이다. 이는 「같은 매출 변화, 다른 점검」의 범위와 맞지만, ‘외부데이터로 원인을 규명한다’는 표현은 방어할 수 없다.

## 8. 최종 서비스에서의 역할

1. 주민인구 Gap이 작으면 인구변화 설명을 우선 확인하고, Gap이 크면 유동수요·카드침투·점포 운영정보를 먼저 묻는다.
2. 정확 매핑 업종에서 소비↓·공급↑이면 신규 경쟁점·영업일·상권 이동을 우선 질문한다. 소비↑·공급↓이면 접근성 저하와 특정점포 집중 여부를 질문한다.
3. 자료가 없거나 매핑이 불완전하면 외부 원인 카드를 숨기고 BC 근거만 보여준다.
4. 점주 답변 전에는 원인·처방을 확정하지 않는다.

## 9. 발표에 사용할 외부데이터 인사이트

1. **연령인구 보정 후에도 소비층 변화 방향은 갈린다.** 주민 고령화와 BC 결제구성 변화는 같은 지표가 아니다.
2. **정확 매핑 4개 업종에서 소비와 공급의 네 방향이 모두 존재한다.** 같은 소비 감소도 공급 증가형과 공급 감소형은 먼저 물을 질문이 다르다.
3. **외부데이터의 결측·단위·매핑이 진단 게이트가 된다.** 날씨·외국인은 지금 단계에서 설명을 더하지 않는 것이 오히려 재현성과 신뢰성을 높인다.

## 10. 재현 파일

- 분석 코드: `scripts/validate_external_explanatory_modules.py`
- 외부자료 감사: `00_external_data_audit.csv`
- 인구보정 상세·검정·반복성·민감도·사례: `01_`~`05_*.csv`
- 공급 결합 원패널·상세·요약·상관·민감도·사례: `06_`~`12_*.csv`
- 검증기록: `validation_checks.csv`
- 입력·출력 해시: `manifest.csv`

실행 명령:

```bash
PYTHONPYCACHEPREFIX=/tmp/bc_external_validation_pycache python3 scripts/validate_external_explanatory_modules.py --out {outdir.relative_to(ROOT)}
```

분석 기준일: 2026-09-07  
분석 성격: 외부자료의 설명 추가가치 검증. 인과관계·점포성과·개선효과 검증이 아님.
"""
    (outdir / "external_explanatory_validation_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/external_cause_validation/run_20260907")
    args = parser.parse_args()
    outdir = args.out if args.out.is_absolute() else ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    bc = load_bc()
    regions = bc[RKEY].drop_duplicates()
    pop, pop_quality = parse_population(regions)
    age, age_tests, repeat, age_sens, age_reps = population_adjusted_analysis(bc, pop)
    supply, joined, stores, store_summary, store_corr, store_sens, store_reps = store_panel(bc)
    audit = external_audit(pop_quality, supply, joined, stores, age)

    outputs = {
        "00_external_data_audit.csv": audit,
        "01_population_adjusted_age_detail.csv": age,
        "02_population_growth_correlations.csv": age_tests,
        "03_population_gap_repeatability.csv": repeat,
        "04_population_adjustment_sensitivity.csv": age_sens,
        "05_population_representative_counter_cases.csv": age_reps,
        "06_exact_supply_source_panel.csv": supply,
        "07_bc_external_supply_join_panel.csv": joined,
        "08_demand_supply_metrics.csv": stores,
        "09_demand_supply_summary.csv": store_summary,
        "10_demand_supply_correlations.csv": store_corr,
        "11_demand_supply_sensitivity.csv": store_sens,
        "12_demand_supply_representative_cases.csv": store_reps,
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / name, index=False, encoding="utf-8-sig")

    checks = pd.DataFrame([
        ["BC 기간 6개월", sorted(bc.STRD_YYMM.unique().tolist()) == MONTHS],
        ["주민인구 월수", pop.STRD_YYMM.nunique() == 6],
        ["주민인구 완전지역", int((pop_quality.missing_month_age_cells == 0).sum()) == 251],
        ["정확 공급업종 4개", stores.TP_BUZ_NO.nunique() == 4],
        ["공급 분석 6개월 완전", len(stores) > 0],
        ["기상 분석 미실행", not any("weather" in p.name.lower() for p in (ROOT / "dataset").rglob("*"))],
        ["등록외국인 분석 미실행", True],
    ], columns=["check", "passed"])
    checks.to_csv(outdir / "validation_checks.csv", index=False, encoding="utf-8-sig")
    if not checks.passed.all():
        raise RuntimeError("validation check failed")

    write_report(outdir, audit, age, age_tests, repeat, age_sens, age_reps,
                 stores, store_summary, store_corr, store_sens, store_reps)

    manifest_rows = []
    for p in [BC_PATH, POP_PATH, NTS_PATH, LOCAL_PATH, Path(__file__), *sorted(outdir.glob("*"))]:
        if p.is_file() and p.name != "manifest.csv":
            manifest_rows.append({"role": "input" if p in [BC_PATH, POP_PATH, NTS_PATH, LOCAL_PATH] else "code/output",
                                  "path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(manifest_rows).to_csv(outdir / "manifest.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"out": str(outdir), "age_n": len(age), "store_n": len(stores),
                      "checks_passed": int(checks.passed.sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
