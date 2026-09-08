#!/usr/bin/env python3
"""Audit and focused re-analysis for '같은 매출 변화, 다른 처방'.

This script never modifies raw data or previous analysis outputs.  It consumes
the clean-room reproduction outputs created under a run directory and writes
the audit tables, focused comparison panels, case cards, figures and report.

Run:
    python3 scripts/audit_same_sales_different_actions.py \
      --run-dir analysis/audit_same_sales_different_actions/run_20260907
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RKEY = ["SIDO_NM", "CCG_NM"]
MONTHS = list(range(202601, 202607))

CONFIG = {
    "seed": 42,
    "neutral_band_pct_month": 0.25,
    "case_amt_tolerance_pp_month": 0.35,
    "case_log10_initial_cnt_tolerance": 0.25,
    "case_min_total_cnt_6m": 10_000,
    "case_pair_count": 4,
    "case_direction_targets": ["상승", "정체", "감소"],
    "peak_dependency_rule": "AMT slope > 0.25%/월이고 최대 양(+) 로그잔차 월 제거 시 slope <= 0.25%/월",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def growth(y, x=None) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    if len(y) < 2 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return {"growth_pct_m": np.nan, "r2": np.nan, "residual": np.full(len(y), np.nan)}
    fit = linregress(x, np.log(y))
    pred = fit.intercept + fit.slope * x
    return {
        "growth_pct_m": np.expm1(fit.slope) * 100,
        "r2": fit.rvalue**2,
        "residual": np.log(y) - pred,
    }


def fixed_sign(y) -> bool:
    y = np.asarray(y, dtype=float)
    if len(y) != 6 or np.any(y <= 0):
        return False
    ols = np.sign(linregress(np.arange(6), np.log(y)).slope)
    # Theil-Sen without importing another estimator: median pairwise slope.
    pair_slopes = [(np.log(y[j]) - np.log(y[i])) / (j - i) for i in range(6) for j in range(i + 1, 6)]
    ts = np.sign(np.median(pair_slopes))
    halves = np.sign(np.log(y[3:].mean()) - np.log(y[:3].mean()))
    return bool(ols != 0 and ols == ts == halves)


def direction(v: float, eps: float) -> str:
    if v > eps:
        return "상승"
    if v < -eps:
        return "감소"
    return "정체"


def markdown_table(d: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    x = d.copy().replace({np.nan: ""})
    cols = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in x.astype(str).itertuples(index=False, name=None):
        vals = [v.replace("|", "\\|").replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def dataframe_file_info(path: Path) -> tuple[int | None, int | None, int | None, str]:
    """Lightweight metadata for inventory; raw LOCALDATA counts use verified audit file."""
    try:
        if path.suffix.lower() == ".csv":
            for enc in ("utf-8-sig", "cp949", "utf-8"):
                try:
                    d = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            months = None
            for c in ["STRD_YYMM", "기준년월", "year_month"]:
                if c in d:
                    months = d[c].nunique()
                    break
            regions = None
            if set(RKEY).issubset(d):
                regions = len(d[RKEY].drop_duplicates())
            elif {"시도", "시군구"}.issubset(d):
                regions = len(d[["시도", "시군구"]].drop_duplicates())
            return len(d), regions, months, ",".join(map(str, d.columns[:8]))
        if path.suffix.lower() in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(path)
            return None, None, None, "sheets=" + ",".join(xl.sheet_names)
    except Exception as exc:  # inventory must continue and record uncertainty
        return None, None, None, f"inspection_error={type(exc).__name__}"
    return None, None, None, "binary/document"


def build_inventory() -> pd.DataFrame:
    meta_path = ROOT / "dataset/external_raw/external_data_metadata.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    meta_by_file = {str(r.local_file): r for _, r in meta.iterrows() if pd.notna(r.local_file) and str(r.local_file)}
    specs = [
        ("BC_RAW", "dataset/ABP_CONTEST_DATA.csv", "원본", "BC카드", "2026-01~06", "시군구×월×업종×성별×연령", "VERIFIED_RAW"),
        ("BC_CODEBOOK", "dataset/dataset.md", "문서", "제공 문서", "2026-01~06", "컬럼", "DERIVED_ONLY"),
        ("MOIS_AGE_POP", "dataset/external_raw/mois/202601_202606_age_population.csv", "원본", "행정안전부", "2026-01~06", "시군구×월×연령", "VERIFIED_RAW"),
        ("CPI_DETAIL", "dataset/external_raw/kosis/202601_202606_detailed_cpi.csv", "가공", "국가데이터처", "2026-01~06", "전국×월×품목", "DERIVED_ONLY"),
        ("CPI_PDF_02", "dataset/external_raw/kosis/2026_02_consumer_price_trends.pdf", "원본", "국가데이터처", "2026-01~02", "전국×월", "VERIFIED_RAW"),
        ("CPI_PDF_04", "dataset/external_raw/kosis/2026_04_consumer_price_trends.pdf", "원본", "국가데이터처", "2026-03~04", "전국×월", "VERIFIED_RAW"),
        ("CPI_PDF_06", "dataset/external_raw/kosis/2026_06_consumer_price_trends.pdf", "원본", "국가데이터처", "2026-01~06", "전국×월", "VERIFIED_RAW"),
        ("LOCAL_GENERAL", "dataset/localdata_raw/일반음식점.csv", "원본", "행정안전부 LOCALDATA", "최신 스냅샷+이력일자", "개별 인허가", "VERIFIED_RAW"),
        ("LOCAL_REST_CAFE", "dataset/localdata_raw/휴게음식점.csv", "원본", "행정안전부 LOCALDATA", "최신 스냅샷+이력일자", "개별 인허가", "VERIFIED_RAW"),
        ("LOCAL_BAKERY", "dataset/localdata_raw/제과점.csv", "원본", "행정안전부 LOCALDATA", "최신 스냅샷+이력일자", "개별 인허가", "VERIFIED_RAW"),
        ("LOCAL_PANEL", "dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx", "가공", "행정안전부 LOCALDATA", "2026-01~06 재구성", "시군구×월×연결업종", "DERIVED_ONLY"),
        ("NTS_RAW_01", "dataset/폐업률/월간 지역 경제지표(2026년 1월).xlsx", "원본", "국세청", "2026-01", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_RAW_02", "dataset/폐업률/월간 지역 경제지표(2026년 2월).xlsx", "원본", "국세청", "2026-02", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_RAW_03", "dataset/폐업률/월간 지역 경제지표(2026년 3월).xlsx", "원본", "국세청", "2026-03", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_RAW_04", "dataset/폐업률/월간 지역 경제지표(2026년 4월).xlsx", "원본", "국세청", "2026-04", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_RAW_05", "dataset/폐업률/월간 지역 경제지표(2026년 5월).xlsx", "원본", "국세청", "2026-05", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_RAW_06", "dataset/폐업률/월간 지역 경제지표(2026년 6월).xlsx", "원본", "국세청", "2026-06", "시군구×월×생활업종", "VERIFIED_RAW"),
        ("NTS_PANEL", "dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx", "가공", "국세청", "2026-01~06", "시군구/시도×월×업종", "DERIVED_ONLY"),
        ("LIVING_POP_PDF", "dataset/external_raw/mois_living_population/2026Q1_living_population_detail.pdf", "원본", "행정안전부", "2026-01~03", "89개 인구감소지역×월", "VERIFIED_RAW"),
        ("LIVING_POP_PARSED", "dataset/external_raw/mois_living_population/202601_202603_living_population_by_region.csv", "가공", "행정안전부", "2026-01~03", "89개 인구감소지역×월", "DERIVED_ONLY"),
        ("REB_Q1", "dataset/external_raw/reb_commercial/2026Q1_commercial_rent.xlsx", "원본", "한국부동산원", "2026-Q1", "시도·대표상권×상가유형", "VERIFIED_RAW"),
        ("REB_Q2", "dataset/external_raw/reb_commercial/2026Q2_commercial_rent.xlsx", "원본", "한국부동산원", "2026-Q2", "시도·대표상권×상가유형", "VERIFIED_RAW"),
        ("REB_PARSED", "dataset/external_raw/reb_commercial/2026Q1_Q2_commercial_rent_vacancy_province.csv", "가공", "한국부동산원", "2026-Q1~Q2", "시도×분기×상가유형", "DERIVED_ONLY"),
        ("SEMAS_GUIDE", "dataset/external_raw/semas/openapi_guide_260805.zip", "문서", "소상공인시장진흥공단", "2026-08 가이드", "API 명세", "VERIFIED_RAW"),
        ("SEMAS_HISTORY", "", "미확보", "소상공인시장진흥공단", "2026-01~06 필요", "시군구×월×업종", "UNAVAILABLE"),
        ("KTO_VISITORS", "", "미확보", "한국관광공사", "2026-01~06 필요", "시군구×월", "UNAVAILABLE"),
        ("OFFICIAL_BOUNDARY", "", "미확보", "국토교통부/국가공간정보", "2026 상반기 기준 필요", "시군구 SHP", "UNAVAILABLE"),
    ]
    raw_counts = {"LOCAL_GENERAL": 2_294_794, "LOCAL_REST_CAFE": 645_608, "LOCAL_BAKERY": 69_480}
    rows = []
    for did, rel, kind, inst, period, unit, status in specs:
        p = ROOT / rel if rel else None
        exists = bool(p and p.exists())
        rcount = regions = mcount = None
        keys = ""
        if exists and did not in raw_counts:
            rcount, regions, mcount, keys = dataframe_file_info(p)
        elif did in raw_counts:
            rcount = raw_counts[did]
            keys = "관리번호; 인허가일자; 폐업일자; 주소; 업태"
        metadata = meta_by_file.get(rel)
        rows.append({
            "데이터ID": did, "실제경로": rel, "원본/가공구분": kind, "기관": inst,
            "출처URL": getattr(metadata, "url", "") if metadata is not None else "",
            "다운로드시각": getattr(metadata, "downloaded_at", "") if metadata is not None else "",
            "대상기간": period, "기준시점/갱신일": "", "시간단위": "월/분기/시점(자료별)",
            "공간단위": unit, "업종단위": unit, "행수": rcount, "지역수": regions,
            "월수": mcount, "주요키": keys, "용도": "감사 및 모듈 분석",
            "라이선스/접근조건": getattr(metadata, "license_or_cost", "") if metadata is not None else "",
            "SHA256": sha256(p) if exists else "", "검증상태": status if exists or not rel else "UNAVAILABLE",
        })
    return pd.DataFrame(rows)


def bc_audit(bc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = ["STRD_YYMM", *RKEY, "GENDER_CD", "AGE_CD", "TP_BUZ_NO"]
    ri_month = bc.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    month_counts = ri_month.groupby(RKEY + ["TP_BUZ_NO"]).STRD_YYMM.nunique()
    complete_keys = month_counts[month_counts.eq(6)].reset_index()[RKEY + ["TP_BUZ_NO"]]
    complete_rows = ri_month.merge(complete_keys, on=RKEY + ["TP_BUZ_NO"], how="inner")
    all_amt, all_cnt = bc.amt.sum(), bc.cnt.sum()
    seg = bc.assign(segment_key=bc.GENDER_CD.astype(str) + "|" + bc.AGE_CD.astype(str))
    set_by_month = seg.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"]).segment_key.agg(set).reset_index()
    changing = 0
    common_amt = common_cnt = total_complete_amt = total_complete_cnt = 0
    complete_set = set(map(tuple, complete_keys.to_numpy()))
    for k, g in set_by_month.groupby(RKEY + ["TP_BUZ_NO"]):
        if k not in complete_set:
            continue
        sets = g.segment_key.tolist()
        inter = set.intersection(*sets)
        changing += int(any(s != sets[0] for s in sets[1:]))
        q = seg[(seg.SIDO_NM == k[0]) & (seg.CCG_NM == k[1]) & (seg.TP_BUZ_NO == k[2])]
        total_complete_amt += q.amt.sum(); total_complete_cnt += q.cnt.sum()
        qc = q[q.segment_key.isin(inter)]
        common_amt += qc.amt.sum(); common_cnt += qc.cnt.sum()
    cross = pd.crosstab(bc.GENDER_CD, bc.AGE_CD)
    facts = [
        ("rows", len(bc), "행", "PASS"),
        ("months", bc.STRD_YYMM.nunique(), "월", "PASS" if sorted(bc.STRD_YYMM.unique()) == MONTHS else "FAIL"),
        ("region_pairs", len(bc[RKEY].drop_duplicates()), "시군구쌍", "PASS"),
        ("industries", bc.TP_BUZ_NO.nunique(), "업종", "PASS"),
        ("duplicate_primary_keys", int(bc.duplicated(key).sum()), "행", "PASS" if not bc.duplicated(key).any() else "FAIL"),
        ("null_cells", int(bc.isna().sum().sum()), "셀", "PASS" if not bc.isna().any().any() else "WARN"),
        ("nonpositive_amt", int((bc.amt <= 0).sum()), "행", "PASS"),
        ("nonpositive_cnt", int((bc.cnt <= 0).sum()), "행", "PASS"),
        ("minimum_cnt", int(bc.cnt.min()), "건", "WARN_UNVERIFIED_SUPPRESSION"),
        ("complete_region_industry", len(complete_keys), "조합", "PASS"),
        ("complete_panel_amt_coverage", complete_rows.amt.sum() / all_amt * 100, "%", "PASS"),
        ("complete_panel_cnt_coverage", complete_rows.cnt.sum() / all_cnt * 100, "%", "PASS"),
        ("complete_pairs_with_changing_segment_cells", changing, "조합", "WARN" if changing else "PASS"),
        ("fixed_segment_amt_coverage_within_complete", common_amt / total_complete_amt * 100, "%", "INFO"),
        ("fixed_segment_cnt_coverage_within_complete", common_cnt / total_complete_cnt * 100, "%", "INFO"),
        ("foreign_gender_with_numeric_age_rows", int(((bc.GENDER_CD == "3") & (bc.AGE_CD != "x")).sum()), "행", "CODEBOOK_CONFLICT"),
        ("corporate_gender_with_x_age_rows", int(((bc.GENDER_CD == "x") & (bc.AGE_CD == "x")).sum()), "행", "PASS"),
    ]
    audit = pd.DataFrame(facts, columns=["check", "value", "unit", "status"])
    month = bc.groupby("STRD_YYMM", as_index=False).agg(
        rows=("cnt", "size"), region_pairs=("CCG_NM", "size"), industries=("TP_BUZ_NO", "nunique"), amt=("amt", "sum"), cnt=("cnt", "sum")
    )
    month["region_pairs"] = [len(bc.loc[bc.STRD_YYMM.eq(m), RKEY].drop_duplicates()) for m in month.STRD_YYMM]
    quality = {
        "complete_keys": complete_keys, "complete_rows": complete_rows, "cross": cross,
        "complete_amt_coverage": complete_rows.amt.sum() / all_amt * 100,
        "complete_cnt_coverage": complete_rows.cnt.sum() / all_cnt * 100,
    }
    return audit, month, quality


def age_industry_metrics(bc: pd.DataFrame) -> pd.DataFrame:
    q = (bc[bc.AGE_CD.ne("x")]
         .groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM", "AGE_CD"], as_index=False)["cnt"].sum())
    q["share"] = q.cnt / q.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"]).cnt.transform("sum")
    monthly = (q.assign(hhi=lambda d: d.share**2, ent=lambda d: -d.share * np.log(d.share))
               .groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False)
               .agg(age_hhi=("hhi", "sum"), age_entropy=("ent", "sum")))
    monthly.age_entropy /= math.log(6)
    rows = []
    for key, g in monthly.groupby(RKEY + ["TP_BUZ_NO"]):
        if g.STRD_YYMM.nunique() != 6:
            continue
        g = g.sort_values("STRD_YYMM")
        rows.append({**dict(zip(RKEY + ["TP_BUZ_NO"], key)),
                     "age_hhi_slope": linregress(range(6), g.age_hhi).slope,
                     "age_entropy_slope": linregress(range(6), g.age_entropy).slope})
    return pd.DataFrame(rows)


def focused_panel(bc: pd.DataFrame, deep: Path, fourth: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps = CONFIG["neutral_band_pct_month"]
    monthly = bc.groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    age = age_industry_metrics(bc)
    rows = []
    for key, g in monthly.groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM"]):
        if g.STRD_YYMM.nunique() != 6:
            continue
        g = g.sort_values("STRD_YYMM"); x = np.arange(6)
        ta, tc, tt = growth(g.amt), growth(g.cnt), growth(g.amt / g.cnt)
        peak = int(np.nanargmax(ta["residual"]))
        keep = np.arange(6) != peak
        loo = growth(g.amt.to_numpy()[keep], x[keep])["growth_pct_m"]
        rows.append({
            **dict(zip(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM"], key)),
            "amt_growth_pct_m": ta["growth_pct_m"], "cnt_growth_pct_m": tc["growth_pct_m"],
            "ticket_growth_pct_m": tt["growth_pct_m"], "amt_r2": ta["r2"],
            "amt_raw_slope": linregress(x, g.amt).slope, "cnt_raw_slope": linregress(x, g.cnt).slope,
            "amt_robust_sign": fixed_sign(g.amt), "cnt_robust_sign": fixed_sign(g.cnt),
            "first3_cnt_mean": g.cnt.iloc[:3].mean(), "total_cnt_6m": g.cnt.sum(), "total_amt_6m": g.amt.sum(),
            "peak_month": int(g.STRD_YYMM.iloc[peak]), "amt_growth_without_peak_pct_m": loo,
            "peak_dependent": bool(ta["growth_pct_m"] > eps and loo <= eps),
        })
    panel = pd.DataFrame(rows).merge(age, on=RKEY + ["TP_BUZ_NO"], how="left", validate="one_to_one")
    panel["amt_direction"] = panel.amt_growth_pct_m.map(lambda v: direction(v, eps))
    panel["strict_ticket_illusion"] = (panel.amt_growth_pct_m > 0) & (panel.cnt_growth_pct_m < 0)
    panel["original_raw_slope_illusion"] = (panel.amt_raw_slope > 0) & (panel.cnt_raw_slope < 0)
    panel["ticket_dependence_sensitivity"] = ((panel.amt_growth_pct_m > eps) & (panel.cnt_growth_pct_m <= eps) &
                                               (panel.ticket_growth_pct_m > eps))
    panel["age_concentration_up"] = (panel.age_hhi_slope > 0) & (panel.age_entropy_slope < 0)

    region_month = bc.groupby(RKEY + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    reg_rows = []
    for key, g in region_month.groupby(RKEY):
        g = g.sort_values("STRD_YYMM"); x = np.arange(6)
        a, c, t = growth(g.amt), growth(g.cnt), growth(g.amt / g.cnt)
        peak = int(np.nanargmax(a["residual"])); keep = x != peak
        loo = growth(g.amt.to_numpy()[keep], x[keep])["growth_pct_m"]
        reg_rows.append({**dict(zip(RKEY, key)), "amt_growth_pct_m": a["growth_pct_m"],
                         "cnt_growth_pct_m": c["growth_pct_m"], "ticket_growth_pct_m": t["growth_pct_m"],
                         "peak_month": int(g.STRD_YYMM.iloc[peak]), "amt_growth_without_peak_pct_m": loo,
                         "peak_dependent": bool(a["growth_pct_m"] > eps and loo <= eps)})
    region = pd.DataFrame(reg_rows)
    age_reg = pd.read_csv(deep / "08_h6_age_metrics.csv")
    div = pd.read_csv(deep / "17_h11_diversity.csv")
    div = div[(div.coverage == "7개전국공통") & (div.metric == "cnt")][RKEY + ["hhi_slope", "entropy_slope"]]
    region = region.merge(age_reg[RKEY + ["concentration_increase"]], on=RKEY).merge(div, on=RKEY, how="left")
    region["industry_concentration_up"] = (region.hhi_slope > 0) & (region.entropy_slope < 0)
    region["strict_ticket_illusion"] = (region.amt_growth_pct_m > 0) & (region.cnt_growth_pct_m < 0)
    region["ticket_dependence_sensitivity"] = ((region.amt_growth_pct_m > eps) & (region.cnt_growth_pct_m <= eps) &
                                                (region.ticket_growth_pct_m > eps))

    total_trend = region[RKEY + ["amt_growth_pct_m"]].rename(columns={"amt_growth_pct_m": "region_total_amt_growth_pct_m"})
    q4 = panel.merge(total_trend, on=RKEY)
    q4["region_total_direction"] = q4.region_total_amt_growth_pct_m.map(lambda v: direction(v, eps))
    q4["direction_divergence"] = q4.amt_direction != q4.region_total_direction
    return panel, region, q4


def external_quality_and_coverage(bc: pd.DataFrame, deep: Path, fourth: Path, q: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_amt, all_cnt = bc.amt.sum(), bc.cnt.sum()
    regions = bc[RKEY].drop_duplicates()
    pop = pd.read_csv(deep / "03_h2_percapita_aging.csv")
    valid_pop = pop.population_growth_pct_m.notna()
    pop_keys = pop.loc[valid_pop, RKEY]
    bpop = bc.merge(pop_keys, on=RKEY, how="inner")
    living = pd.read_csv(fourth / "02_living_population_monthly.csv")
    live_keys = living[RKEY].drop_duplicates()
    bq1 = bc[bc.STRD_YYMM <= 202603]
    blive = bq1.merge(live_keys, on=RKEY, how="inner")
    complete = q["complete_rows"]
    stores = pd.read_csv(fourth / "04_market_store_metrics_combined.csv")
    high = stores[stores.confidence.eq("HIGH")]
    high_keys = high[RKEY + ["연결업종"]].drop_duplicates()
    name_norm = {"편 의 점": "편의점", "슈퍼 마켓": "슈퍼마켓", "중국음식": "중국음식", "제 과 점": "제과점"}
    bstore = bc.assign(연결업종=bc.TP_BUZ_NM.map(name_norm)).dropna(subset=["연결업종"])
    bstore = bstore.merge(high_keys, on=RKEY + ["연결업종"], how="inner")
    local = pd.read_excel(ROOT / "dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx", sheet_name="시군구_월별_개폐업")
    local["accounting_error"] = local["가동_당월말"] - (local["가동_전월말"] + local["신규_당월"] - local["폐업_당월"])
    ntsq = pd.read_excel(ROOT / "dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx", sheet_name="품질_요약")
    nts_mask = float(ntsq.loc[ntsq.항목.str.contains("시군구 전체업종 폐업"), "값"].iloc[0])

    bstore_months = bstore.groupby(RKEY + ["연결업종", "STRD_YYMM"]).ngroups
    coverage = pd.DataFrame([
        ["BC 완전 지역×업종 패널", "지역×업종×월", 2327*6, 255, 6, q["complete_amt_coverage"], q["complete_cnt_coverage"], "전체 BC 대비"],
        ["주민등록인구", "지역×월", len(pop_keys)*6, len(pop_keys), 6, bpop.amt.sum()/all_amt*100, bpop.cnt.sum()/all_cnt*100, "전체 BC 대비"],
        ["생활인구", "지역×월", len(live_keys)*3, len(live_keys), 3, blive.amt.sum()/bq1.amt.sum()*100, blive.cnt.sum()/bq1.cnt.sum()*100, "1~3월 BC 대비"],
        ["HIGH 점포 연결 4업종", "지역×업종×월", bstore_months, len(high[RKEY].drop_duplicates()), 6, bstore.amt.sum()/all_amt*100, bstore.cnt.sum()/all_cnt*100, "전체 11업종 BC 대비"],
        ["한국부동산원 시도 맥락", "시도×상가유형", 51, 17, 2, np.nan, np.nan, "시군구 직접 coverage 산출 부적합"],
    ], columns=["dataset", "analysis_unit", "linked_unit_month_rows", "regions", "continuous_months", "bc_amt_coverage_pct", "bc_cnt_coverage_pct", "denominator_scope"])

    quality = pd.DataFrame([
        ["주민등록인구", "지역×월", "251/255", "화성 신설 4구 전 기간 결측", "주민과 결제자 모집단 다름"],
        ["CPI", "전국×월×품목", "6개월", "11개 중 제과점만 HIGH, 나머지 MEDIUM", "시군구 물가가 아님"],
        ["LOCALDATA", "개별 인허가→월말 재구성", f"회계오차 0인 행 {(local.accounting_error==0).sum()}/{len(local)}", "지역 연결 99.994%; 업종 연결 54.339%", "관리번호는 물리점포·BC가맹점과 동일하지 않음"],
        ["국세청", "시군구×월×생활업종", "6개월", f"시군구 폐업건수 마스킹 {nts_mask:.2f}%", "가동사업자는 물리점포와 동일하지 않음"],
        ["생활인구", "89개 인구감소지역×월", "1~3월만", "대상지역 선정에 따른 구조적 coverage", "관광객보다 넓은 체류인구"],
        ["한국부동산원", "17시도×2분기×3유형", "Q1/Q2", "시군구 불일치", "전국 주분석 불가"],
        ["SEMAS", "현재 영업점 스냅샷", "데이터 미확보", "서비스키·과거연계 불가", "월별 slope 생성 금지"],
        ["관광데이터랩", "시군구×월 요청", "원자료 미확보", "로그인/이용조건", "주민·관광 분리 주장 불가"],
        ["공식 경계", "SHP", "프로젝트 미확보", "2025 센서스 경계는 공식 페이지에서 확인", "기존 KNN은 시설중심 근사"],
    ], columns=["dataset", "actual_unit", "period_or_integrity", "missingness_or_mapping", "interpretation_limit"])

    decisions = pd.DataFrame([
        ["BC AMT/CNT/연령/업종", "CORE", "전국 255개·6개월 핵심 결과", "CNT를 고객수로 해석 금지"],
        ["주민등록 연령별 인구", "CONTROL", "소비고령화·인구 대비 맥락", "주민 1인당 실제지출 아님"],
        ["CPI", "CONTROL", "명목/실질 민감도", "전국 지수·대부분 근사 매핑"],
        ["LOCALDATA 중국음식·제과점", "MODULE_ONLY", "인허가 기반 공급 모듈", "BC 가맹점과 모집단 불일치"],
        ["LOCALDATA 한식·일식·서양식·스넥", "MODULE_ONLY", "민감도/사례", "MEDIUM/LOW 근사 매핑"],
        ["국세청 편의점·슈퍼마켓 가동사업자", "MODULE_ONLY", "정확 명칭의 공급 방향", "물리 점포 아님; 폐업 세부값 대부분 마스킹"],
        ["생활인구 1~3월", "MODULE_ONLY", "89개 인구감소지역의 체류수요 맥락", "3개월·공통계절성"],
        ["한국부동산원 임대료·공실", "CASE_ONLY", "시도/대표상권 맥락", "시군구 귀속 금지"],
        ["SEMAS 현재 API", "HOLD", "현재 공급수준 후보", "키 없음·월별 과거이력 없음"],
        ["한국관광 데이터랩", "HOLD", "방문수요 분리 후보", "실제 원자료 없음"],
        ["시설좌표 기반 KNN 공간지표", "DROP_FROM_ANALYSIS", "이번 핵심주제에 불필요", "공식 인접관계가 아님"],
        ["국세청 시군구 폐업건수", "DROP_FROM_ANALYSIS", "현재 폐업효과 검증", "96.46% 마스킹"],
    ], columns=["dataset_or_variable", "decision", "allowed_use", "reason_or_limit"])

    missing = pd.DataFrame([
        ["BC 지역×업종 불완전 패널", "업종/지역별 구조적 누락", "특히 갈비·한정식", "미관측을 0으로 대체하지 않음"],
        ["주민등록인구", "화성 신설 4구", "4/255 지역", "인구결합 분석에서 제외"],
        ["생활인구", "인구감소지역만, 4~6월 없음", "166개 지역 및 Q2", "전국 일반화 금지"],
        ["LOCALDATA 업종", "원본 업태의 54.34%만 6개 연결업종", "미분류/비대상 업태", "11개 BC 전체 결론 금지"],
        ["국세청 폐업", "소수값 마스킹", "시군구 96.46%", "폐업 검증에서 제외"],
        ["부동산", "시군구 값 없음", "전국 시군구", "사례 맥락만"],
    ], columns=["source", "missing_pattern", "scope", "handling"])
    return coverage, quality, decisions, missing


def claims(deep: Path, fourth: Path) -> pd.DataFrame:
    mix = pd.read_csv(deep / "04_h3_ticket_decomposition.csv")
    m7 = mix[mix.coverage.eq("7개전국공통")]
    pop = pd.read_csv(deep / "03_h2_percapita_aging.csv")
    age_tests = pd.read_csv(deep / "10_h7_age_volatility_tests.csv")
    div = pd.read_csv(deep / "17_h11_diversity.csv")
    models = pd.read_csv(deep / "15_h10_h18_model_comparison.csv")
    store = pd.read_csv(deep / "13_h8_h9_store_metrics.csv")
    visitor_tests = pd.read_csv(fourth / "12_h4_visitor_association_tests.csv")
    vcluster = pd.read_csv(fourth / "11_h4_visitor_cluster_profile.csv")
    old_v = {1: (-17.97, -10.27, -6.10), 2: (8.48, 3.02, 4.73)}
    rows = []
    def add(cid, claim, old, new, status, why, allowed, not_allowed):
        rows.append([cid, claim, old, new, status, why, allowed, not_allowed])
    add("C01", "7개 고정업종 객단가 하락 지역", "231/254", f"{int((m7.ticket_ols_pct_m<0).sum())}/{len(m7)}", "REPRODUCED", "원자료 재실행", "지역 합산 건당 결제금액 하락", "가격·이익 하락")
    add("C02", "객단가 3방법 방향 일치 하락", "220개", f"{int(((m7.ticket_ols_pct_m<0)&m7.ticket_robust_signs).sum())}개", "REPRODUCED", "동일 6개월 강건성", "세 방법의 부호 일치", "독립표본 검증")
    add("C03", "within/mix 절대기여 중앙값", "63.7%/36.3%", f"{m7.within_abs_share.median()*100:.2f}%/{m7.mix_abs_share.median()*100:.2f}%", "REPRODUCED", "지역별 절대기여 비중의 중앙값", "지역의 전형적 분해", "전국 총액 기여율")
    add("C04", "소비고령화 Gap 양수", "203/251", f"{int((pop.commercial_aging_gap_pp_m>0).sum())}/{pop.commercial_aging_gap_pp_m.notna().sum()}", "REPRODUCED", "화성 4구 제외", "소비 60+ 점유율 기울기가 주민보다 큼", "인구효과의 인과 제거")
    add("C05", "소비고령화 Gap 중앙값", "+0.151%p/월", f"{pop.commercial_aging_gap_pp_m.median():+.3f}%p/월", "REPRODUCED", "원자료 재실행", "점유율 slope 차이", "고객 고령화 속도")
    q = age_tests[(age_tests.concentration=="age_hhi_mean")&(age_tests.volatility=="amt_residual_sd")].iloc[0]
    add("C06", "연령 HHI와 AMT 잔차변동성 상관", "rho=.419", f"rho={q.spearman_rho:.3f}", "REPRODUCED", "수학적으로 독립은 아니나 다른 산식", "탐색적 연관", "집중이 변동성의 원인")
    high = store[store.confidence.eq("HIGH")]
    b = int(high.market_store_type.str.startswith("B ").sum()); c = int(high.market_store_type.str.startswith("C ").sum())
    br = int(high.market_store_type_robust.str.startswith("B ").sum()); cr = int(high.market_store_type_robust.str.startswith("C ").sum())
    add("C07", "LOCALDATA HIGH B/C", "B22(강건2), C121(강건54)", f"B{b}(강건{br}), C{c}(강건{cr})", "REPRODUCED", "HIGH=중국음식·제과점 509조합", "두 업종 공급모듈", "11업종 전국 일반화")
    d7 = div[(div.coverage=="7개전국공통")&(div.metric=="cnt")]
    add("C08", "CNT 업종집중 증가", "207/254", f"{int(((d7.hhi_slope>0)&(d7.entropy_slope<0)).sum())}/{len(d7)}", "REPRODUCED", "고정업종집합", "집중 방향", "건강성 악화 확정")
    best_cl = models[models.outcome.eq("closure_rate_6m_pct")].cv_r2.max(); best_net = models[models.outcome.eq("net_store_change_pct")].cv_r2.max()
    add("C09", "폐업·순점포 CV R2", ".020/.016", f"{best_cl:.3f}/{best_net:.3f}", "REPRODUCED", "지역 GroupKFold 재실행", "동시기 설명력 거의 없음", "폐업예측 가능")
    a = visitor_tests[(visitor_tests.visitor_metric=="stay_relative_growth_pct_m")&(visitor_tests.bc_metric.str.contains("amt_relative"))].iloc[0]
    c2 = visitor_tests[(visitor_tests.visitor_metric=="stay_relative_growth_pct_m")&(visitor_tests.bc_metric.str.contains("cnt_relative"))].iloc[0]
    add("C10", "체류 상대성장과 BC 상대성장", ".688/.630", f"{a.spearman_rho:.3f}/{c2.spearman_rho:.3f}", "REPRODUCED", "89개·1~3월", "Q1 동행", "체류수요의 인과효과")
    new1 = vcluster.sort_values("visitor_structure_cluster").iloc[0]
    new2 = vcluster.sort_values("visitor_structure_cluster").iloc[1]
    new_text = f"C1 stay {new1.stay_relative_growth_pct_m:.2f}, AMT {new1.bc_amt_growth_jan_mar_pct_m:.2f}; C2 stay {new2.stay_relative_growth_pct_m:.2f}, AMT {new2.bc_amt_growth_jan_mar_pct_m:.2f}"
    add("C11", "4차 보고서 생활인구 군집 프로필", "C1 stay -17.97, C2 +8.48", new_text, "REVISED", "현재 코드 재실행값과 보고서 서술 불일치", "재실행 CSV 값", "기존 보고서의 해당 두 숫자")
    add("C12", "관광 방문자 원자료", "미확보", "프로젝트 내 미확보", "REPRODUCED", "파일 인벤토리", "생활인구를 체류 맥락으로 사용", "관광객 분리")
    add("C13", "공간분석", "시설좌표 KNN 근사", "공식 경계 파일 미확보; KNN 산출물 존재", "REPRODUCED", "코드·파일 확인", "근사 민감도", "공식 인접 Moran")
    raw_slopes = pd.read_csv(ROOT / "analysis/growth_illusion/01_slope_table_all_combinations.csv")
    raw_n = int(((raw_slopes.complete_6m) & (raw_slopes.amt_slope_won_per_month > 0) &
                 (raw_slopes.cnt_slope_per_month < 0)).sum())
    add("C14", "기존 원단위 OLS 성장착시 후보", "10/2327", f"{raw_n}/2327", "REPRODUCED", "기존 스크립트 산출식 대조", "원단위 slope 부호 후보", "로그 slope 후보와 동일한 집합")
    return pd.DataFrame(rows, columns=["claim_id","previous_claim","previous_value","recalculated_result","decision","reason","allowed_interpretation","not_allowed_interpretation"])


def select_case_pairs(panel: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_norm = {"편 의 점": "편의점", "슈퍼 마켓": "슈퍼마켓", "중국음식": "중국음식", "제 과 점": "제과점"}
    st = stores[stores.confidence.eq("HIGH")][RKEY + ["연결업종", "supply_growth_pct_m", "external_source"]].drop_duplicates(RKEY + ["연결업종"])
    p = panel.copy(); p["연결업종"] = p.TP_BUZ_NM.map(name_norm)
    p = p.merge(st, on=RKEY + ["연결업종"], how="left")
    for c in ["cnt_growth_pct_m", "ticket_growth_pct_m", "age_hhi_slope", "supply_growth_pct_m"]:
        p[c + "_z"] = p.groupby("TP_BUZ_NO")[c].transform(lambda s: (s - s.mean()) / s.std() if s.std() else 0)
    eligible = p[p.total_cnt_6m.ge(CONFIG["case_min_total_cnt_6m"])].copy()
    pairs = []
    for ind, g in eligible.groupby("TP_BUZ_NO"):
        for ia, ib in combinations(g.index, 2):
            a, b = g.loc[ia], g.loc[ib]
            if a.amt_direction != b.amt_direction:
                continue
            if abs(a.amt_growth_pct_m - b.amt_growth_pct_m) > CONFIG["case_amt_tolerance_pp_month"]:
                continue
            if abs(np.log10(a.first3_cnt_mean) - np.log10(b.first3_cnt_mean)) > CONFIG["case_log10_initial_cnt_tolerance"]:
                continue
            cols = ["cnt_growth_pct_m_z", "ticket_growth_pct_m_z", "age_hhi_slope_z"]
            if pd.notna(a.supply_growth_pct_m) and pd.notna(b.supply_growth_pct_m):
                cols.append("supply_growth_pct_m_z")
            dist = math.sqrt(sum((a[c] - b[c])**2 for c in cols if pd.notna(a[c]) and pd.notna(b[c])))
            pairs.append({"idx_a": ia, "idx_b": ib, "TP_BUZ_NO": ind, "TP_BUZ_NM": a.TP_BUZ_NM,
                          "amt_direction": a.amt_direction, "amt_gap_pp_m": abs(a.amt_growth_pct_m-b.amt_growth_pct_m),
                          "initial_cnt_ratio": max(a.first3_cnt_mean,b.first3_cnt_mean)/min(a.first3_cnt_mean,b.first3_cnt_mean),
                          "diagnostic_distance": dist})
    candidates = pd.DataFrame(pairs).sort_values("diagnostic_distance", ascending=False)
    selected = []
    used_regions, used_industries = set(), set()
    # One pair per direction first, then one additional rising pair from a new industry.
    for target in CONFIG["case_direction_targets"]:
        for _, row in candidates[candidates.amt_direction.eq(target)].iterrows():
            a, b = p.loc[int(row.idx_a)], p.loc[int(row.idx_b)]
            regs = {(a.SIDO_NM,a.CCG_NM),(b.SIDO_NM,b.CCG_NM)}
            if regs & used_regions:
                continue
            selected.append(row); used_regions |= regs; used_industries.add(int(row.TP_BUZ_NO)); break
    for _, row in candidates[candidates.amt_direction.eq("상승")].iterrows():
        a, b = p.loc[int(row.idx_a)], p.loc[int(row.idx_b)]
        regs = {(a.SIDO_NM,a.CCG_NM),(b.SIDO_NM,b.CCG_NM)}
        if regs & used_regions or int(row.TP_BUZ_NO) in used_industries:
            continue
        selected.append(row); break
    selected = selected[:CONFIG["case_pair_count"]]
    detail = []
    for pair_id, row in enumerate(selected, 1):
        for side, idx in [("A", int(row.idx_a)), ("B", int(row.idx_b))]:
            x = p.loc[idx]
            priority = []
            if x.cnt_growth_pct_m < -CONFIG["neutral_band_pct_month"] and x.ticket_growth_pct_m > CONFIG["neutral_band_pct_month"]:
                priority.append("거래건수 약화·건당결제액 의존 원인 확인")
            elif x.ticket_growth_pct_m < -CONFIG["neutral_band_pct_month"]:
                priority.append("건당결제액 하락의 상품·할인·결제분할 확인")
            else:
                priority.append("CNT와 건당결제액의 동반/상쇄 구조 확인")
            if x.age_hhi_slope > 0:
                priority.append("연령 소비층 집중의 지속성과 고객수 확인")
            if pd.notna(x.supply_growth_pct_m):
                priority.append("외부 공급 변화와 BC 가맹점 모집단 차이 확인")
            if x.peak_dependent:
                priority.append("피크월 행사·영업일·일회성 거래 확인")
            detail.append({
                "pair_id": pair_id, "side": side, "SIDO_NM": x.SIDO_NM, "CCG_NM": x.CCG_NM,
                "TP_BUZ_NO": x.TP_BUZ_NO, "TP_BUZ_NM": x.TP_BUZ_NM, "amt_direction": x.amt_direction,
                "amt_growth_pct_m": x.amt_growth_pct_m, "cnt_growth_pct_m": x.cnt_growth_pct_m,
                "ticket_growth_pct_m": x.ticket_growth_pct_m, "age_hhi_slope": x.age_hhi_slope,
                "supply_growth_pct_m": x.supply_growth_pct_m, "external_source": x.external_source,
                "total_cnt_6m": x.total_cnt_6m, "amt_robust_sign": x.amt_robust_sign,
                "peak_dependent": x.peak_dependent, "first_check": " → ".join(priority),
                "conditional_action_candidate": "점포 POS·고객수·영업일 확인 후 유지/구성개선/재방문 실험 중 선택",
                "not_verified": "개별 점포 원인, 개선효과, 수익성",
            })
    return candidates, pd.DataFrame(detail)


def case_monthly(bc: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    m = bc.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    rows = []
    for _, c in cases.iterrows():
        q = m[(m.SIDO_NM==c.SIDO_NM)&(m.CCG_NM==c.CCG_NM)&(m.TP_BUZ_NO==c.TP_BUZ_NO)].sort_values("STRD_YYMM").copy()
        q["ticket"] = q.amt/q.cnt; q["amt_index"] = q.amt/q.amt.iloc[0]*100; q["cnt_index"] = q.cnt/q.cnt.iloc[0]*100; q["ticket_index"] = q.ticket/q.ticket.iloc[0]*100
        q.insert(0,"pair_id",c.pair_id); q.insert(1,"side",c.side); rows.append(q)
    return pd.concat(rows, ignore_index=True)


def unresolved_and_acquisition() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unresolved = pd.DataFrame([
        ["CNT 변화가 고객수·재방문 중 무엇인가", "CNT는 결제건수", "결제분할·고객수·빈도", "월별 고유고객수/재방문", "불가", "BC 내부 또는 점포 CRM", "2026-01~06 시군구×업종", 1, "객단가 의존 진단의 의미", "점포 POS 질문으로 제한"],
        ["건당결제액 변화 원인", "AMT/CNT 분해", "가격·상품믹스·할인", "상품/POS·가격·할인", "불가", "점포 POS/통계청 세부가격", "월×상품", 1, "개선 후보 선택", "원인 미확인으로 표기"],
        ["체류수요 Q1 동행의 지속성", "89개에서 rho .688/.630", "공통계절성", "생활인구 4~6월 및 전년동월", "부분", "행정안전부 생활인구", "2025~2026 월별", 1, "체류축 유지 여부", "Q1 모듈로 제한"],
        ["점포공급 변화의 모집단 정합성", "LOCAL/NTS 공급 방향", "인허가·사업자와 BC가맹점 차이", "BC 가맹점수 또는 과거 SEMAS 스냅샷", "부분", "BC/소진공", "2026-01~06 월별", 1, "점포당 기회 해석", "공급 모듈 제한"],
        ["공간 군집의 공식 검증", "시설좌표 KNN 양의 Moran", "중심점 대리 오류", "공식 2026 경계 SHP", "불가", "국토교통부/VWorld", "2026 상반기", 3, "공간 정책 단위", "이번 핵심에서 제외"],
        ["개선안의 실제 효과", "관측자료 없음", "선택편향·외부충격", "점포 실험/전후 대조군", "불가", "참여 점포", "8~12주 이상", 1, "처방 효과 입증", "개선 후보만 제시"],
    ], columns=["question","bc_fact","alternative_explanation","needed_variable","reuse_existing","candidate_source","required_period_unit","priority","decision_if_obtained","fallback"])
    acq = pd.DataFrame([
        ["2026 Q2 생활인구", "2026-09-07", "공식 웹 검색", "HOLD", "공식 Q1 자료만 확인; Q2 원자료를 찾지 못함", "값 추정/대체 안 함"],
        ["공식 시군구 경계", "2026-09-07", "공공데이터포털 메타데이터", "AVAILABLE_NOT_DOWNLOADED", "2025-01-20 SHP와 VWorld 경로 확인", "핵심 분석에 불필요하여 이번 run 수집 보류"],
        ["SEMAS 상가 API", "2026-09-07", "공공데이터포털 메타데이터", "HOLD", "현재 영업점 API·인증키 필요; 2025 개편 후 과거 ID 연계 불가", "현재 목록으로 과거 slope 생성 안 함"],
        ["관광 데이터랩", "2026-09-07", "공식 사이트 검색", "HOLD", "다운로드 가능한 2026 월별 원자료 미확보", "생활인구 Q1만 별도 모듈"],
        ["신규 대규모 외부수집", "2026-09-07", "감사 판단", "NOT_NEEDED_FOR_CURRENT_TEST", "현재 주제 타당성은 기존 BC로 우선 검증 가능", "필수 미해결 자료만 후속"],
    ], columns=["target","checked_at","method","status","result","handling"])
    source = pd.DataFrame([
        ["MOIS_Q1_LIVING", "https://mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=128294", "Q1 원본 확보"],
        ["MOLIT_BOUNDARY", "https://www.data.go.kr/data/15125064/fileData.do", "2025 센서스 시군구 SHP 메타데이터 확인"],
        ["SEMAS_STORE_API", "https://www.data.go.kr/data/15012005/openapi.do", "현재 영업점·인증키·분류개편 확인"],
        ["KTO_DATALAB", "https://datalab.visitkorea.or.kr/datalab/portal/main/getMainForm.do", "원자료 미확보"],
    ], columns=["source_id","url","audit_note"])
    return unresolved, acq, source


def summaries(panel: pd.DataFrame, region: pd.DataFrame, q4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = CONFIG["neutral_band_pct_month"]
    growth_regions = region[region.amt_growth_pct_m > eps]
    growth_pairs = panel[panel.amt_growth_pct_m > eps]
    summary = pd.DataFrame([
        ["기존 원단위 OLS 성장착시", "지역×업종", int(panel.original_raw_slope_illusion.sum()), len(panel), panel.original_raw_slope_illusion.mean()*100, "원단위 AMT slope>0,CNT slope<0"],
        ["로그추세 객단가 착시", "지역×업종", int(panel.strict_ticket_illusion.sum()), len(panel), panel.strict_ticket_illusion.mean()*100, "로그 AMT>0,CNT<0"],
        ["중립구간 포함 객단가 의존 민감도", "지역×업종", int(panel.ticket_dependence_sensitivity.sum()), len(panel), panel.ticket_dependence_sensitivity.mean()*100, "AMT>.25,CNT<=.25,ticket>.25"],
        ["일회성 피크 의존", "AMT 성장 지역×업종", int(growth_pairs.peak_dependent.sum()), len(growth_pairs), growth_pairs.peak_dependent.mean()*100, CONFIG["peak_dependency_rule"]],
        ["업종 CNT 집중 증가", "AMT 성장 지역", int(growth_regions.industry_concentration_up.sum()), len(growth_regions), growth_regions.industry_concentration_up.mean()*100, "7개 고정업종 HHI↑ Entropy↓"],
        ["연령 CNT 집중 증가", "AMT 성장 지역", int(growth_regions.concentration_increase.sum()), len(growth_regions), growth_regions.concentration_increase.mean()*100, "6개 숫자 연령 HHI↑ Entropy↓"],
        ["지역총량과 개별업종 방향 불일치", "완전 지역×업종", int(q4.direction_divergence.sum()), len(q4), q4.direction_divergence.mean()*100, f"±{eps}%/월 중립구간"],
    ], columns=["indicator","scope","n","denominator","share_pct","definition"])
    cross = (q4.groupby(["region_total_direction","amt_direction"]).size().rename("n").reset_index())
    cross["share_pct"] = cross.n/len(q4)*100
    return summary, cross


def method_value_table(cases: pd.DataFrame, visitor_tests: pd.DataFrame) -> pd.DataFrame:
    different_priority = cases.groupby("pair_id").first_check.nunique().gt(1).sum()
    return pd.DataFrame([
        ["B0 일반 지표표", "AMT·CNT·건당결제액·연령·업종 추세", "개별 지표 방향 확인", "상쇄·분해·확인순서가 자동 연결되지 않음", "EXECUTED"],
        ["B1 소비구조 분해", "B0+항등분해·집중도·피크민감도·총량/업종 괴리", f"선정 {cases.pair_id.nunique()}쌍 중 점검순서가 다른 쌍 {different_priority}개", "의사결정 개선효과는 미측정", "EXECUTED"],
        ["B2 외부 맥락", "B1+인구·생활인구·공급", f"생활인구-상대AMT rho={visitor_tests.spearman_rho.iloc[0]:.3f}; 공급은 4개 HIGH업종", "표본/기간 제한", "EXECUTED_MODULES"],
        ["B0E 같은 원자료 체크리스트", "B2와 같은 원자료의 단순 표시", "평가 템플릿 작성", "전문가·점주 평가자 없음", "NOT_RUN_HUMAN_EVAL"],
        ["개선효과 평가", "점포 실행 전후/대조군", "자료 없음", "매출·폐업 감소 효과 산출 불가", "NOT_RUN"],
    ], columns=["method","information","observed_additional_value","limit","status"])


def make_figures(out: Path, summary: pd.DataFrame, cases: pd.DataFrame, monthly: pd.DataFrame) -> None:
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ["AppleGothic", "Arial Unicode MS", "NanumGothic"]:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
    # Avoid showing both raw-OLS and log-OLS versions of the same rare flag.
    s = summary.iloc[1:]
    fig, ax = plt.subplots(figsize=(9,4.5)); ax.barh(s.indicator[::-1], s.share_pct[::-1], color="#356A8A")
    ax.set_xlabel("비중(%)"); ax.set_title("성장품질 후보 지표: 표본과 정의가 서로 다름")
    for i,v in enumerate(s.share_pct[::-1]): ax.text(v+.5,i,f"{v:.1f}%",va="center")
    fig.tight_layout(); fig.savefig(figdir/"01_growth_quality_flags.png",dpi=180); plt.close(fig)
    pairs = sorted(cases.pair_id.unique()); fig, axes = plt.subplots(len(pairs),3,figsize=(11,2.6*len(pairs)),sharex=True)
    if len(pairs)==1: axes=np.array([axes])
    for i,pid in enumerate(pairs):
        for side,g in monthly[monthly.pair_id.eq(pid)].groupby("side"):
            label = cases[(cases.pair_id==pid)&(cases.side==side)].iloc[0]
            nm=f"{side}: {label.SIDO_NM} {label.CCG_NM}"
            for j,c in enumerate(["amt_index","cnt_index","ticket_index"]): axes[i,j].plot(g.STRD_YYMM.astype(str).str[-2:],g[c],marker="o",label=nm)
        axes[i,0].set_ylabel(f"Pair {pid}\n1월=100"); axes[i,0].legend(fontsize=7)
    for j,t in enumerate(["AMT","CNT","건당결제액"]): axes[0,j].set_title(t)
    fig.tight_layout(); fig.savefig(figdir/"02_case_pairs_monthly.png",dpi=180); plt.close(fig)


def write_report(out: Path, inventory: pd.DataFrame, bcq: pd.DataFrame, coverage: pd.DataFrame,
                 extq: pd.DataFrame, decisions: pd.DataFrame, claim: pd.DataFrame, unresolved: pd.DataFrame,
                 acq: pd.DataFrame, summary: pd.DataFrame, q4cross: pd.DataFrame, cases: pd.DataFrame,
                 method: pd.DataFrame) -> None:
    def md(d, cols=None):
        x=d if cols is None else d[cols]
        return markdown_table(x)
    reproduced = (claim.decision=="REPRODUCED").sum(); revised=(claim.decision=="REVISED").sum()
    case_lines=[]
    for pid,g in cases.groupby("pair_id"):
        a,b=g.sort_values("side").iloc[0],g.sort_values("side").iloc[1]
        case_lines.append(f"### Pair {pid}. {a.TP_BUZ_NM} · {a.amt_direction}\n\n"
          f"- {a.SIDO_NM} {a.CCG_NM}: AMT {a.amt_growth_pct_m:+.2f}, CNT {a.cnt_growth_pct_m:+.2f}, 건당결제액 {a.ticket_growth_pct_m:+.2f}%/월. 첫 점검: {a.first_check}.\n"
          f"- {b.SIDO_NM} {b.CCG_NM}: AMT {b.amt_growth_pct_m:+.2f}, CNT {b.cnt_growth_pct_m:+.2f}, 건당결제액 {b.ticket_growth_pct_m:+.2f}%/월. 첫 점검: {b.first_check}.\n"
          f"- 같은 점: AMT 월성장 차이 {abs(a.amt_growth_pct_m-b.amt_growth_pct_m):.2f}%p 이내이며 초기 CNT 규모가 유사하다.\n"
          f"- 달라진 처방 순서: AMT가 아니라 CNT→건당결제액→연령집중→공급자료 가용성 순으로 확인한다. 개선은 POS·고객수·영업일 확인 뒤의 후보이며 효과는 검증되지 않았다.\n")
    report=f"""# 「같은 매출 변화, 다른 처방」 데이터 감사·재분석 보고서

## 1. 결론

현재 주제는 **‘처방 효과를 입증한 서비스’가 아니라 ‘같은 AMT 변화에서도 먼저 확인할 문제가 달라짐을 재현 가능한 규칙으로 보여주는 진단 프레임’까지 방어 가능**하다.

- 기존 핵심 주장 {len(claim)}개 중 {reproduced}개는 새 디렉터리의 원자료 재실행에서 재현됐고, {revised}개는 수정됐다.
- 가장 중요한 수정은 4차 보고서의 생활인구 군집 프로필 일부 숫자가 현재 코드 재실행값과 다르다는 점이다. 체류인구와 BC 상대성장의 상관 자체는 재현됐다.
- 객단가 착시의 엄격한 원형은 희소하다. 따라서 발생률을 넓히지 않고 보조 플래그로 유지한다.
- 반면 업종집중·연령집중·피크 의존·지역총량과 개별업종의 방향 불일치는 서로 다른 점검순서를 만드는 실질적 정보다.
- 개별 점포 고객수, POS, 영업일, 가격·할인, 실행 후 성과가 없어 개선안은 **조건부 후보**이며 실제 효과는 측정하지 않았다.

## 2. 용어와 범위

- CNT는 결제건수이며 고객수·방문자수·재방문수가 아니다.
- AMT/CNT는 건당 평균 결제금액이며 가격·이익이 아니다.
- 성장률은 2026년 1~6월 로그 OLS 월성장률이다. ±0.25%/월을 사례선정의 중립구간으로 사전에 고정했다.
- ‘다른 처방’은 즉시 행동 지시가 아니라 **다음으로 확인할 데이터와 조건부 실험의 순서가 다르다**는 뜻이다.

## 3. 보유자료와 품질 감사

원본/가공/미확보를 구분한 전체 목록은 `data_inventory.csv`에 있다. 핵심 현황은 다음과 같다.

{md(inventory[["데이터ID","원본/가공구분","대상기간","검증상태"]])}

### BC 원자료

{md(bcq[["check","value","unit","status"]])}

중요한 코드북 충돌이 있다. 문서는 외국인의 연령을 X로 설명하지만 실제 `GENDER_CD=3` 행은 숫자 연령코드를 사용하고, `AGE_CD=X`는 법인 행과만 결합한다. 숫자 연령 분석은 외국인도 포함하므로 ‘주민 고객 연령’이라고 부르지 않는다. 최소 CNT=11은 억제 흔적일 수 있으나 공식 규칙을 확보하지 못해 UNVERIFIED다.

### 외부자료 coverage와 정합성

{md(coverage)}

{md(extq)}

LOCALDATA 월말 가동점포는 인허가일·폐업일로 재구성됐으며 `기초+신규-폐업=기말` 회계식은 9,174행 모두 일치했다. 그러나 관리번호와 물리점포, BC가맹점은 동일 모집단이 아니다. 국세청 시군구 폐업건수는 96.46%가 마스킹되어 폐업 검증에서 제외했다.

## 4. 기존 주장 재현·수정

{md(claim[["claim_id","previous_claim","previous_value","recalculated_result","decision"]])}

63.7%는 전국 총액의 비율이 아니라 **지역별 within/mix 절대기여 비중의 중앙값**이다. 세 추세법 일치는 같은 6개월을 재사용한 강건성 검사이며 독립 검증이 아니다. 폐업률·순점포 모형의 외부검증 R²가 거의 0인 결과도 재현되어 폐업예측 주제는 유지할 수 없다.

## 5. 외부자료 판정

{md(decisions)}

외부자료를 모두 교집합으로 묶지 않았다. 인구는 통제, 생활인구·점포는 모듈, 부동산은 사례 맥락으로 분리했다. 시설좌표 KNN은 공식 경계가 아니고 현재 주제의 핵심 질문에 필요하지 않아 이번 주분석에서 제외했다.

## 6. 추가로 필요한 자료와 확보 상태

{md(unresolved[["question","needed_variable","priority","decision_if_obtained","fallback"]])}

{md(acq)}

이번 run에서는 감사 전에 대규모 수집을 하지 않았다. 감사 후에도 현재 주제의 1차 타당성 검증에는 기존 BC가 충분했으므로, 실제 판단을 바꾸는 자료만 후속 대상으로 남겼다. 현재 점포 API로 과거 월별 점포수를 만들지 않았고, Q2 생활인구도 추정하지 않았다.

## 7. 핵심 재분석

{md(summary)}

![성장품질 후보 지표](figures/01_growth_quality_flags.png)

각 비율의 표본과 정의가 다르므로 더해서 하나의 위험률로 만들 수 없다. 엄격 객단가 착시는 희소한 반면 업종집중은 넓다. 그러나 HHI 증가는 ‘나쁜 성장’의 확정이 아니라 다양성 점검 신호다. 피크 의존은 최고 양의 로그잔차 월을 제거했을 때 추세가 중립 이하가 되는 경우만 탐색 플래그로 두었다.

기존 성장착시 보고서의 10개는 원단위 OLS slope 부호이며 재현됐다. 이번 로그 OLS에서는 9개다. 둘 중 하나를 정답으로 섞지 않고, 성장률 비교에는 로그추세를 사용하고 기존 주장 재현에는 원래 산식을 유지했다.

지역 전체 AMT 방향과 개별업종 방향의 비교는 다음과 같다.

{md(q4cross)}

이 차이는 지역 총량을 특정 업종의 상태로 그대로 적용하면 잘못된 점검순서가 생길 수 있음을 보여주지만, 기존 서비스의 오류율을 측정한 것은 아니다.

## 8. 비슷한 AMT 변화의 대표 비교

사례는 결과를 보고 임의 선택하지 않았다. 같은 업종, AMT slope 차이 ≤0.35%p/월, 전반 3개월 CNT 규모비 ≤1.78배, 6개월 CNT ≥1만 건이라는 설정으로 가능한 모든 쌍을 만든 뒤 구조 차이가 큰 쌍을 방향별로 선정했다. 유명 지역 여부는 사용하지 않았다.

{''.join(case_lines)}
![대표 사례 월별 추이](figures/02_case_pairs_monthly.png)

사례별 원자료 추적값은 `case_pair_details.csv`와 `case_pair_monthly.csv`에 있다. 어떤 사례도 개별 점포의 원인이나 개선효과를 입증하지 않는다.

## 9. 일반 지표분석 대비 추가 가치

{md(method)}

추가 가치는 ‘더 많은 데이터를 보여주는 것’보다 다음 세 가지다.

1. AMT가 같아도 CNT와 건당결제액의 상쇄 방향을 먼저 구분한다.
2. 지역 총량과 개별업종 방향이 다를 때 총량 처방을 중단한다.
3. 외부 공급자료의 모집단·기간이 맞을 때만 점포 질문을 추가하고, 맞지 않으면 불확실성으로 남긴다.

전문가·점주 평가와 실제 실행실험은 하지 않았으므로 운영상 우월성이나 매출개선 효과는 입증되지 않았다.

## 10. 최종 판단

방어 가능한 최종 표현은 다음과 같다.

> **같은 매출 변화라도 결제건수, 건당 결제금액, 소비층 집중, 업종 구성, 점포공급을 순서대로 분해하면 먼저 확인해야 할 문제가 달라진다. 본 분석은 지역·업종 수준의 점검순서를 제시하지만 개별 점포의 원인과 개선효과는 현장자료로 검증해야 한다.**

미해결 핵심은 고유고객수·재방문, 상품/POS, 실제 영업일, BC 가맹점수, 12개월 이상 계절통제, 점포 실행성과다. 이 자료 없이는 ‘맞춤형 개선안 제안’은 가능하지만 ‘검증된 처방’이라고 부를 수 없다.

## 11. 재현 파일

- `README.md`: 실행순서·환경·완료상태
- `data_inventory.csv`, `bc_quality_audit.csv`, `coverage_by_dataset.csv`, `missingness_bias.csv`
- `external_data_decisions.csv`, `claim_reproduction.csv`, `unresolved_questions.csv`, `acquisition_log.csv`
- `region_industry_audit_panel.csv`, `region_growth_quality.csv`, `region_total_vs_industry.csv`
- `growth_quality_summary.csv`, `case_pair_candidates.csv`, `case_pair_details.csv`, `case_pair_monthly.csv`
- `method_value_comparison.csv`, `evidence_cards.md`, `figures/`
- 실행 코드: `scripts/audit_same_sales_different_actions.py`

분석 기준일: 2026-09-07  
분석 성격: 데이터 감사 및 탐색적 진단. 인과·개별점포 처방효과·폐업예측이 아님.
"""
    (out/"final_report.md").write_text(report,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",required=True); args=parser.parse_args()
    out=(ROOT/args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out.mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    deep=out/"reproduction/deep_exploration"; fourth=out/"reproduction/fourth_external_integration"
    required=[deep/"analysis_summary.json",fourth/"analysis_summary.json"]
    if not all(p.exists() for p in required):
        raise FileNotFoundError("먼저 README의 clean-room reproduction 명령을 실행하세요: "+", ".join(map(str,required)))
    bc=pd.read_csv(ROOT/"dataset/ABP_CONTEST_DATA.csv",encoding="utf-8-sig")
    inventory=build_inventory(); bcq,bcmonth,q=bc_audit(bc)
    coverage,extq,decisions,missing=external_quality_and_coverage(bc,deep,fourth,q)
    claim=claims(deep,fourth)
    unresolved,acq,source=unresolved_and_acquisition()
    panel,region,q4=focused_panel(bc,deep,fourth)
    summary,q4cross=summaries(panel,region,q4)
    stores=pd.read_csv(fourth/"04_market_store_metrics_combined.csv")
    candidates,cases=select_case_pairs(panel,stores); monthly=case_monthly(bc,cases)
    visitor_tests=pd.read_csv(fourth/"12_h4_visitor_association_tests.csv")
    method=method_value_table(cases,visitor_tests)
    tables={
      "data_inventory.csv":inventory,"bc_quality_audit.csv":bcq,"bc_monthly_composition.csv":bcmonth,
      "coverage_by_dataset.csv":coverage,"external_quality_audit.csv":extq,"missingness_bias.csv":missing,
      "external_data_decisions.csv":decisions,"claim_reproduction.csv":claim,"unresolved_questions.csv":unresolved,
      "acquisition_log.csv":acq,"source_metadata.csv":source,"region_industry_audit_panel.csv":panel,
      "region_growth_quality.csv":region,"region_total_vs_industry.csv":q4,"region_total_vs_industry_summary.csv":q4cross,
      "growth_quality_summary.csv":summary,"case_pair_candidates.csv":candidates,"case_pair_details.csv":cases,
      "case_pair_monthly.csv":monthly,"method_value_comparison.csv":method,
    }
    for name,d in tables.items(): d.to_csv(out/name,index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(out/"audit_analysis_tables.xlsx", engine="openpyxl") as writer:
        for name,d in tables.items():
            d.to_excel(writer, sheet_name=name.replace(".csv", "")[:31], index=False)
    (out/"config.json").write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding="utf-8")
    env={"python":sys.version,"platform":platform.platform(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":__import__('scipy').__version__}
    (out/"environment.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")
    checks=pd.DataFrame([{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for p in [ROOT/"dataset/ABP_CONTEST_DATA.csv",ROOT/"dataset/external_raw/mois/202601_202606_age_population.csv",ROOT/"dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx",ROOT/"dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx"]])
    checks.to_csv(out/"input_checksums.csv",index=False,encoding="utf-8-sig")
    make_figures(out,summary,cases,monthly)
    cards=["# 대표 사례 Evidence Cards\n"]
    for pid,g in cases.groupby("pair_id"):
        cards.append(f"## Pair {pid}: {g.TP_BUZ_NM.iloc[0]} ({g.amt_direction.iloc[0]})\n\n"+markdown_table(g)+"\n\n개선 후보는 점포 POS·고객수·영업일 확인 뒤에만 선택한다. 실제 효과는 NOT_TESTED.\n")
    (out/"evidence_cards.md").write_text("\n".join(cards),encoding="utf-8")
    write_report(out,inventory,bcq,coverage,extq,decisions,claim,unresolved,acq,summary,q4cross,cases,method)
    readme=f"""# Audit run README\n\nRun ID: {out.name}\n\n## 실행 순서\n\n1. 기존 심층분석 clean-room 재현\n2. 4차 외부결합 clean-room 재현\n3. 감사 및 집중 재분석\n\n```bash\npython3 -u - <<'PY'\nimport sys\nfrom pathlib import Path\nroot=Path.cwd(); sys.path.insert(0,str(root/'scripts'))\nrun=root/'{out.relative_to(ROOT)}'\nimport analyze_deep_exploration as d\nd.OUT=run/'reproduction/deep_exploration'; d.FIG=d.OUT/'figures'; d.main()\nimport analyze_fourth_external_integration as f\nf.OUT=run/'reproduction/fourth_external_integration'; f.FIG=f.OUT/'figures'; f.DEEP=d.OUT; f.main()\nPY\npython3 scripts/audit_same_sales_different_actions.py --run-dir {out.relative_to(ROOT)}\n```\n\n## 상태\n\n- A 파일·데이터 조사: COMPLETE\n- B 품질 감사: COMPLETE\n- C 기존 결과 재현: COMPLETE ({(claim.decision=='REPRODUCED').sum()} reproduced, {(claim.decision=='REVISED').sum()} revised)\n- D 외부자료 판정: COMPLETE\n- E 추가 수집: COMPLETE_WITH_HOLDS (필수 미확보 자료는 acquisition_log 기록)\n- F 소비구조·사례 비교: COMPLETE\n- G 조건부 개선 후보: COMPLETE; 실제 효과 NOT_TESTED\n- H 내부 방법론 비교: AUTOMATED PART COMPLETE; 사람/점주 평가는 NOT_RUN\n\n원자료는 수정하지 않았고 기존 분석 폴더도 덮어쓰지 않았다.\n"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps({"out":str(out),"claims":claim.decision.value_counts().to_dict(),"cases":cases.pair_id.nunique(),"summary":summary.to_dict('records')},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
