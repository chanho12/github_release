#!/usr/bin/env python3
"""Exploratory weather check for BC monthly peaks using the acquired KMA package.

This is deliberately a limitation test, not a causal model. It uses only regions
with a conservative ASOS-to-BC mapping and keeps six-month/small-sample warnings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RKEY = ["SIDO_NM", "CCG_NM"]
METRICS = ["avg_temp_c", "precipitation_total_mm", "sunshine_hours", "days_precip_ge_1mm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis/external_data_acquisition/run_20260907")
    args = ap.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    bc = pd.read_csv(ROOT / "dataset/ABP_CONTEST_DATA.csv")
    bc["TP_BUZ_NM_CLEAN"] = bc.TP_BUZ_NM.str.replace(" ", "", regex=False)
    monthly = (bc.groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "STRD_YYMM"], as_index=False)
               [["amt", "cnt"]].sum())
    national = (monthly.groupby(["TP_BUZ_NO", "STRD_YYMM"], as_index=False).amt.sum()
                .rename(columns={"amt": "national_amt"}))
    monthly = monthly.merge(national, on=["TP_BUZ_NO", "STRD_YYMM"], validate="many_to_one")
    monthly["log_amt_share"] = np.log(monthly.amt / monthly.national_amt)

    weather = pd.read_csv(ROOT / "dataset/외부데이터/기상청_기상/202601_202606_ASOS_BC지역_연결패널.csv")
    # Multiple stations in one municipality are averaged; no interpolation to uncovered regions.
    region_weather = weather.groupby(RKEY + ["STRD_YYMM"], as_index=False)[METRICS].mean()
    region_weather.to_csv(out / "weather_region_month_conservative.csv", index=False, encoding="utf-8-sig")
    z = monthly.merge(region_weather, on=RKEY + ["STRD_YYMM"], how="inner", validate="many_to_one")

    core_path = ROOT / "analysis/final_topic_validation/run_20260907/core_region_industry_metrics.csv"
    core = pd.read_csv(core_path)[RKEY + ["TP_BUZ_NO", "peak_dependent", "total_cnt_6m"]]
    z = z.merge(core, on=RKEY + ["TP_BUZ_NO"], how="left", validate="many_to_one")

    corr_rows, peak_rows = [], []
    keys = RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"]
    for key, g in z.groupby(keys):
        g = g.sort_values("STRD_YYMM")
        if g.STRD_YYMM.nunique() != 6:
            continue
        base = {**dict(zip(keys, key)), "total_cnt_6m": float(g.total_cnt_6m.iloc[0]),
                "peak_dependent": bool(g.peak_dependent.iloc[0]),
                "relative_amt_peak_month": int(g.loc[g.log_amt_share.idxmax(), "STRD_YYMM"])}
        for metric in METRICS:
            valid = g[["log_amt_share", metric]].dropna()
            rho, p = spearmanr(valid.log_amt_share, valid[metric]) if len(valid) == 6 else (np.nan, np.nan)
            corr_rows.append({**base, "weather_metric": metric, "n_months": len(valid),
                              "spearman_rho": rho, "p_value_unadjusted": p})
            peak_rows.append({**base, "weather_metric": metric,
                              "weather_max_month": int(g.loc[g[metric].idxmax(), "STRD_YYMM"]),
                              "same_peak_month": int(g.loc[g.log_amt_share.idxmax(), "STRD_YYMM"]) ==
                                                 int(g.loc[g[metric].idxmax(), "STRD_YYMM"])})
    corrs = pd.DataFrame(corr_rows); peaks = pd.DataFrame(peak_rows)
    corrs.to_csv(out / "weather_spike_combo_correlations.csv", index=False, encoding="utf-8-sig")
    peaks.to_csv(out / "weather_peak_concurrence.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for min_cnt in [0, 100_000]:
        for only_peak in [False, True]:
            for metric in METRICS:
                q = corrs[(corrs.weather_metric == metric) & (corrs.total_cnt_6m >= min_cnt)]
                p = peaks[(peaks.weather_metric == metric) & (peaks.total_cnt_6m >= min_cnt)]
                if only_peak:
                    q = q[q.peak_dependent]; p = p[p.peak_dependent]
                summaries.append({"min_total_cnt_6m": min_cnt, "peak_dependent_only": only_peak,
                                  "weather_metric": metric, "combos": len(q),
                                  "regions": q[RKEY].drop_duplicates().shape[0],
                                  "median_rho": q.spearman_rho.median(),
                                  "abs_rho_ge_0_7_pct": q.spearman_rho.abs().ge(.7).mean() * 100 if len(q) else np.nan,
                                  "positive_rho_pct": q.spearman_rho.gt(0).mean() * 100 if len(q) else np.nan,
                                  "same_peak_month_pct": p.same_peak_month.mean() * 100 if len(p) else np.nan})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "weather_spike_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
