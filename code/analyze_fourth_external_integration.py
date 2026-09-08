#!/usr/bin/env python3
"""Fourth exploration: BC demand versus stores, visitors, rent and vacancy.

This script intentionally separates national analyses from partial-coverage case
validation.  Missing external observations are never filled with zero.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.gettempdir()) / "bc_fourth_external"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP / "xdg"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu, spearmanr, ttest_1samp
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_deep_exploration import localdata_integrate
from analyze_growth_illusion import configure_plot
from analyze_third_exploration import RKEY, load_bc, parse_population, trend, h7_h9_business


OUT = ROOT / "analysis/fourth_external_integration"
FIG = OUT / "figures"
LIVING = ROOT / "dataset/external_raw/mois_living_population/202601_202603_living_population_by_region.csv"
REB = ROOT / "dataset/external_raw/reb_commercial/2026Q1_Q2_commercial_rent_vacancy_province.csv"
MAPPING = ROOT / "dataset/external_raw/semas/bc_store_mapping.csv"
DEEP = ROOT / "analysis/deep_exploration"


def fdr(df: pd.DataFrame, pcol: str = "p") -> pd.DataFrame:
    out = df.copy().sort_values(pcol)
    n = len(out)
    if not n:
        out["p_fdr"] = []
        return out
    vals = out[pcol].to_numpy() * n / np.arange(1, n + 1)
    out["p_fdr"] = np.minimum.accumulate(vals[::-1])[::-1].clip(0, 1)
    return out


def rank_biserial(a, b) -> float:
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if not len(a) or not len(b):
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def trend3(y) -> float:
    y = np.asarray(y, float)
    if len(y) != 3 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return np.nan
    return np.expm1(linregress(np.arange(3), np.log(y)).slope) * 100


def prepare_store_panel(bc, population_month):
    """Use the best executable monthly source per matched external category."""
    local, local_month = localdata_integrate(bc, population_month)
    nts, _, _, _, _ = h7_h9_business(bc)

    # Physical LOCALDATA is retained for food industries; NTS adds the two exact
    # retail categories that LOCALDATA food permits cannot cover.
    l = local.copy()
    l["external_source"] = "LOCALDATA"
    l["source_scope"] = "physical_food_establishment"

    n = nts[nts["연결업종"].isin(["편의점", "슈퍼마켓"])].copy()
    rename = {
        "amt_robust": "amt_robust_signs",
        "cnt_robust": "cnt_robust_signs",
        "avg_ticket_robust": "ticket_robust_signs",
        "가동_당월_growth_pct_m": "가동_당월말_growth_pct_m",
        "가동_당월_robust": "가동_당월말_robust_signs",
        "amt_per_active_business_growth_pct_m": "amt_per_store_growth_pct_m",
        "amt_per_active_business_robust": "amt_per_store_robust_signs",
        "cnt_per_active_business_growth_pct_m": "cnt_per_store_growth_pct_m",
        "cnt_per_active_business_robust": "cnt_per_store_robust_signs",
    }
    n = n.rename(columns=rename)
    n["confidence"] = "HIGH"
    n["external_source"] = "NTS"
    n["source_scope"] = "active_business_taxpayer"
    n["total_cnt_6m"] = np.nan
    # Obtain BC six-month volume for threshold sensitivity.
    code = {"편의점": 4010, "슈퍼마켓": 4020}
    vol = (bc[bc.TP_BUZ_NO.isin(code.values())]
           .groupby(RKEY + ["TP_BUZ_NO"], as_index=False).cnt.sum()
           .assign(연결업종=lambda q: q.TP_BUZ_NO.map({v: k for k, v in code.items()}))
           [RKEY + ["연결업종", "cnt"]].rename(columns={"cnt": "_volume"}))
    n = n.merge(vol, on=RKEY + ["연결업종"], how="left")
    n["total_cnt_6m"] = n["_volume"]
    n["initial_stores"] = np.nan
    n["openings_6m"] = np.nan
    n["closures_6m"] = np.nan
    n["opening_rate_6m_pct"] = np.nan
    n["closure_rate_6m_pct"] = np.nan
    n["net_store_change_pct"] = np.nan
    n["real_amt_growth_pct_m"] = np.nan
    n["real_amt_per_store_growth_pct_m"] = np.nan

    common = sorted(set(l.columns) | set(n.columns))
    stores = pd.concat([l.reindex(columns=common), n.reindex(columns=common)], ignore_index=True)
    stores["supply_growth_pct_m"] = stores["가동_당월말_growth_pct_m"]
    stores["demand_supply_gap_amt"] = stores.amt_growth_pct_m - stores.supply_growth_pct_m
    stores["demand_supply_gap_cnt"] = stores.cnt_growth_pct_m - stores.supply_growth_pct_m
    stores["competition_pressure_cnt"] = stores.supply_growth_pct_m - stores.cnt_growth_pct_m
    stores["competition_pressure_real_amt"] = stores.supply_growth_pct_m - stores.real_amt_growth_pct_m

    stores["h1_type"] = np.select(
        [
            (stores.amt_growth_pct_m >= 0) & (stores.supply_growth_pct_m >= 0) & (stores.amt_per_store_growth_pct_m >= 0),
            (stores.amt_growth_pct_m >= 0) & (stores.supply_growth_pct_m >= 0) & (stores.amt_per_store_growth_pct_m < 0),
            (stores.amt_growth_pct_m < 0) & (stores.supply_growth_pct_m >= 0),
            (stores.amt_growth_pct_m < 0) & (stores.supply_growth_pct_m < 0),
            (stores.amt_growth_pct_m >= 0) & (stores.supply_growth_pct_m < 0),
        ],
        ["A 시장·공급·점포당기회 동반성장", "B 시장성장·공급과속·점포당기회감소",
         "C 수요감소·공급증가", "D 수요·공급 동반축소", "E 시장성장·공급감소"],
        default="판정불가",
    )
    robust_cols = ["amt_robust_signs", "가동_당월말_robust_signs", "amt_per_store_robust_signs"]
    stores["h1_three_method_robust"] = stores[robust_cols].fillna(0).astype(bool).all(axis=1)

    # Density uses mean six-month resident population and beginning store count.
    mean_pop = population_month.groupby(RKEY, as_index=False).population.mean()
    stores = stores.merge(mean_pop, on=RKEY, how="left")
    stores["stores_per_10000"] = stores.initial_stores / stores.population * 10000
    for col in ["competition_pressure_cnt", "competition_pressure_real_amt", "stores_per_10000"]:
        stores[col + "_industry_pct"] = stores.groupby("연결업종")[col].rank(pct=True) * 100
        stores[col + "_vs_industry_median"] = stores[col] - stores.groupby("연결업종")[col].transform("median")
    return stores, local_month


def h1_h3_tables(stores):
    summary = []
    for conf in ["HIGH", "HIGH+MEDIUM"]:
        base = stores[stores.confidence.eq("HIGH")] if conf == "HIGH" else stores[stores.confidence.isin(["HIGH", "MEDIUM"])]
        for threshold in [0, 1_000, 10_000, 100_000, 500_000]:
            q = base[(base.total_cnt_6m.fillna(0) >= threshold) & base.h1_type.ne("판정불가")]
            for robust in [False, True]:
                z = q[q.h1_three_method_robust] if robust else q
                for typ, n in z.h1_type.value_counts().items():
                    summary.append({"confidence": conf, "cnt_threshold": threshold, "robust_only": robust,
                                    "h1_type": typ, "n": int(n), "denominator": len(z),
                                    "share_pct": n / len(z) * 100 if len(z) else np.nan,
                                    "region_n": z.loc[z.h1_type.eq(typ), RKEY].drop_duplicates().shape[0],
                                    "industry_n": z.loc[z.h1_type.eq(typ), "연결업종"].nunique()})
    h1 = pd.DataFrame(summary)

    q = stores[stores.confidence.eq("HIGH")].dropna(subset=["demand_supply_gap_cnt"]).copy()
    cuts = q.demand_supply_gap_cnt.quantile([0, .1, .2, .8, .9, 1]).to_dict()
    q["gap_group"] = pd.cut(q.demand_supply_gap_cnt,
                            [-np.inf, cuts[.1], cuts[.2], cuts[.8], cuts[.9], np.inf],
                            labels=["하위10%", "10~20%", "중앙60%", "80~90%", "상위10%"], include_lowest=True)
    metrics = ["demand_supply_gap_amt", "demand_supply_gap_cnt", "amt_per_store_growth_pct_m",
               "cnt_per_store_growth_pct_m", "opening_rate_6m_pct", "closure_rate_6m_pct", "net_store_change_pct"]
    h2 = q.groupby("gap_group", observed=True)[metrics].agg(["count", "median"]).reset_index()
    h2.columns = ["gap_group"] + [f"{a}_{b}" for a, b in h2.columns.tolist()[1:]]
    tests = []
    for metric in metrics:
        a = q.loc[q.gap_group.eq("하위10%"), metric].dropna()
        b = q.loc[q.gap_group.eq("상위10%"), metric].dropna()
        if len(a) >= 5 and len(b) >= 5:
            u = mannwhitneyu(a, b, alternative="two-sided")
            tests.append({"metric": metric, "bottom_n": len(a), "top_n": len(b),
                          "bottom_median": a.median(), "top_median": b.median(),
                          "rank_biserial_bottom_vs_top": rank_biserial(a, b), "p": u.pvalue})
    h2_tests = fdr(pd.DataFrame(tests))

    h3 = stores[RKEY + ["연결업종", "confidence", "external_source", "competition_pressure_cnt",
                        "competition_pressure_cnt_industry_pct", "competition_pressure_cnt_vs_industry_median",
                        "competition_pressure_real_amt", "competition_pressure_real_amt_industry_pct",
                        "competition_pressure_real_amt_vs_industry_median", "stores_per_10000",
                        "stores_per_10000_industry_pct", "total_cnt_6m"]].copy()
    return h1, q, h2, h2_tests, h3


def visitor_analysis(bc, region_features):
    living = pd.read_csv(LIVING)
    living["stay_share_89"] = living.stay_population / living.groupby("STRD_YYMM").stay_population.transform("sum")
    bc3 = bc[bc.STRD_YYMM <= 202603].groupby(RKEY + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    bc3["amt_share_national"] = bc3.amt / bc3.groupby("STRD_YYMM").amt.transform("sum")
    bc3["cnt_share_national"] = bc3.cnt / bc3.groupby("STRD_YYMM").cnt.transform("sum")
    rows = []
    for key, g in living.groupby(RKEY):
        g = g.sort_values("STRD_YYMM")
        row = dict(zip(RKEY, key))
        for col in ["living_population", "resident_population_pdf", "stay_population", "registered_foreigner_population"]:
            row[col + "_growth_pct_m"] = trend3(g[col])
            row[col + "_mean"] = g[col].mean()
        row["stay_to_resident_ratio"] = (g.stay_population / g.resident_population_pdf).mean()
        row["stay_relative_growth_pct_m"] = trend3(g.stay_share_89)
        rows.append(row)
    visitor = pd.DataFrame(rows)

    bc_rows = []
    for key, g in bc3.groupby(RKEY):
        g = g.sort_values("STRD_YYMM")
        bc_rows.append({**dict(zip(RKEY, key)), "bc_amt_growth_jan_mar_pct_m": trend3(g.amt),
                        "bc_cnt_growth_jan_mar_pct_m": trend3(g.cnt),
                        "bc_amt_relative_growth_jan_mar_pct_m": trend3(g.amt_share_national),
                        "bc_cnt_relative_growth_jan_mar_pct_m": trend3(g.cnt_share_national)})
    visitor = visitor.merge(pd.DataFrame(bc_rows), on=RKEY, how="left")

    # BC foreigner share/contribution is a contextual signal, not a visitor count.
    seg = pd.read_csv(DEEP / "07_h4_segment_region.csv")
    foreign = seg[seg.segment.eq("외국인")][RKEY + ["amt_growth_pct_m", "amt_share_slope_pp_m"]].rename(
        columns={"amt_growth_pct_m": "foreigner_amt_growth_pct_m", "amt_share_slope_pp_m": "foreigner_amt_share_slope_pp_m"})
    visitor = visitor.merge(foreign, on=RKEY, how="left").merge(region_features, on=RKEY, how="left")
    visitor["admin_type"] = np.where(visitor.CCG_NM.str.endswith("군"), "군", "시·구")

    # Test whether standardized growth structures naturally separate.
    ccols = ["bc_amt_growth_jan_mar_pct_m", "bc_cnt_growth_jan_mar_pct_m",
             "resident_population_pdf_growth_pct_m", "stay_relative_growth_pct_m",
             "foreigner_amt_share_slope_pp_m"]
    x = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(visitor[ccols]))
    scores = []
    for k in range(2, 6):
        labels = KMeans(k, n_init=50, random_state=42).fit_predict(x)
        scores.append({"k": k, "silhouette": silhouette_score(x, labels)})
    score = pd.DataFrame(scores)
    best = int(score.loc[score.silhouette.idxmax(), "k"])
    visitor["visitor_structure_cluster"] = KMeans(best, n_init=100, random_state=42).fit_predict(x) + 1
    cluster = visitor.groupby("visitor_structure_cluster")[ccols + ["stay_to_resident_ratio"]].median()
    cluster.insert(0, "n", visitor.groupby("visitor_structure_cluster").size())
    cluster = cluster.reset_index()

    # H5: top versus bottom quartile in official stay-population growth.
    lo, hi = visitor.stay_relative_growth_pct_m.quantile([.25, .75])
    visitor["visitor_growth_group"] = np.select(
        [visitor.stay_relative_growth_pct_m <= lo, visitor.stay_relative_growth_pct_m >= hi],
        ["하위25%", "상위25%"], default="중간50%")
    compare_cols = ["amt_growth_pct_m", "cnt_growth_pct_m", "ticket_growth_pct_m",
                    "commercial_aging_gap_pp_m", "bc_age_hhi_slope", "industry_hhi_slope",
                    "cnt_js_first3_last3", "amt_residual_sd", "cnt_residual_sd",
                    "regional_perstore_amt_growth", "regional_perstore_cnt_growth"]
    tests = []
    for col in compare_cols:
        a = visitor.loc[visitor.visitor_growth_group.eq("상위25%"), col].dropna()
        b = visitor.loc[visitor.visitor_growth_group.eq("하위25%"), col].dropna()
        if len(a) >= 5 and len(b) >= 5:
            u = mannwhitneyu(a, b, alternative="two-sided")
            tests.append({"metric": col, "high_n": len(a), "low_n": len(b), "high_median": a.median(),
                          "low_median": b.median(), "rank_biserial": rank_biserial(a, b), "p": u.pvalue})
    assoc = []
    for xcol in ["stay_relative_growth_pct_m", "stay_to_resident_ratio"]:
        for ycol in ["bc_amt_relative_growth_jan_mar_pct_m", "bc_cnt_relative_growth_jan_mar_pct_m",
                     "foreigner_amt_share_slope_pp_m", "commercial_aging_gap_pp_m"]:
            q = visitor[[xcol, ycol]].dropna(); rho, p = spearmanr(q[xcol], q[ycol])
            assoc.append({"visitor_metric": xcol, "bc_metric": ycol, "n": len(q), "spearman_rho": rho, "p": p})
    panel = living.merge(bc3, on=RKEY + ["STRD_YYMM"], how="inner")
    for month_a, month_b in [(202601, 202602), (202602, 202603)]:
        wide = panel[panel.STRD_YYMM.isin([month_a, month_b])].pivot(
            index=RKEY, columns="STRD_YYMM", values=["stay_share_89", "amt_share_national", "cnt_share_national"])
        dx = np.log(wide[("stay_share_89", month_b)] / wide[("stay_share_89", month_a)])
        for metric in ["amt_share_national", "cnt_share_national"]:
            dy = np.log(wide[(metric, month_b)] / wide[(metric, month_a)])
            rho, p = spearmanr(dx, dy)
            assoc.append({"visitor_metric": f"stay_relative_change_{month_a}_{month_b}",
                          "bc_metric": f"{metric}_change_{month_a}_{month_b}", "n": len(wide),
                          "spearman_rho": rho, "p": p})
    return living, visitor, score, cluster, fdr(pd.DataFrame(assoc)), fdr(pd.DataFrame(tests))


def reb_analysis(bc, stores):
    reb = pd.read_csv(REB)
    prov_bc_month = bc.groupby(["SIDO_NM", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    rows = []
    for sido, g in prov_bc_month.groupby("SIDO_NM"):
        g = g.sort_values("STRD_YYMM")
        rows.append({"SIDO_NM": sido, "bc_amt_growth_pct_m": trend(g.amt)["ols_pct_m"],
                     "bc_cnt_growth_pct_m": trend(g.cnt)["ols_pct_m"]})
    prov = pd.DataFrame(rows)
    st = stores.groupby("SIDO_NM", as_index=False).agg(
        perstore_amt_growth_pct_m=("amt_per_store_growth_pct_m", "median"),
        demand_supply_gap_cnt=("demand_supply_gap_cnt", "median"),
        store_growth_pct_m=("supply_growth_pct_m", "median"))
    prov = prov.merge(st, on="SIDO_NM", how="left")

    wide = reb.pivot_table(index=["SIDO_NM", "property_type"], columns="metric",
                           values=["2026Q1", "2026Q2", "qoq_change", "qoq_change_pct"]).reset_index()
    wide.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in wide.columns]
    wide = wide.merge(prov, on="SIDO_NM", how="left")
    wide["rent_growth_pct"] = wide["qoq_change_pct_rent_1000won_m2"]
    wide["vacancy_change_pp"] = wide["qoq_change_vacancy_rate_pct"]
    wide["vacancy_q2_pct"] = wide["2026Q2_vacancy_rate_pct"]
    wide["h6_type"] = np.select([
        (wide.bc_amt_growth_pct_m >= 0) & (wide.perstore_amt_growth_pct_m >= 0) & (wide.rent_growth_pct >= 0),
        (wide.bc_amt_growth_pct_m >= 0) & (wide.perstore_amt_growth_pct_m < 0) & (wide.rent_growth_pct >= 0),
        (wide.bc_amt_growth_pct_m < 0) & (wide.rent_growth_pct >= 0),
        (wide.bc_amt_growth_pct_m < 0) & (wide.rent_growth_pct < 0),
        (wide.bc_amt_growth_pct_m >= 0) & (wide.rent_growth_pct < 0),
    ], ["A 소비·점포당소비·임대료 증가", "B 소비증가·점포당소비감소·임대료증가",
        "C 소비감소·임대료증가", "D 소비·임대료감소", "E 소비증가·임대료감소"], default="판정불가")
    wide["h7_amt_up_vacancy_up"] = (wide.bc_amt_growth_pct_m >= 0) & (wide.vacancy_change_pp > 0)

    tests = []
    for typ, g in wide.groupby("property_type"):
        for ext in ["rent_growth_pct", "vacancy_change_pp", "vacancy_q2_pct"]:
            for bc_col in ["bc_amt_growth_pct_m", "bc_cnt_growth_pct_m", "perstore_amt_growth_pct_m", "demand_supply_gap_cnt"]:
                q = g[[ext, bc_col]].dropna()
                rho, p = spearmanr(q[ext], q[bc_col])
                tests.append({"property_type": typ, "external_metric": ext, "bc_metric": bc_col,
                              "n": len(q), "spearman_rho": rho, "p": p})
    return wide, fdr(pd.DataFrame(tests))


def core_region_features(stores):
    pop = pd.read_csv(DEEP / "03_h2_percapita_aging.csv")
    agevol = pd.read_csv(DEEP / "09_h7_age_volatility_region.csv")
    div = pd.read_csv(DEEP / "17_h11_diversity.csv")
    div = div[(div.coverage.eq("7개전국공통")) & (div.metric.eq("cnt"))][RKEY + ["hhi_slope", "entropy_slope"]].rename(
        columns={"hhi_slope": "industry_hhi_slope", "entropy_slope": "industry_entropy_slope"})
    shift = pd.read_csv(DEEP / "19_h12_portfolio_shift.csv")
    shift = shift[shift.coverage.eq("7개전국공통")][RKEY + ["cnt_js_first3_last3"]]
    mix = pd.read_csv(DEEP / "04_h3_ticket_decomposition.csv")
    mix = mix[mix.coverage.eq("7개전국공통")][RKEY + ["within_industry_effect", "between_industry_mix_effect"]]
    st = stores.groupby(RKEY, as_index=False).agg(
        regional_store_growth=("supply_growth_pct_m", "median"),
        regional_perstore_amt_growth=("amt_per_store_growth_pct_m", "median"),
        regional_perstore_cnt_growth=("cnt_per_store_growth_pct_m", "median"),
        regional_demand_supply_gap=("demand_supply_gap_cnt", "median"),
        regional_competition_pressure=("competition_pressure_cnt", "median"),
        regional_closure_rate=("closure_rate_6m_pct", "median"))
    out = pop.merge(agevol, on=RKEY).merge(div, on=RKEY, how="left").merge(shift, on=RKEY, how="left").merge(mix, on=RKEY, how="left").merge(st, on=RKEY, how="left")
    out["ticket_growth_pct_m"] = (1 + out.amt_growth_pct_m / 100) / (1 + out.cnt_growth_pct_m / 100) * 100 - 100
    out["amt_fitted_change_6m_pct"] = np.expm1(np.log1p(out.amt_growth_pct_m / 100) * 5) * 100
    return out


def h8_similar_growth(features, visitor, reb):
    q = features[features.amt_fitted_change_6m_pct.between(5, 10)].copy()
    q = q.merge(visitor[RKEY + ["stay_relative_growth_pct_m", "stay_to_resident_ratio"]], on=RKEY, how="left")
    province_reb = reb.groupby("SIDO_NM", as_index=False).agg(rent_growth_pct=("rent_growth_pct", "median"),
                                                               vacancy_change_pp=("vacancy_change_pp", "median"))
    q = q.merge(province_reb, on="SIDO_NM", how="left")
    return q


def pca_clusters(features, visitor):
    visitor_cols = visitor[RKEY + ["stay_relative_growth_pct_m", "stay_to_resident_ratio"]]
    d = features.merge(visitor_cols, on=RKEY, how="left")
    core_cols = ["amt_growth_pct_m", "cnt_growth_pct_m", "ticket_growth_pct_m",
                 "within_industry_effect", "between_industry_mix_effect", "population_growth_pct_m",
                 "amt_pc_growth_pct_m", "cnt_pc_growth_pct_m", "commercial_aging_gap_pp_m",
                 "bc_age_hhi_slope", "regional_store_growth", "regional_perstore_amt_growth",
                 "regional_perstore_cnt_growth", "regional_demand_supply_gap", "industry_hhi_slope",
                 "cnt_js_first3_last3", "amt_residual_sd"]
    xraw = d[core_cols].replace([np.inf, -np.inf], np.nan)
    x = SimpleImputer(strategy="median").fit_transform(xraw)
    # Winsorize before scaling.
    x = pd.DataFrame(x, columns=core_cols)
    for c in core_cols:
        lo, hi = x[c].quantile([.01, .99]); x[c] = x[c].clip(lo, hi)
    z = StandardScaler().fit_transform(x)
    pca = PCA().fit(z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cumulative, .70) + 1)
    scores = pca.transform(z)[:, :ncomp]
    selection = []
    for k in range(2, 7):
        labels = AgglomerativeClustering(k, linkage="ward").fit_predict(scores)
        selection.append({"k": k, "silhouette": silhouette_score(scores, labels)})
    selection = pd.DataFrame(selection)
    best = int(selection.loc[selection.silhouette.idxmax(), "k"])
    labels = AgglomerativeClustering(best, linkage="ward").fit_predict(scores) + 1

    loadings = pd.DataFrame(pca.components_[:ncomp].T, index=core_cols,
                            columns=[f"PC{i+1}" for i in range(ncomp)]).reset_index(names="feature")
    score_df = d[RKEY].copy()
    for i in range(ncomp): score_df[f"PC{i+1}"] = scores[:, i]
    score_df["cluster"] = labels
    profile = d.assign(cluster=labels).groupby("cluster")[core_cols].median()
    profile.insert(0, "n", pd.Series(labels).value_counts().sort_index().values)
    profile = profile.reset_index()

    # Perturbation stability.
    rng = np.random.default_rng(42); aris = []
    for _ in range(20):
        noisy = scores + rng.normal(0, .05, scores.shape)
        alt = AgglomerativeClustering(best, linkage="ward").fit_predict(noisy) + 1
        aris.append(adjusted_rand_score(labels, alt))
    quality = {"n_components_70pct": ncomp, "explained_variance": float(cumulative[ncomp - 1]),
               "cluster_k": best, "silhouette": float(selection.silhouette.max()),
               "perturbation_ari_mean": float(np.mean(aris)), "perturbation_ari_min": float(np.min(aris))}
    return d, loadings, score_df, selection, profile, quality


def cv_r2(data, features, target, groups):
    q = data[features + [target, groups]].dropna(subset=[target, groups]).copy()
    x = q[features].replace([np.inf, -np.inf], np.nan)
    nsplit = min(5, q[groups].nunique())
    cv = GroupKFold(nsplit)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    pred = cross_val_predict(model, x, q[target], groups=q[groups], cv=cv)
    return len(q), r2_score(q[target], pred)


def negative_tests(stores, visitor, reb_tests):
    rows = []
    for conf in ["HIGH", "HIGH+MEDIUM"]:
        q = stores[stores.confidence.eq("HIGH")] if conf == "HIGH" else stores[stores.confidence.isin(["HIGH", "MEDIUM"])]
        q = q.dropna(subset=["amt_growth_pct_m", "amt_per_store_growth_pct_m"])
        same = np.sign(q.amt_growth_pct_m) == np.sign(q.amt_per_store_growth_pct_m)
        rows.append({"test": "N1", "scope": conf, "n": len(q), "metric": "same_sign_share_pct", "value": same.mean() * 100})

    visitor = visitor.copy(); visitor["SIDO_GROUP"] = visitor.SIDO_NM
    base = ["resident_population_pdf_growth_pct_m", "resident_population_pdf_mean"]
    plus = base + ["stay_relative_growth_pct_m", "stay_to_resident_ratio"]
    for target in ["bc_amt_relative_growth_jan_mar_pct_m", "bc_cnt_relative_growth_jan_mar_pct_m"]:
        n, a = cv_r2(visitor, base, target, "SIDO_GROUP")
        _, b = cv_r2(visitor, plus, target, "SIDO_GROUP")
        rows += [{"test": "N2", "scope": target, "n": n, "metric": "base_cv_r2", "value": a},
                 {"test": "N2", "scope": target, "n": n, "metric": "plus_visitor_cv_r2", "value": b},
                 {"test": "N2", "scope": target, "n": n, "metric": "delta_cv_r2", "value": b-a}]

    best = reb_tests.loc[reb_tests.spearman_rho.abs().idxmax()]
    rows.append({"test": "N3", "scope": f"{best.property_type}:{best.external_metric}:{best.bc_metric}",
                 "n": int(best.n), "metric": "max_abs_spearman_rho", "value": abs(best.spearman_rho),
                 "p_fdr": best.p_fdr})

    # Centered controls make the intercept the adjusted sample mean aging gap.
    controls = visitor[["commercial_aging_gap_pp_m", "stay_relative_growth_pct_m", "stay_to_resident_ratio",
                        "resident_population_pdf_growth_pct_m", "resident_population_pdf_mean", "admin_type"]].dropna().copy()
    controls["is_gun"] = controls.admin_type.eq("군").astype(float)
    cols = ["stay_relative_growth_pct_m", "stay_to_resident_ratio", "resident_population_pdf_growth_pct_m",
            "resident_population_pdf_mean", "is_gun"]
    X = controls[cols].copy(); X["resident_population_pdf_mean"] = np.log1p(X.resident_population_pdf_mean)
    X = (X - X.mean()) / X.std(ddof=0)
    X.insert(0, "intercept", 1.0)
    y = controls.commercial_aging_gap_pp_m.to_numpy()
    beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    residual = y - X.to_numpy() @ beta
    se = math.sqrt((residual @ residual) / (len(y) - X.shape[1]) * np.linalg.inv(X.to_numpy().T @ X.to_numpy())[0, 0])
    tval = beta[0] / se
    # Two-sided normal approximation is sufficient for this descriptive audit.
    p = math.erfc(abs(tval) / math.sqrt(2))
    rows.append({"test": "N4", "scope": "89개 인구감소지역", "n": len(y),
                 "metric": "adjusted_mean_aging_gap_pp_m", "value": beta[0], "p": p})
    return pd.DataFrame(rows)


def create_plots(stores, visitor, reb, h8, pca_scores, loadings):
    configure_plot(); FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"A 시장·공급·점포당기회 동반성장":"#4C78A8", "B 시장성장·공급과속·점포당기회감소":"#F58518",
              "C 수요감소·공급증가":"#E45756", "D 수요·공급 동반축소":"#72B7B2", "E 시장성장·공급감소":"#54A24B"}
    q = stores[stores.confidence.eq("HIGH")]
    for typ, g in q.groupby("h1_type"):
        ax.scatter(g.supply_growth_pct_m, g.amt_growth_pct_m, s=14, alpha=.6, label=typ, c=colors.get(typ, "gray"))
    ax.axhline(0, c="black", lw=.8); ax.axvline(0, c="black", lw=.8)
    ax.set(xlabel="점포 공급 성장률(%/월)", ylabel="BC AMT 성장률(%/월)", title="시장 성장과 점포공급의 조합(HIGH 매핑)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "01_market_supply_types.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(visitor.stay_relative_growth_pct_m, visitor.bc_amt_relative_growth_jan_mar_pct_m,
               c=np.where(visitor.admin_type.eq("군"), "#E45756", "#4C78A8"), alpha=.72)
    ax.axhline(0, c="gray", lw=.8); ax.axvline(0, c="gray", lw=.8)
    ax.set(xlabel="체류인구 상대성장률(%/월, 1~3월)", ylabel="BC AMT 전국대비 상대성장률(%/월, 1~3월)", title="공식 체류인구와 BC 소비의 상대성장")
    fig.tight_layout(); fig.savefig(FIG / "02_visitor_bc_growth.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for typ, g in reb.groupby("property_type"):
        axes[0].scatter(g.rent_growth_pct, g.bc_amt_growth_pct_m, label=typ, alpha=.75)
        axes[1].scatter(g.vacancy_change_pp, g.bc_amt_growth_pct_m, label=typ, alpha=.75)
    axes[0].set(xlabel="임대료 Q/Q 변화(%)", ylabel="BC AMT 성장률(%/월)", title="소비와 임대료")
    axes[1].set(xlabel="공실률 Q/Q 변화(%p)", ylabel="BC AMT 성장률(%/월)", title="소비와 공실")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "03_reb_context.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(h8.regional_demand_supply_gap, h8.commercial_aging_gap_pp_m,
               c=h8.amt_fitted_change_6m_pct, cmap="viridis", s=38)
    ax.axhline(0, c="gray", lw=.8); ax.axvline(0, c="gray", lw=.8)
    ax.set(xlabel="지역 수요-공급 Gap(%p/월)", ylabel="소비고령화 Gap(%p/월)", title="같은 6개월 AMT +5~10% 지역의 서로 다른 환경")
    fig.tight_layout(); fig.savefig(FIG / "04_same_growth_different_environment.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cl, g in pca_scores.groupby("cluster"):
        ax.scatter(g.PC1, g.PC2, s=25, alpha=.7, label=f"Cluster {cl}")
    ax.set(xlabel="PC1", ylabel="PC2", title="통합 구조 feature의 PCA와 Ward 군집")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "05_integrated_pca_clusters.png", dpi=180); plt.close(fig)

    top = loadings.set_index("feature")[[c for c in loadings if c.startswith("PC")][:3]]
    fig, ax = plt.subplots(figsize=(8, 7)); im=ax.imshow(top, cmap="RdBu_r", aspect="auto", vmin=-.6, vmax=.6)
    ax.set_yticks(range(len(top)), top.index); ax.set_xticks(range(top.shape[1]), top.columns)
    fig.colorbar(im, ax=ax, label="PCA loading"); ax.set_title("잠재차원을 구성하는 변수")
    fig.tight_layout(); fig.savefig(FIG / "06_pca_loadings.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    bc = load_bc()
    population_month = parse_population(bc[RKEY].drop_duplicates())
    stores, _ = prepare_store_panel(bc, population_month)
    h1, h2_detail, h2_summary, h2_tests, h3 = h1_h3_tables(stores)
    features = core_region_features(stores)
    living, visitor, visitor_k, visitor_clusters, visitor_assoc, h5 = visitor_analysis(bc, features)
    reb, reb_tests = reb_analysis(bc, stores)
    h8 = h8_similar_growth(features, visitor, reb)
    integrated, loadings, pca_scores, cluster_selection, cluster_profile, pca_quality = pca_clusters(features, visitor)
    negatives = negative_tests(stores, visitor, reb_tests)
    create_plots(stores, visitor, reb, h8, pca_scores, loadings)

    availability = pd.DataFrame([
        ["BC카드", "ABP_CONTEST_DATA", "2026.01~06", "시군구×월×업종×연령×성별", 255, "확보", "주 분석"],
        ["행정안전부", "주민등록 연령별 인구", "2026.01~06", "시군구×월", 251, "확보", "화성 신설 4구 결측"],
        ["행정안전부", "인구감소지역 생활인구", "2026.01~03", "89개 시군구×월", 89, "확보", "체류인구는 관광객보다 넓은 개념"],
        ["LOCALDATA", "식품 인허가", "2026.01~06 재구성", "시군구×월×연결업종", 255, "확보", "매핑 HIGH 2개·MEDIUM 4개"],
        ["국세청", "월간 지역 경제지표", "2026.01~06", "시군구×월×생활업종", 255, "확보", "편의점·슈퍼마켓 점포 대리"],
        ["소상공인시장진흥공단", "상가(상권)정보 API", "현재 시점", "개별 상가", 0, "API키 미확보", "무료·자동승인이나 인증키 필요; 월별 이력 미확보"],
        ["한국관광공사", "한국관광데이터랩 방문자", "2026.01~06 요청", "시군구×월", 0, "미확보", "로그인·사전설문 다운로드 제한; 추정하지 않음"],
        ["한국부동산원", "상업용부동산 임대동향", "2026.Q1~Q2", "시도·대표상권×분기", 17, "확보", "시군구 직접 매칭 불가; 시도 검증"],
        ["국가데이터처", "세부 CPI", "2026.01~06", "전국×월", 255, "확보", "업종 대응 일부 MEDIUM"],
    ], columns=["기관", "데이터", "기간", "공간·시간단위", "coverage_regions", "상태", "한계·사용법"])

    tables = {
        "00_external_data_availability.csv": availability,
        "01_bc_store_mapping.csv": pd.read_csv(MAPPING),
        "02_living_population_monthly.csv": living,
        "03_visitor_demand_region.csv": visitor,
        "04_market_store_metrics_combined.csv": stores,
        "05_h1_market_opportunity_summary.csv": h1,
        "06_h2_gap_group_detail.csv": h2_detail,
        "07_h2_gap_group_summary.csv": h2_summary,
        "08_h2_gap_extreme_tests.csv": h2_tests,
        "09_h3_competition_pressure.csv": h3,
        "10_h4_visitor_cluster_selection.csv": visitor_k,
        "11_h4_visitor_cluster_profile.csv": visitor_clusters,
        "12_h4_visitor_association_tests.csv": visitor_assoc,
        "13_h5_visitor_growth_comparison.csv": h5,
        "14_h6_h7_reb_province.csv": reb,
        "15_h6_h7_reb_tests.csv": reb_tests,
        "16_h8_similar_amt_growth_regions.csv": h8,
        "17_h9_pca_loadings.csv": loadings,
        "18_h9_pca_scores_clusters.csv": pca_scores,
        "19_h9_cluster_selection.csv": cluster_selection,
        "20_h9_cluster_profile.csv": cluster_profile,
        "21_negative_tests.csv": negatives,
        "22_integrated_region_features.csv": integrated,
    }
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "fourth_external_integration_tables.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:28], index=False)

    summary = {
        "h1_high": h1[(h1.confidence.eq("HIGH")) & (h1.cnt_threshold.eq(0)) & (~h1.robust_only)].to_dict("records"),
        "h1_high_robust": h1[(h1.confidence.eq("HIGH")) & (h1.cnt_threshold.eq(0)) & (h1.robust_only)].to_dict("records"),
        "visitor_regions": len(visitor), "visitor_clusters": visitor_clusters.to_dict("records"),
        "h5_tests": h5.to_dict("records"),
        "h6_types": reb.h6_type.value_counts().to_dict(),
        "h7_amt_up_vacancy_up": int(reb.h7_amt_up_vacancy_up.sum()),
        "h7_denominator": len(reb), "pca_quality": pca_quality,
        "same_growth_regions": len(h8), "negative_tests": negatives.to_dict("records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
