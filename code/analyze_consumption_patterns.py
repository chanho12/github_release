"""2026년 1~6월 BC카드 데이터의 지역 소비구조 변화 패턴 탐색.

실행:
    python3 scripts/analyze_consumption_patterns.py

출력:
    analysis/consumption_patterns/
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "bc_consumption_pattern_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max((os.cpu_count() or 2) - 1, 1))
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\..*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from analyze_growth_illusion import (
    AGE_NAMES,
    DATA_PATH,
    INDUSTRY_GROUPS,
    INDUSTRY_NAMES,
    bh_adjust,
    build_slope_table,
    configure_plot,
    slope_stats,
    structure_shift,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "consumption_patterns"
FIG = OUT / "figures"
MONTHS = list(range(202601, 202607))
REGION_KEYS = ["SIDO_NM", "CCG_NM"]
RESTAURANT_CODES = [8001, 8002, 8003, 8004, 8005, 8006]


def log_slope(values: np.ndarray) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) != 6 or np.isnan(y).any() or (y <= 0).any():
        return {"beta": np.nan, "pct_month": np.nan, "fitted_change_pct": np.nan, "r2": np.nan}
    result = linregress(x, np.log(y))
    return {
        "beta": float(result.slope),
        "pct_month": float(np.expm1(result.slope) * 100),
        "fitted_change_pct": float(np.expm1(result.slope * 5) * 100),
        "r2": float(result.rvalue**2),
    }


def linear_slope(values: np.ndarray) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    result = linregress(np.arange(len(y), dtype=float), y)
    return {
        "slope": float(result.slope),
        "fitted_change": float(result.slope * 5),
        "r2": float(result.rvalue**2),
        "p": float(result.pvalue),
    }


def amt_decomposition(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = data.groupby(REGION_KEYS + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    monthly["avg_ticket"] = monthly["amt"] / monthly["cnt"]
    rows = []
    for key, group in monthly.groupby(REGION_KEYS, sort=False):
        group = group.sort_values("STRD_YYMM")
        amt = log_slope(group["amt"].to_numpy())
        cnt = log_slope(group["cnt"].to_numpy())
        ticket = log_slope(group["avg_ticket"].to_numpy())
        denom = abs(cnt["beta"]) + abs(ticket["beta"])
        if cnt["beta"] >= 0 and ticket["beta"] >= 0:
            if denom == 0:
                driver = "정체"
            elif abs(cnt["beta"]) / denom >= 0.6:
                driver = "건수 주도 성장"
            elif abs(ticket["beta"]) / denom >= 0.6:
                driver = "객단가 주도 성장"
            else:
                driver = "건수·객단가 동반 성장"
        elif cnt["beta"] < 0 <= ticket["beta"]:
            driver = "객단가 상승·건수 약화"
        elif ticket["beta"] < 0 <= cnt["beta"]:
            driver = "건수 상승·객단가 약화"
        else:
            driver = "건수·객단가 동반 약화"
        rows.append(
            {
                "SIDO_NM": key[0],
                "CCG_NM": key[1],
                "amt_log_beta": amt["beta"],
                "cnt_log_beta": cnt["beta"],
                "ticket_log_beta": ticket["beta"],
                "amt_growth_pct_month": amt["pct_month"],
                "cnt_growth_pct_month": cnt["pct_month"],
                "ticket_growth_pct_month": ticket["pct_month"],
                "amt_fitted_change_pct": amt["fitted_change_pct"],
                "cnt_fitted_change_pct": cnt["fitted_change_pct"],
                "ticket_fitted_change_pct": ticket["fitted_change_pct"],
                "amt_r2": amt["r2"],
                "cnt_r2": cnt["r2"],
                "ticket_r2": ticket["r2"],
                "decomposition_error": amt["beta"] - cnt["beta"] - ticket["beta"],
                "cnt_abs_contribution_share": abs(cnt["beta"]) / denom if denom else np.nan,
                "ticket_abs_contribution_share": abs(ticket["beta"]) / denom if denom else np.nan,
                "cnt_net_contribution_pct": cnt["beta"] / amt["beta"] * 100 if abs(amt["beta"]) > 1e-12 else np.nan,
                "ticket_net_contribution_pct": ticket["beta"] / amt["beta"] * 100 if abs(amt["beta"]) > 1e-12 else np.nan,
                "amt_driver_type": driver,
                "total_amt_6m": group["amt"].sum(),
                "total_cnt_6m": group["cnt"].sum(),
            }
        )
    return pd.DataFrame(rows), monthly


def age_concentration(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    human = data.loc[data["AGE_CD"].astype(str).isin(AGE_NAMES)].copy()
    age = human.groupby(REGION_KEYS + ["STRD_YYMM", "AGE_CD"], as_index=False)["cnt"].sum()
    totals = age.groupby(REGION_KEYS + ["STRD_YYMM"], as_index=False)["cnt"].sum().rename(columns={"cnt": "human_cnt"})
    age = age.merge(totals, on=REGION_KEYS + ["STRD_YYMM"], validate="many_to_one")
    age["age_cnt_share"] = age["cnt"] / age["human_cnt"]

    concentration = (
        age.assign(
            hhi_component=lambda x: x["age_cnt_share"] ** 2,
            entropy_component=lambda x: -x["age_cnt_share"] * np.log(x["age_cnt_share"]),
        )
        .groupby(REGION_KEYS + ["STRD_YYMM"], as_index=False)
        .agg(age_hhi=("hhi_component", "sum"), age_entropy_raw=("entropy_component", "sum"), observed_age_groups=("AGE_CD", "nunique"))
    )
    concentration["age_entropy_normalized"] = concentration["age_entropy_raw"] / math.log(len(AGE_NAMES))

    metric_rows = []
    for key, group in concentration.groupby(REGION_KEYS, sort=False):
        group = group.sort_values("STRD_YYMM")
        hhi = linear_slope(group["age_hhi"].to_numpy())
        entropy = linear_slope(group["age_entropy_normalized"].to_numpy())
        hhi_diff = group["age_hhi"].diff()
        entropy_diff = group["age_entropy_normalized"].diff()
        metric_rows.append(
            {
                "SIDO_NM": key[0],
                "CCG_NM": key[1],
                "age_hhi_slope_month": hhi["slope"],
                "age_hhi_fitted_change": hhi["fitted_change"],
                "age_hhi_r2": hhi["r2"],
                "age_entropy_slope_month": entropy["slope"],
                "age_entropy_fitted_change": entropy["fitted_change"],
                "age_entropy_r2": entropy["r2"],
                "concentration_increase": bool(hhi["slope"] > 0 and entropy["slope"] < 0),
                "concentration_increase_intervals": int(((hhi_diff > 0) & (entropy_diff < 0)).sum()),
                "min_observed_age_groups": int(group["observed_age_groups"].min()),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    share_rows = []
    for key, group in age.groupby(REGION_KEYS + ["AGE_CD"], sort=False):
        group = group.sort_values("STRD_YYMM")
        if group["STRD_YYMM"].nunique() != 6:
            continue
        stats = linear_slope(group["age_cnt_share"].to_numpy())
        share_rows.append(
            {
                "SIDO_NM": key[0],
                "CCG_NM": key[1],
                "AGE_CD": str(key[2]),
                "연령대": AGE_NAMES[str(key[2])],
                "age_share_slope_pp_month": stats["slope"] * 100,
                "age_share_fitted_change_pp": stats["fitted_change"] * 100,
                "age_share_r2": stats["r2"],
            }
        )
    return metrics, concentration, pd.DataFrame(share_rows)


def transition_age_comparison(
    shifts: pd.DataFrame, age_metrics: pd.DataFrame, age_shares: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = shifts[REGION_KEYS + ["amt_cnt_both_shift"]].rename(columns={"amt_cnt_both_shift": "transition_group"})
    age_joined = age_shares.merge(membership, on=REGION_KEYS, validate="many_to_one")
    rows = []
    for age_code, age_name in AGE_NAMES.items():
        subset = age_joined.loc[age_joined["AGE_CD"] == age_code]
        a = subset.loc[subset["transition_group"], "age_share_slope_pp_month"]
        b = subset.loc[~subset["transition_group"], "age_share_slope_pp_month"]
        test = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "metric": f"age_share_{age_code}",
                "연령대": age_name,
                "transition_n": len(a),
                "general_n": len(b),
                "transition_median_pp_month": a.median(),
                "general_median_pp_month": b.median(),
                "median_difference_pp_month": a.median() - b.median(),
                "mannwhitney_p": test.pvalue,
                "rank_biserial": 2 * test.statistic / (len(a) * len(b)) - 1,
            }
        )
    age_result = pd.DataFrame(rows)
    age_result["p_fdr"] = bh_adjust(age_result["mannwhitney_p"])

    region = age_metrics.merge(membership, on=REGION_KEYS, validate="one_to_one")
    extra = []
    for column, label in [
        ("age_hhi_fitted_change", "HHI 6개월 적합 변화"),
        ("age_entropy_fitted_change", "정규화 Entropy 6개월 적합 변화"),
    ]:
        a = region.loc[region["transition_group"], column]
        b = region.loc[~region["transition_group"], column]
        test = mannwhitneyu(a, b, alternative="two-sided")
        extra.append(
            {
                "metric": column,
                "지표": label,
                "transition_n": len(a),
                "general_n": len(b),
                "transition_median": a.median(),
                "general_median": b.median(),
                "median_difference": a.median() - b.median(),
                "mannwhitney_p": test.pvalue,
                "rank_biserial": 2 * test.statistic / (len(a) * len(b)) - 1,
            }
        )
    extra_result = pd.DataFrame(extra)
    extra_result["p_fdr"] = bh_adjust(extra_result["mannwhitney_p"])
    return age_result, extra_result


def restaurant_weakening(slopes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = slopes.loc[slopes["TP_BUZ_NO"].isin(RESTAURANT_CODES)].copy()
    complete = subset.loc[subset["complete_6m"]].copy()
    complete["cnt_weakened_vs_national"] = complete["relative_cnt_slope_pct_month"] < 0
    detail = complete[
        REGION_KEYS
        + ["TP_BUZ_NO", "TP_BUZ_NM", "relative_cnt_slope_pct_month", "cnt_weakened_vs_national", "total_cnt_6m"]
    ].copy()
    summary = (
        complete.groupby(REGION_KEYS, as_index=False)
        .agg(
            restaurant_eligible_count=("TP_BUZ_NO", "nunique"),
            restaurant_weakened_count=("cnt_weakened_vs_national", "sum"),
            restaurant_mean_relative_cnt_slope=("relative_cnt_slope_pct_month", "mean"),
            restaurant_total_cnt_6m=("total_cnt_6m", "sum"),
        )
    )
    all_regions = slopes[REGION_KEYS].drop_duplicates()
    summary = all_regions.merge(summary, on=REGION_KEYS, how="left")
    summary["restaurant_eligible_count"] = summary["restaurant_eligible_count"].fillna(0).astype(int)
    summary["restaurant_weakened_count"] = summary["restaurant_weakened_count"].fillna(0).astype(int)
    summary["restaurant_weakening_rate"] = summary["restaurant_weakened_count"] / summary["restaurant_eligible_count"].replace(0, np.nan)
    summary["all_6_restaurants_observed"] = summary["restaurant_eligible_count"] == 6
    return summary, detail


def make_feature_table(
    decomposition: pd.DataFrame,
    age_metrics: pd.DataFrame,
    age_shares: pd.DataFrame,
    shifts: pd.DataFrame,
    weakening: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    age_wide = age_shares.pivot(index=REGION_KEYS, columns="AGE_CD", values="age_share_slope_pp_month").reset_index()
    age_wide = age_wide.rename(columns={code: f"age_share_slope_{code}" for code in AGE_NAMES})
    selected_shift = [
        "amt_share_slope_pp_month_생활구매형",
        "cnt_share_slope_pp_month_생활구매형",
        "amt_share_slope_pp_month_외식형",
        "cnt_share_slope_pp_month_외식형",
        "amt_cnt_both_shift",
    ]
    table = decomposition.merge(age_metrics, on=REGION_KEYS, validate="one_to_one")
    table = table.merge(age_wide, on=REGION_KEYS, validate="one_to_one")
    table = table.merge(shifts[REGION_KEYS + selected_shift], on=REGION_KEYS, validate="one_to_one")
    table = table.merge(weakening, on=REGION_KEYS, validate="one_to_one")
    # 연령 점유율 6개는 합이 0이므로 60대 이상을 기준범주로 제외한다.
    features = [
        "amt_growth_pct_month",
        "cnt_growth_pct_month",
        "ticket_growth_pct_month",
        "age_hhi_slope_month",
        "age_entropy_slope_month",
        "age_share_slope_1",
        "age_share_slope_2",
        "age_share_slope_3",
        "age_share_slope_4",
        "age_share_slope_5",
        "amt_share_slope_pp_month_생활구매형",
        "cnt_share_slope_pp_month_생활구매형",
        "amt_share_slope_pp_month_외식형",
        "cnt_share_slope_pp_month_외식형",
        "restaurant_weakening_rate",
    ]
    return table, features


def cluster_regions(
    feature_table: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    X_raw = feature_table[features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=features, index=X_raw.index)
    lower = X_imputed.quantile(0.01)
    upper = X_imputed.quantile(0.99)
    X_winsor = X_imputed.clip(lower=lower, upper=upper, axis=1)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_winsor)
    blocks = {
        "성장동력": ["amt_growth_pct_month", "cnt_growth_pct_month", "ticket_growth_pct_month"],
        "연령구조": ["age_hhi_slope_month", "age_entropy_slope_month"] + [f"age_share_slope_{code}" for code in ["1", "2", "3", "4", "5"]],
        "업종구조": [
            "amt_share_slope_pp_month_생활구매형",
            "cnt_share_slope_pp_month_생활구매형",
            "amt_share_slope_pp_month_외식형",
            "cnt_share_slope_pp_month_외식형",
        ],
        "외식약화": ["restaurant_weakening_rate"],
    }
    weights = np.ones(len(features), dtype=float)
    for columns in blocks.values():
        block_weight = 1 / math.sqrt(len(columns))
        for column in columns:
            weights[features.index(column)] = block_weight
    X = X_standardized * weights

    selection_rows = []
    fitted: dict[int, KMeans] = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X)
        fitted[k] = model
        selection_rows.append({"k": k, "silhouette": silhouette_score(X, labels), "inertia": model.inertia_})
    selection = pd.DataFrame(selection_rows)
    best_k = int(selection.loc[selection["silhouette"].idxmax(), "k"])
    model = fitted[best_k]
    labels = model.labels_

    stability = []
    for seed in range(20):
        candidate = KMeans(n_clusters=best_k, random_state=seed, n_init=20).fit_predict(X)
        stability.append(adjusted_rand_score(labels, candidate))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    clustered = feature_table.copy()
    clustered["cluster"] = labels + 1
    clustered["pca1"] = coords[:, 0]
    clustered["pca2"] = coords[:, 1]

    standardized = pd.DataFrame(X_standardized, columns=features, index=feature_table.index)
    standardized["cluster"] = labels + 1
    profile_z = standardized.groupby("cluster")[features].mean().reset_index()

    summary_rows = []
    for cluster, group in clustered.groupby("cluster"):
        center = model.cluster_centers_[cluster - 1]
        indices = group.index.to_numpy()
        distance = np.linalg.norm(X[indices] - center, axis=1)
        representative = group.loc[indices[np.argmin(distance)]]
        summary_rows.append(
            {
                "cluster": cluster,
                "region_count": len(group),
                "region_rate_pct": len(group) / len(clustered) * 100,
                "representative_region": f"{representative['SIDO_NM']} {representative['CCG_NM']}",
                "amt_growth_median": group["amt_growth_pct_month"].median(),
                "cnt_growth_median": group["cnt_growth_pct_month"].median(),
                "ticket_growth_median": group["ticket_growth_pct_month"].median(),
                "hhi_change_median": group["age_hhi_fitted_change"].median(),
                "entropy_change_median": group["age_entropy_fitted_change"].median(),
                "life_amt_share_slope_median": group["amt_share_slope_pp_month_생활구매형"].median(),
                "restaurant_weakening_rate_median": group["restaurant_weakening_rate"].median(),
                "transition_region_count": int(group["amt_cnt_both_shift"].sum()),
                "concentration_increase_count": int(group["concentration_increase"].sum()),
                "age60_share_slope_median_pp_month": group["age_share_slope_6"].median(),
                "county_name_count": int(group["CCG_NM"].str.endswith("군").sum()),
            }
        )
    cluster_summary = pd.DataFrame(summary_rows)
    metadata = {
        "best_k": best_k,
        "silhouette": float(selection.loc[selection["k"] == best_k, "silhouette"].iloc[0]),
        "stability_ari_mean": float(np.mean(stability)),
        "stability_ari_min": float(np.min(stability)),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "features": features,
        "block_weighting": {name: {"feature_count": len(columns), "per_feature_weight": 1 / math.sqrt(len(columns))} for name, columns in blocks.items()},
    }
    return clustered, selection, profile_z, cluster_summary, metadata


def plot_outputs(
    decomposition: pd.DataFrame,
    age_metrics: pd.DataFrame,
    age_comparison: pd.DataFrame,
    weakening: pd.DataFrame,
    clustered: pd.DataFrame,
    selection: pd.DataFrame,
    profile_z: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    configure_plot()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    order = decomposition["amt_driver_type"].value_counts().index
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
    size_scale = np.log1p(decomposition["total_cnt_6m"])
    sizes = 20 + 160 * (size_scale - size_scale.min()) / (size_scale.max() - size_scale.min())
    for color, label in zip(colors, order):
        mask = decomposition["amt_driver_type"] == label
        ax.scatter(
            decomposition.loc[mask, "cnt_growth_pct_month"],
            decomposition.loc[mask, "ticket_growth_pct_month"],
            s=sizes.loc[mask],
            color=color,
            label=label,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="AMT 성장의 건수·객단가 분해", xlabel="CNT 로그 성장률(%/월)", ylabel="객단가 로그 성장률(%/월)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_amt_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for flag, color, label in [(False, "#999999", "기타"), (True, "#D55E00", "집중도 증가")]:
        part = age_metrics.loc[age_metrics["concentration_increase"] == flag]
        ax.scatter(part["age_hhi_fitted_change"], part["age_entropy_fitted_change"], color=color, label=label, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="연령별 CNT 소비층 집중도 변화", xlabel="HHI 6개월 적합 변화", ylabel="정규화 Entropy 6개월 적합 변화")
    fig.tight_layout()
    fig.savefig(FIG / "02_age_concentration.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot = age_comparison.copy()
    x = np.arange(len(plot))
    width = 0.36
    ax.bar(x - width / 2, plot["transition_median_pp_month"], width, label="외식→생활구매 38개 지역")
    ax.bar(x + width / 2, plot["general_median_pp_month"], width, label="일반지역")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, plot["연령대"])
    ax.set(title="전환지역과 일반지역의 연령별 CNT 점유율 변화", ylabel="중앙값(%p/월)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_transition_age_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    counts = weakening["restaurant_weakened_count"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#0072B2")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 1, str(value), ha="center")
    ax.set(title="지역별 전국 대비 CNT 약화 외식업종 수", xlabel="약화 업종 수(관측된 외식업종 중, 최대 6)", ylabel="시군구 수")
    fig.tight_layout()
    fig.savefig(FIG / "04_restaurant_weakening_width.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["silhouette"], marker="o")
    best = selection.loc[selection["silhouette"].idxmax()]
    ax.scatter([best["k"]], [best["silhouette"]], color="#D55E00", s=90, zorder=3)
    ax.set(title="군집 수 선택", xlabel="군집 수 k", ylabel="Silhouette score", xticks=selection["k"])
    fig.tight_layout()
    fig.savefig(FIG / "05_cluster_selection.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    cluster_values = sorted(clustered["cluster"].unique())
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_values)))
    for color, cluster in zip(cluster_colors, cluster_values):
        part = clustered.loc[clustered["cluster"] == cluster]
        ax.scatter(part["pca1"], part["pca2"], color=color, label=f"Cluster {cluster}", s=65, alpha=0.8)
    ax.legend()
    ax.set(title="시군구 소비변화 군집 — PCA 투영", xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    fig.savefig(FIG / "06_cluster_pca.png", bbox_inches="tight")
    plt.close(fig)

    labels = {
        "amt_growth_pct_month": "AMT 성장",
        "cnt_growth_pct_month": "CNT 성장",
        "ticket_growth_pct_month": "객단가 성장",
        "age_hhi_slope_month": "연령 HHI",
        "age_entropy_slope_month": "연령 Entropy",
        "age_share_slope_1": "20대 이하 점유율",
        "age_share_slope_2": "20대 점유율",
        "age_share_slope_3": "30대 점유율",
        "age_share_slope_4": "40대 점유율",
        "age_share_slope_5": "50대 점유율",
        "amt_share_slope_pp_month_생활구매형": "생활구매 AMT 비중",
        "cnt_share_slope_pp_month_생활구매형": "생활구매 CNT 비중",
        "amt_share_slope_pp_month_외식형": "외식 AMT 비중",
        "cnt_share_slope_pp_month_외식형": "외식 CNT 비중",
        "restaurant_weakening_rate": "외식 약화율",
    }
    heat = profile_z.set_index("cluster").rename(columns=labels)
    fig, ax = plt.subplots(figsize=(15, max(3.5, len(heat) * 0.9)))
    limit = max(1.0, float(np.nanmax(np.abs(heat.to_numpy()))))
    image = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    for row in range(len(heat.index)):
        for column in range(len(heat.columns)):
            value = heat.iloc[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("전체 평균 대비 표준편차")
    ax.set(title="군집별 소비변화 feature 프로필", xlabel="", ylabel="Cluster")
    fig.tight_layout()
    fig.savefig(FIG / "07_cluster_profile_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_outputs(tables: dict[str, pd.DataFrame]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "consumption_pattern_analysis_tables.xlsx", engine="openpyxl") as writer:
        for filename, frame in tables.items():
            sheet = Path(filename).stem[:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)


def main() -> None:
    configure_plot()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, dtype={"GENDER_CD": "string", "AGE_CD": "string"})
    data["TP_BUZ_NM"] = data["TP_BUZ_NO"].map(INDUSTRY_NAMES)

    decomposition, region_monthly = amt_decomposition(data)
    national_monthly = data.groupby("STRD_YYMM", as_index=False)[["amt", "cnt"]].sum()
    national_monthly["avg_ticket"] = national_monthly["amt"] / national_monthly["cnt"]
    national_stats = {name: log_slope(national_monthly[column].to_numpy()) for name, column in [("amt", "amt"), ("cnt", "cnt"), ("ticket", "avg_ticket")]}
    national_decomposition = pd.DataFrame(
        [
            {
                "amt_growth_pct_month": national_stats["amt"]["pct_month"],
                "cnt_growth_pct_month": national_stats["cnt"]["pct_month"],
                "ticket_growth_pct_month": national_stats["ticket"]["pct_month"],
                "amt_fitted_change_pct": national_stats["amt"]["fitted_change_pct"],
                "cnt_fitted_change_pct": national_stats["cnt"]["fitted_change_pct"],
                "ticket_fitted_change_pct": national_stats["ticket"]["fitted_change_pct"],
                "cnt_net_contribution_pct": national_stats["cnt"]["beta"] / national_stats["amt"]["beta"] * 100,
                "ticket_net_contribution_pct": national_stats["ticket"]["beta"] / national_stats["amt"]["beta"] * 100,
                "decomposition_error": national_stats["amt"]["beta"] - national_stats["cnt"]["beta"] - national_stats["ticket"]["beta"],
            }
        ]
    )
    age_metrics, age_monthly, age_shares = age_concentration(data)
    shifts = structure_shift(data)
    age_comparison, concentration_comparison = transition_age_comparison(shifts, age_metrics, age_shares)
    slopes, _ = build_slope_table(data)
    weakening, weakening_detail = restaurant_weakening(slopes)
    features, feature_columns = make_feature_table(decomposition, age_metrics, age_shares, shifts, weakening)
    clustered, cluster_selection, cluster_profile, cluster_summary, cluster_metadata = cluster_regions(features, feature_columns)

    driver_summary = (
        decomposition.groupby("amt_driver_type", as_index=False)
        .agg(region_count=("CCG_NM", "size"), median_amt_growth=("amt_growth_pct_month", "median"), median_cnt_growth=("cnt_growth_pct_month", "median"), median_ticket_growth=("ticket_growth_pct_month", "median"))
    )
    driver_summary["region_rate_pct"] = driver_summary["region_count"] / len(decomposition) * 100
    concentration_top = age_metrics.sort_values("age_hhi_fitted_change", ascending=False).head(30)

    tables = {
        "00_national_amt_decomposition.csv": national_decomposition,
        "01_amt_decomposition_by_region.csv": decomposition,
        "02_region_monthly_amt_cnt_ticket.csv": region_monthly,
        "03_amt_driver_summary.csv": driver_summary.sort_values("region_count", ascending=False),
        "04_age_concentration_by_region.csv": age_metrics,
        "05_age_concentration_monthly.csv": age_monthly,
        "06_age_share_slopes_by_region.csv": age_shares,
        "07_concentration_increase_top30.csv": concentration_top,
        "08_transition_age_comparison.csv": age_comparison,
        "09_transition_concentration_comparison.csv": concentration_comparison,
        "10_restaurant_weakening_by_region.csv": weakening,
        "11_restaurant_weakening_detail.csv": weakening_detail,
        "12_region_feature_cluster.csv": clustered,
        "13_cluster_selection.csv": cluster_selection,
        "14_cluster_profile_zscore.csv": cluster_profile,
        "15_cluster_summary.csv": cluster_summary,
    }
    save_outputs(tables)
    plot_outputs(decomposition, age_metrics, age_comparison, weakening, clustered, cluster_selection, cluster_profile)

    metrics = {
        "raw_rows": len(data),
        "regions": len(decomposition),
        "max_decomposition_error": float(decomposition["decomposition_error"].abs().max()),
        "amt_growth_regions": int((decomposition["amt_log_beta"] > 0).sum()),
        "driver_counts": {str(k): int(v) for k, v in decomposition["amt_driver_type"].value_counts().items()},
        "concentration_increase_regions": int(age_metrics["concentration_increase"].sum()),
        "concentration_increase_4plus_intervals": int((age_metrics["concentration_increase_intervals"] >= 4).sum()),
        "transition_regions": int(shifts["amt_cnt_both_shift"].sum()),
        "restaurant_full_coverage_regions": int(weakening["all_6_restaurants_observed"].sum()),
        "restaurant_weakened_count_distribution": {str(k): int(v) for k, v in weakening["restaurant_weakened_count"].value_counts().sort_index().items()},
        "restaurant_eligible_4plus_regions": int((weakening["restaurant_eligible_count"] >= 4).sum()),
        "restaurant_eligible_4plus_half_weakened": int(((weakening["restaurant_eligible_count"] >= 4) & (weakening["restaurant_weakening_rate"] >= 0.5)).sum()),
        "cluster": cluster_metadata,
    }
    (OUT / "analysis_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
