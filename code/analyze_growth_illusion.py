"""BC카드 2026-01~06 상권 성장착시 탐색 분석.

실행:
    python3 scripts/analyze_growth_illusion.py

출력:
    analysis/growth_illusion/
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

# VS Code/Jupyter와 제한된 실행 환경에서도 글꼴 캐시 경고 없이 재현되도록
# 분석 전용 임시 캐시를 사용한다.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "bc_growth_illusion_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, linregress, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "ABP_CONTEST_DATA.csv"
OUT = ROOT / "analysis" / "growth_illusion"
FIG = OUT / "figures"
MONTHS = list(range(202601, 202607))

INDUSTRY_NAMES = {
    4004: "대형할인점",
    4010: "편의점",
    4020: "슈퍼마켓",
    8001: "일반한식",
    8002: "갈비전문점",
    8003: "한정식",
    8004: "일식회집",
    8005: "중국음식",
    8006: "서양음식",
    8021: "스넥",
    8301: "제과점",
}

AGE_NAMES = {
    "1": "20대 이하",
    "2": "20대",
    "3": "30대",
    "4": "40대",
    "5": "50대",
    "6": "60대 이상",
}

INDUSTRY_GROUPS = {
    4004: "생활구매형",
    4010: "생활구매형",
    4020: "생활구매형",
    8001: "외식형",
    8002: "외식형",
    8003: "외식형",
    8004: "외식형",
    8005: "외식형",
    8006: "외식형",
    8021: "간식형",
    8301: "간식형",
}

QUADRANTS = {
    (True, False): "성장착시 후보",
    (True, True): "건강성장",
    (False, True): "건수성장·금액감소",
    (False, False): "동반감소",
}

COLORS = {
    "성장착시 후보": "#D55E00",
    "건강성장": "#009E73",
    "건수성장·금액감소": "#0072B2",
    "동반감소": "#999999",
}


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "AppleGothic",
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def slope_stats(values: np.ndarray, log: bool = False) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) != 6 or np.isnan(y).any() or (log and (y <= 0).any()):
        return {"slope": np.nan, "rate_pct": np.nan, "r2": np.nan, "p": np.nan}
    target = np.log(y) if log else y
    result = linregress(x, target)
    return {
        "slope": float(result.slope),
        "rate_pct": float((math.exp(result.slope) - 1) * 100) if log else np.nan,
        "r2": float(result.rvalue**2),
        "p": float(result.pvalue),
    }


def quadrant(amt_slope: float, cnt_slope: float) -> str | None:
    if pd.isna(amt_slope) or pd.isna(cnt_slope):
        return None
    return QUADRANTS[(amt_slope > 0, cnt_slope > 0)]


def max_consecutive(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (center - margin) * 100, (center + margin) * 100


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    n = len(values)
    for rank_from_end, index in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        running = min(running, values[index] * n / rank)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def build_slope_table(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "TP_BUZ_NM"]
    panel = data.groupby(keys + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()

    national = (
        data.groupby(["TP_BUZ_NO", "STRD_YYMM"], as_index=False)[["amt", "cnt"]]
        .sum()
        .rename(columns={"amt": "national_amt", "cnt": "national_cnt"})
    )
    panel = panel.merge(national, on=["TP_BUZ_NO", "STRD_YYMM"], how="left", validate="many_to_one")
    panel["relative_amt_share"] = panel["amt"] / panel["national_amt"]
    panel["relative_cnt_share"] = panel["cnt"] / panel["national_cnt"]
    panel["avg_ticket"] = panel["amt"] / panel["cnt"]

    regions = data[["SIDO_NM", "CCG_NM"]].drop_duplicates().assign(_key=1)
    industries = (
        pd.DataFrame({"TP_BUZ_NO": list(INDUSTRY_NAMES), "TP_BUZ_NM": list(INDUSTRY_NAMES.values())})
        .assign(_key=1)
    )
    full = regions.merge(industries, on="_key").drop(columns="_key")

    rows: list[dict[str, object]] = []
    grouped = {key: group.sort_values("STRD_YYMM") for key, group in panel.groupby(keys, sort=False)}
    for row in full.itertuples(index=False):
        key = (row.SIDO_NM, row.CCG_NM, row.TP_BUZ_NO, row.TP_BUZ_NM)
        group = grouped.get(key)
        base: dict[str, object] = {
            "SIDO_NM": row.SIDO_NM,
            "CCG_NM": row.CCG_NM,
            "TP_BUZ_NO": row.TP_BUZ_NO,
            "TP_BUZ_NM": row.TP_BUZ_NM,
            "n_months": 0 if group is None else group["STRD_YYMM"].nunique(),
        }
        base["complete_6m"] = base["n_months"] == 6
        if not base["complete_6m"]:
            rows.append(base)
            continue

        assert group is not None
        amt_raw = slope_stats(group["amt"].to_numpy())
        cnt_raw = slope_stats(group["cnt"].to_numpy())
        amt_log = slope_stats(group["amt"].to_numpy(), log=True)
        cnt_log = slope_stats(group["cnt"].to_numpy(), log=True)
        rel_amt = slope_stats(group["relative_amt_share"].to_numpy())
        rel_cnt = slope_stats(group["relative_cnt_share"].to_numpy())
        ticket = slope_stats(group["avg_ticket"].to_numpy())
        base.update(
            {
                "amt_slope_won_per_month": amt_raw["slope"],
                "cnt_slope_per_month": cnt_raw["slope"],
                "amt_slope_pct_month": amt_raw["slope"] / group["amt"].mean() * 100,
                "cnt_slope_pct_month": cnt_raw["slope"] / group["cnt"].mean() * 100,
                "amt_log_trend_pct_month": amt_log["rate_pct"],
                "cnt_log_trend_pct_month": cnt_log["rate_pct"],
                "amt_slope_r2": amt_raw["r2"],
                "cnt_slope_r2": cnt_raw["r2"],
                "amt_slope_p": amt_raw["p"],
                "cnt_slope_p": cnt_raw["p"],
                "relative_amt_slope_pct_month": rel_amt["slope"] / group["relative_amt_share"].mean() * 100,
                "relative_cnt_slope_pct_month": rel_cnt["slope"] / group["relative_cnt_share"].mean() * 100,
                "relative_amt_r2": rel_amt["r2"],
                "relative_cnt_r2": rel_cnt["r2"],
                "relative_amt_p": rel_amt["p"],
                "relative_cnt_p": rel_cnt["p"],
                "avg_ticket_slope_pct_month": ticket["slope"] / group["avg_ticket"].mean() * 100,
                "total_amt_6m": group["amt"].sum(),
                "total_cnt_6m": group["cnt"].sum(),
                "absolute_type": quadrant(amt_raw["slope"], cnt_raw["slope"]),
                "relative_type": quadrant(rel_amt["slope"], rel_cnt["slope"]),
            }
        )
        base["h1_candidate"] = base["absolute_type"] == "성장착시 후보"
        base["h2_candidate"] = base["relative_type"] == "성장착시 후보"
        base["h1_h2_candidate"] = bool(base["h1_candidate"] and base["h2_candidate"])
        rows.append(base)

    slopes = pd.DataFrame(rows)
    for column in ["h1_candidate", "h2_candidate", "h1_h2_candidate"]:
        slopes[column] = slopes[column].map(lambda value: bool(value) if pd.notna(value) else False)
    return slopes, panel


def quadrant_summary(slopes: pd.DataFrame) -> pd.DataFrame:
    valid = slopes.loc[slopes["complete_6m"]].copy()
    rows = []
    for basis, column in [("절대 추세", "absolute_type"), ("전국 동일업종 대비 상대 추세", "relative_type")]:
        counts = valid[column].value_counts()
        for type_name in QUADRANTS.values():
            count = int(counts.get(type_name, 0))
            rows.append({"기준": basis, "유형": type_name, "조합수": count, "비중_pct": count / len(valid) * 100})
    return pd.DataFrame(rows)


def industry_summary(slopes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    valid = slopes.loc[slopes["complete_6m"]].copy()
    rows = []
    for (code, name), group in valid.groupby(["TP_BUZ_NO", "TP_BUZ_NM"], sort=True):
        h1 = int(group["h1_candidate"].sum())
        h2 = int(group["h2_candidate"].sum())
        both = int(group["h1_h2_candidate"].sum())
        h1_low, h1_high = wilson_interval(h1, len(group))
        h2_low, h2_high = wilson_interval(h2, len(group))
        rows.append(
            {
                "TP_BUZ_NO": code,
                "TP_BUZ_NM": name,
                "analyzable_combos": len(group),
                "h1_count": h1,
                "h1_rate_pct": h1 / len(group) * 100,
                "h1_ci_low": h1_low,
                "h1_ci_high": h1_high,
                "h2_count": h2,
                "h2_rate_pct": h2 / len(group) * 100,
                "h2_ci_low": h2_low,
                "h2_ci_high": h2_high,
                "h1_h2_count": both,
                "h1_h2_rate_pct": both / len(group) * 100,
            }
        )
    summary = pd.DataFrame(rows).sort_values("h2_rate_pct", ascending=False)

    contingency = pd.crosstab(valid["TP_BUZ_NM"], valid["h2_candidate"])
    contingency = contingency.reindex(columns=[False, True], fill_value=0)
    chi2, p, dof, _ = chi2_contingency(contingency)
    return summary, {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(math.sqrt(chi2 / len(valid)))}


def age_comparison(data: pd.DataFrame, slopes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = slopes.loc[
        slopes["complete_6m"] & slopes["relative_type"].isin(["성장착시 후보", "건강성장"]),
        ["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "relative_type"],
    ]
    human = data.loc[data["AGE_CD"].astype(str).isin(AGE_NAMES)].copy()
    age = human.groupby(["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "STRD_YYMM", "AGE_CD"], as_index=False)["cnt"].sum()
    totals = age.groupby(["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "STRD_YYMM"], as_index=False)["cnt"].sum().rename(columns={"cnt": "human_cnt"})
    age = age.merge(totals, on=["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "STRD_YYMM"], validate="many_to_one")
    age["share"] = age["cnt"] / age["human_cnt"]
    age = age.merge(membership, on=["SIDO_NM", "CCG_NM", "TP_BUZ_NO"], how="inner", validate="many_to_one")

    combo_rows = []
    keys = ["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "relative_type", "AGE_CD"]
    for key, group in age.groupby(keys):
        group = group.sort_values("STRD_YYMM")
        if group["STRD_YYMM"].nunique() != 6:
            continue
        cnt_s = slope_stats(group["cnt"].to_numpy())
        share_s = slope_stats(group["share"].to_numpy())
        combo_rows.append(
            dict(zip(keys, key))
            | {
                "age_cnt_slope_pct_month": cnt_s["slope"] / group["cnt"].mean() * 100,
                "age_share_slope_pp_month": share_s["slope"] * 100,
            }
        )
    combo = pd.DataFrame(combo_rows)

    result_rows = []
    for age_code, age_name in AGE_NAMES.items():
        row: dict[str, object] = {"AGE_CD": age_code, "연령대": age_name}
        for type_name, prefix in [("성장착시 후보", "illusion"), ("건강성장", "healthy")]:
            subset = combo.loc[(combo["AGE_CD"].astype(str) == age_code) & (combo["relative_type"] == type_name)]
            row[f"{prefix}_n"] = len(subset)
            row[f"{prefix}_cnt_slope_median"] = subset["age_cnt_slope_pct_month"].median()
            row[f"{prefix}_share_slope_median"] = subset["age_share_slope_pp_month"].median()
        a = combo.loc[(combo["AGE_CD"].astype(str) == age_code) & (combo["relative_type"] == "성장착시 후보")]
        b = combo.loc[(combo["AGE_CD"].astype(str) == age_code) & (combo["relative_type"] == "건강성장")]
        row["cnt_mannwhitney_p"] = mannwhitneyu(a["age_cnt_slope_pct_month"], b["age_cnt_slope_pct_month"], alternative="two-sided").pvalue
        row["share_mannwhitney_p"] = mannwhitneyu(a["age_share_slope_pp_month"], b["age_share_slope_pp_month"], alternative="two-sided").pvalue
        result_rows.append(row)
    result = pd.DataFrame(result_rows)
    result["cnt_p_fdr"] = bh_adjust(result["cnt_mannwhitney_p"])
    result["share_p_fdr"] = bh_adjust(result["share_mannwhitney_p"])
    return result, combo


def structure_shift(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.copy()
    grouped["업종그룹"] = grouped["TP_BUZ_NO"].map(INDUSTRY_GROUPS)
    monthly = grouped.groupby(["SIDO_NM", "CCG_NM", "STRD_YYMM", "업종그룹"], as_index=False)[["amt", "cnt"]].sum()
    totals = monthly.groupby(["SIDO_NM", "CCG_NM", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum().rename(columns={"amt": "total_amt", "cnt": "total_cnt"})
    monthly = monthly.merge(totals, on=["SIDO_NM", "CCG_NM", "STRD_YYMM"], validate="many_to_one")
    monthly["amt_share"] = monthly["amt"] / monthly["total_amt"]
    monthly["cnt_share"] = monthly["cnt"] / monthly["total_cnt"]

    rows = []
    for key, group in monthly.groupby(["SIDO_NM", "CCG_NM", "업종그룹"]):
        group = group.sort_values("STRD_YYMM")
        if group["STRD_YYMM"].nunique() != 6:
            continue
        amt_s = slope_stats(group["amt_share"].to_numpy())
        cnt_s = slope_stats(group["cnt_share"].to_numpy())
        rows.append(
            {
                "SIDO_NM": key[0],
                "CCG_NM": key[1],
                "업종그룹": key[2],
                "amt_share_slope_pp_month": amt_s["slope"] * 100,
                "cnt_share_slope_pp_month": cnt_s["slope"] * 100,
                "amt_share_fitted_change_pp": amt_s["slope"] * 100 * 5,
                "cnt_share_fitted_change_pp": cnt_s["slope"] * 100 * 5,
            }
        )
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["SIDO_NM", "CCG_NM"], columns="업종그룹").reset_index()
    wide.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in wide.columns]
    wide["amt_external_to_life"] = (wide["amt_share_slope_pp_month_외식형"] < 0) & (wide["amt_share_slope_pp_month_생활구매형"] > 0)
    wide["cnt_external_to_life"] = (wide["cnt_share_slope_pp_month_외식형"] < 0) & (wide["cnt_share_slope_pp_month_생활구매형"] > 0)
    wide["amt_cnt_both_shift"] = wide["amt_external_to_life"] & wide["cnt_external_to_life"]
    return wide.sort_values(["amt_cnt_both_shift", "amt_share_fitted_change_pp_생활구매형"], ascending=[False, False])


def national_industry_trends(data: pd.DataFrame) -> pd.DataFrame:
    monthly = data.groupby(["TP_BUZ_NO", "TP_BUZ_NM", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    rows = []
    for (code, name), group in monthly.groupby(["TP_BUZ_NO", "TP_BUZ_NM"]):
        group = group.sort_values("STRD_YYMM")
        amt_s = slope_stats(group["amt"].to_numpy())
        cnt_s = slope_stats(group["cnt"].to_numpy())
        rows.append(
            {
                "TP_BUZ_NO": code,
                "TP_BUZ_NM": name,
                "amt_slope_pct_month": amt_s["slope"] / group["amt"].mean() * 100,
                "cnt_slope_pct_month": cnt_s["slope"] / group["cnt"].mean() * 100,
                "amt_slope_r2": amt_s["r2"],
                "cnt_slope_r2": cnt_s["r2"],
                "total_amt_6m": group["amt"].sum(),
                "total_cnt_6m": group["cnt"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("TP_BUZ_NO")


def persistence(panel: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    membership = slopes.loc[slopes["complete_6m"], ["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "h1_candidate", "h2_candidate", "h1_h2_candidate"]]
    rows = []
    for key, group in panel.groupby(["SIDO_NM", "CCG_NM", "TP_BUZ_NO"]):
        group = group.sort_values("STRD_YYMM")
        if group["STRD_YYMM"].nunique() != 6:
            continue
        abs_amt = np.diff(np.log(group["amt"].to_numpy()))
        abs_cnt = np.diff(np.log(group["cnt"].to_numpy()))
        rel_amt = np.diff(np.log(group["relative_amt_share"].to_numpy()))
        rel_cnt = np.diff(np.log(group["relative_cnt_share"].to_numpy()))
        abs_flags = ((abs_amt > 0) & (abs_cnt < 0)).tolist()
        rel_flags = ((rel_amt > 0) & (rel_cnt < 0)).tolist()
        rows.append(
            {
                "SIDO_NM": key[0],
                "CCG_NM": key[1],
                "TP_BUZ_NO": key[2],
                "absolute_event_months": int(sum(abs_flags)),
                "absolute_max_consecutive": max_consecutive(abs_flags),
                "relative_event_months": int(sum(rel_flags)),
                "relative_max_consecutive": max_consecutive(rel_flags),
                "relative_event_pattern": "동시 신호 없음"
                if sum(rel_flags) == 0
                else "1개월 단발"
                if sum(rel_flags) == 1
                else "비연속 반복"
                if max_consecutive(rel_flags) == 1
                else "2개월 이상 연속",
            }
        )
    return pd.DataFrame(rows).merge(membership, on=["SIDO_NM", "CCG_NM", "TP_BUZ_NO"], validate="one_to_one")


def representative_cases(slopes: pd.DataFrame, persistence_data: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # H1의 절대 성장착시 후보가 정확히 10개이므로 전부 제시한다.
    candidates = slopes.loc[slopes["h1_candidate"]].merge(
        persistence_data[
            [
                "SIDO_NM",
                "CCG_NM",
                "TP_BUZ_NO",
                "absolute_event_months",
                "absolute_max_consecutive",
                "relative_event_months",
                "relative_max_consecutive",
                "relative_event_pattern",
            ]
        ],
        on=["SIDO_NM", "CCG_NM", "TP_BUZ_NO"],
        validate="one_to_one",
    )
    candidates["absolute_divergence_pp_month"] = candidates["amt_slope_pct_month"] - candidates["cnt_slope_pct_month"]
    candidates["relative_divergence_pp_month"] = candidates["relative_amt_slope_pct_month"] - candidates["relative_cnt_slope_pct_month"]
    representatives = (
        candidates.sort_values(["total_cnt_6m", "absolute_divergence_pp_month"], ascending=False)
        .head(10)
        .copy()
    )
    representatives.insert(0, "case_id", range(1, len(representatives) + 1))

    monthly = panel.merge(
        representatives[["case_id", "SIDO_NM", "CCG_NM", "TP_BUZ_NO", "TP_BUZ_NM"]],
        on=["SIDO_NM", "CCG_NM", "TP_BUZ_NO", "TP_BUZ_NM"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["case_id", "STRD_YYMM"])
    monthly["amt_index"] = monthly.groupby("case_id")["amt"].transform(lambda x: x / x.iloc[0] * 100)
    monthly["cnt_index"] = monthly.groupby("case_id")["cnt"].transform(lambda x: x / x.iloc[0] * 100)
    monthly["relative_amt_index"] = monthly.groupby("case_id")["relative_amt_share"].transform(lambda x: x / x.iloc[0] * 100)
    monthly["relative_cnt_index"] = monthly.groupby("case_id")["relative_cnt_share"].transform(lambda x: x / x.iloc[0] * 100)
    return representatives, monthly


def scatter_plot(slopes: pd.DataFrame, basis: str, path: Path) -> None:
    valid = slopes.loc[slopes["complete_6m"]]
    if basis == "absolute":
        x, y, color = "cnt_slope_pct_month", "amt_slope_pct_month", "absolute_type"
        title = "6개월 절대 추세: AMT slope × CNT slope"
        xlabel, ylabel = "CNT 월평균 추세(%)", "AMT 월평균 추세(%)"
    else:
        x, y, color = "relative_cnt_slope_pct_month", "relative_amt_slope_pct_month", "relative_type"
        title = "전국 동일업종 대비 상대 추세"
        xlabel, ylabel = "Relative CNT 월평균 추세(%)", "Relative AMT 월평균 추세(%)"
    fig, ax = plt.subplots(figsize=(9, 7))
    for type_name in QUADRANTS.values():
        group = valid.loc[valid[color] == type_name]
        ax.scatter(group[x], group[y], s=20, alpha=0.55, c=COLORS[type_name], label=f"{type_name} ({len(group):,})", edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(0, color="black", linewidth=0.9)
    xlim = np.nanpercentile(valid[x], [1, 99])
    ylim = np.nanpercentile(valid[y], [1, 99])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def industry_plot(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.sort_values("h2_rate_pct")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot["TP_BUZ_NM"], plot["h2_rate_pct"], color="#D55E00", alpha=0.85)
    ax.set_xlabel("상대 성장착시 후보 비율(%)")
    ax.set_title("업종별 전국 동일업종 대비 성장착시 후보 비율")
    for y, value in enumerate(plot["h2_rate_pct"]):
        ax.text(value + 0.4, y, f"{value:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def age_plot(summary: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(summary))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width / 2, summary["illusion_cnt_slope_median"], width, label="성장착시 후보", color="#D55E00")
    axes[0].bar(x + width / 2, summary["healthy_cnt_slope_median"], width, label="건강성장", color="#009E73")
    axes[0].set_title("연령대별 CNT 추세 중앙값")
    axes[0].set_ylabel("월평균 추세(%)")
    axes[1].bar(x - width / 2, summary["illusion_share_slope_median"], width, label="성장착시 후보", color="#D55E00")
    axes[1].bar(x + width / 2, summary["healthy_share_slope_median"], width, label="건강성장", color="#009E73")
    axes[1].set_title("연령대별 CNT 점유율 추세 중앙값")
    axes[1].set_ylabel("퍼센트포인트/월")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, summary["연령대"], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def structure_plot(shifts: pd.DataFrame, path: Path) -> None:
    plot = shifts.loc[shifts["amt_cnt_both_shift"]].head(15).copy().sort_values("amt_share_fitted_change_pp_생활구매형")
    labels = plot["SIDO_NM"].str.replace("특별자치도", "").str.replace("광역시", "") + " " + plot["CCG_NM"]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(plot))
    ax.barh(y, plot["amt_share_fitted_change_pp_생활구매형"], color="#0072B2", label="생활구매형")
    ax.barh(y, plot["amt_share_fitted_change_pp_외식형"], color="#D55E00", label="외식형")
    ax.set_yticks(y, labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("6개월 적합 점유율 변화(%p)")
    ax.set_title("외식형→생활구매형 소비구조 이동 상위 지역")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def persistence_plot(persistence_data: pd.DataFrame, path: Path) -> None:
    order = ["동시 신호 없음", "1개월 단발", "비연속 반복", "2개월 이상 연속"]
    counts = persistence_data.loc[persistence_data["h2_candidate"], "relative_event_pattern"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#999999", "#E69F00", "#56B4E9", "#D55E00"])
    ax.set_ylabel("시군구×업종 조합수")
    ax.set_title("상대 성장착시 후보 134개의 월별 지속성")
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, labels=[f"{value}\n({value / counts.sum() * 100:.1f}%)" for value in counts.values], padding=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def representative_plot(representatives: pd.DataFrame, monthly: pd.DataFrame, path: Path) -> None:
    n = len(representatives)
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, rows * 3.1), squeeze=False)
    for ax, case in zip(axes.flat, representatives.itertuples(index=False)):
        group = monthly.loc[monthly["case_id"] == case.case_id]
        labels = group["STRD_YYMM"].astype(str).str[-2:] + "월"
        ax.plot(labels, group["amt_index"], marker="o", color="#D55E00", label="AMT 지수")
        ax.plot(labels, group["cnt_index"], marker="o", color="#0072B2", label="CNT 지수")
        ax.axhline(100, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(f"{case.case_id}. {case.SIDO_NM} {case.CCG_NM} · {case.TP_BUZ_NM}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("대표 성장착시 후보: 1월=100 AMT/CNT 추이", y=1.005, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    slopes: pd.DataFrame,
    panel: pd.DataFrame,
    quadrants: pd.DataFrame,
    industries: pd.DataFrame,
    age_summary: pd.DataFrame,
    age_combo: pd.DataFrame,
    shifts: pd.DataFrame,
    persistence_data: pd.DataFrame,
    representatives: pd.DataFrame,
    representative_monthly: pd.DataFrame,
    national_trends: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "01_slope_table_all_combinations.csv": slopes,
        "02_monthly_panel_observed.csv": panel,
        "03_quadrant_summary.csv": quadrants,
        "04_industry_illusion_rates.csv": industries,
        "05_age_group_comparison.csv": age_summary,
        "06_age_combo_metrics.csv": age_combo,
        "07_region_consumption_structure_shift.csv": shifts,
        "08_persistence_by_combination.csv": persistence_data,
        "09_representative_cases.csv": representatives,
        "10_representative_monthly_trends.csv": representative_monthly,
        "11_national_industry_trends.csv": national_trends,
    }
    for filename, frame in tables.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT / "growth_illusion_analysis_tables.xlsx", engine="openpyxl") as writer:
        for filename, frame in tables.items():
            sheet = filename.split(".")[0][:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)


def main() -> None:
    configure_plot()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, dtype={"GENDER_CD": "string", "AGE_CD": "string"})
    data["TP_BUZ_NM"] = data["TP_BUZ_NO"].map(INDUSTRY_NAMES)

    slopes, panel = build_slope_table(data)
    quadrants = quadrant_summary(slopes)
    industries, industry_test = industry_summary(slopes)
    age_summary, age_combo = age_comparison(data, slopes)
    shifts = structure_shift(data)
    national_trends = national_industry_trends(data)
    persistence_data = persistence(panel, slopes)
    representatives, representative_monthly = representative_cases(slopes, persistence_data, panel)

    save_outputs(
        slopes,
        panel,
        quadrants,
        industries,
        age_summary,
        age_combo,
        shifts,
        persistence_data,
        representatives,
        representative_monthly,
        national_trends,
    )
    scatter_plot(slopes, "absolute", FIG / "01_absolute_amt_cnt_slope_scatter.png")
    scatter_plot(slopes, "relative", FIG / "02_relative_amt_cnt_slope_scatter.png")
    industry_plot(industries, FIG / "03_industry_illusion_rate.png")
    age_plot(age_summary, FIG / "04_age_comparison.png")
    representative_plot(representatives, representative_monthly, FIG / "05_representative_cases.png")
    structure_plot(shifts, FIG / "06_consumption_structure_shift.png")
    persistence_plot(persistence_data, FIG / "07_relative_candidate_persistence.png")

    valid = slopes.loc[slopes["complete_6m"]]
    h2 = persistence_data.loc[persistence_data["h2_candidate"]]
    h1 = persistence_data.loc[persistence_data["h1_candidate"]]
    metrics = {
        "raw_rows": int(len(data)),
        "regions": int(data[["SIDO_NM", "CCG_NM"]].drop_duplicates().shape[0]),
        "theoretical_combinations": int(len(slopes)),
        "complete_combinations": int(valid.shape[0]),
        "h1_count": int(valid["h1_candidate"].sum()),
        "h1_rate_pct": float(valid["h1_candidate"].mean() * 100),
        "h2_count": int(valid["h2_candidate"].sum()),
        "h2_rate_pct": float(valid["h2_candidate"].mean() * 100),
        "h1_h2_count": int(valid["h1_h2_candidate"].sum()),
        "h1_h2_rate_pct": float(valid["h1_h2_candidate"].mean() * 100),
        "h1_both_slopes_p_lt_005": int(
            (valid["h1_candidate"] & valid["amt_slope_p"].lt(0.05) & valid["cnt_slope_p"].lt(0.05)).sum()
        ),
        "h2_both_slopes_p_lt_005": int(
            (valid["h2_candidate"] & valid["relative_amt_p"].lt(0.05) & valid["relative_cnt_p"].lt(0.05)).sum()
        ),
        "industry_test": industry_test,
        "structure_amt_shift_count": int(shifts["amt_external_to_life"].sum()),
        "structure_cnt_shift_count": int(shifts["cnt_external_to_life"].sum()),
        "structure_both_shift_count": int(shifts["amt_cnt_both_shift"].sum()),
        "h2_persistence_counts": {str(k): int(v) for k, v in h2["relative_event_pattern"].value_counts().items()},
        "h2_persistent_2plus_count": int((h2["relative_max_consecutive"] >= 2).sum()),
        "h2_persistent_2plus_rate_pct": float((h2["relative_max_consecutive"] >= 2).mean() * 100),
        "h1_persistence_counts": {str(k): int(v) for k, v in h1["absolute_event_months"].value_counts().sort_index().items()},
        "h1_absolute_persistent_2plus_count": int((h1["absolute_max_consecutive"] >= 2).sum()),
    }
    (OUT / "analysis_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
