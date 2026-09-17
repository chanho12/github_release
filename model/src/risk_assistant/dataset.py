"""Build a leakage-aware month x region x industry learning table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import BC_PATH, EXTERNAL_ROOT


KEYS = ["SIDO_NM", "CCG_NM", "TP_BUZ_NM"]
LABELS = ["demand_decline", "ticket_pressure", "customer_concentration", "competition_pressure"]


def _safe_log_ratio(current: pd.Series, previous: pd.Series) -> pd.Series:
    return np.log1p(current.clip(lower=0)) - np.log1p(previous.clip(lower=0))


def build_monthly_panel() -> pd.DataFrame:
    bc = pd.read_csv(BC_PATH)
    bc = bc.rename(columns={"TPBUZ_NM": "TP_BUZ_NM", "TPBUZ_NO": "TP_BUZ_NO",
                            "AMT": "amt", "CNT": "cnt"})
    bc["TP_BUZ_NM"] = bc["TP_BUZ_NM"].astype(str).str.replace(r"\s+", "", regex=True)
    bc["STRD_YYMM"] = pd.to_numeric(bc["STRD_YYMM"], errors="coerce").astype("Int64")
    bc["segment"] = bc["GENDER_CD"].astype(str) + "_" + bc["AGE_CD"].astype(str)

    base = (bc.groupby(["STRD_YYMM"] + KEYS + ["TP_BUZ_NO"], as_index=False)
              .agg(amt=("amt", "sum"), cnt=("cnt", "sum")))
    segment = (bc.groupby(["STRD_YYMM"] + KEYS + ["segment"], as_index=False)["cnt"].sum())
    segment["share"] = segment["cnt"] / segment.groupby(["STRD_YYMM"] + KEYS)["cnt"].transform("sum")
    hhi = (segment.assign(sq=lambda x: x["share"] ** 2)
                  .groupby(["STRD_YYMM"] + KEYS, as_index=False)
                  .agg(customer_hhi=("sq", "sum"), largest_segment_share=("share", "max")))
    panel = base.merge(hhi, on=["STRD_YYMM"] + KEYS, how="left")
    panel["ticket"] = panel["amt"] / panel["cnt"].replace(0, np.nan)
    panel = panel.sort_values(KEYS + ["STRD_YYMM"]).reset_index(drop=True)

    for column in ["amt", "cnt", "ticket", "customer_hhi"]:
        group = panel.groupby(KEYS, sort=False)[column]
        panel[f"{column}_log_growth_1m"] = _safe_log_ratio(panel[column], group.shift(1))
        panel[f"{column}_log_growth_2m"] = _safe_log_ratio(panel[column], group.shift(2)) / 2

    supply_path = EXTERNAL_ROOT / "통합패널" / "202601_202606_BC_정확업종_수요공급패널.csv"
    supply = pd.read_csv(supply_path).rename(columns={"TP_BUZ_NM_CLEAN": "TP_BUZ_NM"})
    supply = supply[["STRD_YYMM"] + KEYS + ["supply_count"]].drop_duplicates()
    panel = panel.merge(supply, on=["STRD_YYMM"] + KEYS, how="left")
    panel["amt_per_supply_proxy"] = panel["amt"] / panel["supply_count"].replace(0, np.nan)
    panel["supply_log_growth_1m"] = panel.groupby(KEYS, sort=False)["supply_count"].transform(
        lambda s: _safe_log_ratio(s, s.shift(1)))
    panel["throughput_log_growth_1m"] = panel.groupby(KEYS, sort=False)["amt_per_supply_proxy"].transform(
        lambda s: _safe_log_ratio(s, s.shift(1)))

    # Targets always refer to t+1. No future column is allowed in model inputs.
    for column in ["amt_log_growth_1m", "cnt_log_growth_1m", "ticket_log_growth_1m",
                   "customer_hhi", "supply_count", "supply_log_growth_1m", "throughput_log_growth_1m"]:
        panel[f"future_{column}"] = panel.groupby(KEYS, sort=False)[column].shift(-1)
    panel["future_hhi_change"] = panel["future_customer_hhi"] - panel["customer_hhi"]
    panel["target_month"] = panel.groupby(KEYS, sort=False)["STRD_YYMM"].shift(-1).astype("Int64")
    return panel


def derive_labels(panel: pd.DataFrame, train_target_months: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Create future peer-relative outcomes; the argument documents the train boundary."""
    out = panel.copy()
    peer = out.groupby(["target_month", "TP_BUZ_NM"], dropna=False)
    ranks = {
        column: peer[column].rank(pct=True, method="average")
        for column in ["future_cnt_log_growth_1m", "future_amt_log_growth_1m",
                       "future_ticket_log_growth_1m", "future_hhi_change",
                       "future_supply_count", "future_throughput_log_growth_1m"]
    }
    out["demand_decline"] = ((ranks["future_cnt_log_growth_1m"] <= .25) &
                              (ranks["future_amt_log_growth_1m"] <= .25)).astype(int)
    out["ticket_pressure"] = (ranks["future_ticket_log_growth_1m"] <= .25).astype(int)
    out["customer_concentration"] = (ranks["future_hhi_change"] >= .75).astype(int)
    valid_supply = out["future_supply_count"].notna() & out["future_throughput_log_growth_1m"].notna()
    out["competition_pressure"] = np.where(
        valid_supply,
        ((ranks["future_supply_count"] >= .75) &
         (ranks["future_throughput_log_growth_1m"] <= .25)).astype(int),
        np.nan,
    )
    thresholds = {
        "definition": "within each target_month x industry peer percentile",
        "demand_decline": "future CNT and AMT growth both <= Q25",
        "ticket_pressure": "future ticket growth <= Q25",
        "customer_concentration": "future HHI change >= Q75",
        "competition_pressure": "future supply level >= Q75 and throughput growth <= Q25",
        "train_boundary_rows": int(train_target_months.sum()),
    }
    return out, thresholds
