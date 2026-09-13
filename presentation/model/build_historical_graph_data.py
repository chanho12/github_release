#!/usr/bin/env python3
"""Reconstruct historical monthly store dynamics and trend timeline for Neo4j."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.paths import EXTERNAL_ROOT


START = pd.Period("2019-01", freq="M")
END = pd.Period("2026-06", freq="M")


def build_store_history() -> pd.DataFrame:
    path = EXTERNAL_ROOT / "재현용_가공패널" / "폐업원인_정확업종_점포이벤트패널.csv"
    stores = pd.read_csv(path, low_memory=False)
    stores["open_date"] = pd.to_datetime(stores["open_date"], errors="coerce")
    stores["close_date"] = pd.to_datetime(stores["close_date"], errors="coerce")
    keys = ["SIDO_NM", "CCG_NM", "industry"]
    periods = pd.period_range(START, END, freq="M")
    rows = []
    for key, group in stores.groupby(keys, dropna=False):
        baseline_date = START.start_time
        baseline = ((group["open_date"] < baseline_date) &
                    (group["close_date"].isna() | (group["close_date"] >= baseline_date))).sum()
        opened = group["open_date"].dt.to_period("M").value_counts()
        closed = group["close_date"].dt.to_period("M").value_counts()
        active = int(baseline)
        for period in periods:
            opening_count = int(opened.get(period, 0))
            closure_count = int(closed.get(period, 0))
            active_start = active
            active = active + opening_count - closure_count
            rows.append({
                "SIDO_NM": key[0], "CCG_NM": key[1], "TP_BUZ_NM": key[2],
                "STRD_YYMM": int(period.strftime("%Y%m")), "active_start": active_start,
                "openings": opening_count, "closures": closure_count, "active_end": active,
                "net_change": opening_count - closure_count,
                "closure_rate": closure_count / active_start if active_start > 0 else np.nan,
                "churn_rate": (opening_count + closure_count) / active_start if active_start > 0 else np.nan,
            })
    history = pd.DataFrame(rows)
    history.to_csv(EXTERNAL_ROOT / "재현용_가공패널" / "과거_점포동학_201901_202606.csv",
                   index=False, encoding="utf-8-sig")
    return history


def build_trend_timeline() -> pd.DataFrame:
    cases = pd.read_csv(EXTERNAL_ROOT / "시장트렌드_2026" / "실제_마케팅_사례_근거표.csv")
    trend_map = {
        "M01": "quality_value", "M02": "media_to_product", "M03": "seasonal_experience",
        "M04": "hyperlocal_testbed", "M05": "seasonal_calendar", "M06": "quick_commerce",
        "M07": "subscription", "M08": "store_level_crm", "M09": "voice_of_customer",
        "M10": "health_specificity",
    }
    cases["trend_name"] = cases["case_id"].map(trend_map)
    timeline = cases[["case_id", "case_period", "evidence_period", "trend_name", "brand",
                      "actual_action", "reported_result", "source_url", "caution"]]
    timeline.to_csv(EXTERNAL_ROOT / "시장트렌드_2026" / "과거_트렌드_타임라인.csv",
                    index=False, encoding="utf-8-sig")
    return timeline


if __name__ == "__main__":
    stores = build_store_history()
    trends = build_trend_timeline()
    print({"store_history_rows": len(stores), "start": int(stores.STRD_YYMM.min()),
           "end": int(stores.STRD_YYMM.max()), "trend_rows": len(trends)})
