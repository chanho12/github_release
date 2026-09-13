"""Three-month sequence features suited to aggregated region-industry data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import KEYS


SNAPSHOT_NUMERIC = [
    "amt_log_growth_1m", "cnt_log_growth_1m", "ticket_log_growth_1m",
    "customer_hhi", "customer_hhi_log_growth_1m", "largest_segment_share",
    "supply_count", "throughput_log_growth_1m",
]
SEQUENCE_BASE = ["log_amt", "log_cnt", "log_ticket", "customer_hhi", "largest_segment_share"]


def add_temporal_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel.sort_values(KEYS + ["STRD_YYMM"]).copy()
    out["log_amt"] = np.log1p(out["amt"])
    out["log_cnt"] = np.log1p(out["cnt"])
    out["log_ticket"] = np.log1p(out["ticket"])
    temporal = []
    grouped = out.groupby(KEYS, sort=False)
    for column in SEQUENCE_BASE:
        for lag in [1, 2]:
            name = f"{column}_lag{lag}"
            out[name] = grouped[column].shift(lag)
            temporal.append(name)
        mean_name, std_name = f"{column}_3m_mean", f"{column}_3m_std"
        out[mean_name] = grouped[column].transform(lambda s: s.rolling(3, min_periods=3).mean())
        out[std_name] = grouped[column].transform(lambda s: s.rolling(3, min_periods=3).std())
        temporal.extend([mean_name, std_name])
        slope_name = f"{column}_3m_slope"
        out[slope_name] = (out[column] - out[f"{column}_lag2"]) / 2
        temporal.append(slope_name)
    for column in ["amt_log_growth_1m", "cnt_log_growth_1m", "ticket_log_growth_1m",
                   "customer_hhi_log_growth_1m"]:
        lag_name = f"{column}_lag1"
        acceleration = f"{column}_acceleration"
        out[lag_name] = grouped[column].shift(1)
        out[acceleration] = out[column] - out[lag_name]
        temporal.extend([lag_name, acceleration])
    return out, SNAPSHOT_NUMERIC + temporal
