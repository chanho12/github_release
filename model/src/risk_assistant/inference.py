"""Resolve a current region-industry row and produce multilabel probabilities."""
from __future__ import annotations

from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pandas as pd

from .dataset import build_monthly_panel
from .modeling import predict_frame
from .paths import ARTIFACT_ROOT
from .temporal import add_temporal_features


class RegionRiskPredictor:
    def __init__(self, artifact: Path = ARTIFACT_ROOT / "temporal_3m_multilabel.joblib",
                 snapshot_artifact: Path = ARTIFACT_ROOT / "risk_multilabel.joblib"):
        self.artifact = artifact
        self.snapshot_artifact = snapshot_artifact
        self.temporal_bundle = joblib.load(artifact)
        self.panel, _ = add_temporal_features(build_monthly_panel())
        self.panel = self.panel.dropna(subset=["log_amt_lag2"])

    def predict(self, sido: str, sigungu: str, industry: str) -> dict:
        # macOS often stores Hangul as NFD while CSV values are NFC.
        sido = unicodedata.normalize("NFC", str(sido).strip())
        sigungu = unicodedata.normalize("NFC", str(sigungu).strip())
        industry = unicodedata.normalize("NFC", "".join(str(industry).split()))
        rows = self.panel[
            self.panel["SIDO_NM"].eq(sido) & self.panel["CCG_NM"].eq(sigungu) &
            self.panel["TP_BUZ_NM"].eq(industry)
        ].sort_values("STRD_YYMM")
        if rows.empty:
            raise KeyError(f"No region-industry row: {sido}/{sigungu}/{industry}")
        latest = rows.tail(1)
        # Competition remains a separate exact-supply model; common risks use the 3-month sequence model.
        scored = predict_frame(latest, self.snapshot_artifact).iloc[0]
        bundle = self.temporal_bundle
        x = bundle["preprocessor"].transform(latest[bundle["categorical"] + bundle["numeric"]])
        probabilities = np.column_stack([model.predict_proba(x)[:, 1]
                                         for model in bundle["model"].estimators_])[0]
        for label, probability in zip(bundle["labels"], probabilities):
            scored[f"p_{label}"] = float(probability)
        probability_columns = [f"p_{label}" for label in bundle["labels"]]
        if pd.notna(scored.get("p_competition_pressure")):
            probability_columns.append("p_competition_pressure")
        scored["complex_risk"] = int(sum(float(scored[c]) >= .5 for c in probability_columns) >= 2)
        keep = ["STRD_YYMM", "SIDO_NM", "CCG_NM", "TP_BUZ_NM", "amt", "cnt", "ticket",
                "p_demand_decline", "p_ticket_pressure", "p_customer_concentration",
                "p_competition_pressure", "complex_risk"]
        result = {}
        for key in keep:
            value = scored.get(key)
            if pd.isna(value):
                result[key] = None
            elif hasattr(value, "item"):
                result[key] = value.item()
            else:
                result[key] = value
        return result
