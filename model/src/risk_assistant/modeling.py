"""Multilabel model training, temporal evaluation, persistence and inference."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .dataset import KEYS, LABELS, build_monthly_panel, derive_labels
from .paths import ARTIFACT_ROOT


CATEGORICAL = ["SIDO_NM", "TP_BUZ_NM"]
NUMERIC = [
    "amt_log_growth_1m", "amt_log_growth_2m", "cnt_log_growth_1m", "cnt_log_growth_2m",
    "ticket_log_growth_1m", "ticket_log_growth_2m", "customer_hhi",
    "customer_hhi_log_growth_1m", "largest_segment_share", "supply_count",
    "supply_log_growth_1m", "throughput_log_growth_1m",
]


def make_estimator() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", RobustScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")),
    ])


def _score(y_true: pd.Series, probability: np.ndarray) -> dict:
    predicted = (probability >= .5).astype(int)
    result = {"n": int(len(y_true)), "positives": int(y_true.sum()),
              "prevalence": float(y_true.mean()), "f1_at_0_5": float(f1_score(y_true, predicted, zero_division=0))}
    result["roc_auc"] = float(roc_auc_score(y_true, probability)) if y_true.nunique() > 1 else None
    result["average_precision"] = float(average_precision_score(y_true, probability)) if y_true.nunique() > 1 else None
    return result


def _predict_probability(estimator: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    # Some macOS Accelerate builds emit a spurious matmul RuntimeWarning here.
    # Treat non-finite predictions as an error instead of silently accepting them.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        probability = estimator.predict_proba(frame)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Model produced a non-finite probability")
    return probability


def train(output_dir: Path = ARTIFACT_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = build_monthly_panel()
    eligible = raw["target_month"].notna() & raw["amt_log_growth_2m"].notna()
    holdout_month = int(raw.loc[eligible, "target_month"].max())
    train_mask = eligible & (raw["target_month"] < holdout_month)
    test_mask = eligible & (raw["target_month"] == holdout_month)
    panel, thresholds = derive_labels(raw, train_mask)

    models, metrics = {}, {}
    predictions = panel.loc[test_mask, ["STRD_YYMM", "target_month"] + KEYS].copy()
    for label in LABELS:
        label_train = train_mask & panel[label].notna()
        label_test = test_mask & panel[label].notna()
        if panel.loc[label_train, label].nunique() < 2:
            metrics[label] = {"status": "SKIPPED_SINGLE_CLASS"}
            continue
        estimator = make_estimator()
        estimator.fit(panel.loc[label_train, CATEGORICAL + NUMERIC], panel.loc[label_train, label].astype(int))
        probability = _predict_probability(estimator, panel.loc[label_test, CATEGORICAL + NUMERIC])
        models[label] = estimator
        metrics[label] = _score(panel.loc[label_test, label].astype(int), probability)
        predictions.loc[label_test[test_mask].to_numpy(), f"p_{label}"] = probability

    bundle = {
        "models": models, "categorical": CATEGORICAL, "numeric": NUMERIC, "labels": list(models),
        "thresholds": thresholds, "holdout_target_month": holdout_month,
        "competition_industries": sorted(panel.loc[train_mask & panel["competition_pressure"].notna(),
                                                        "TP_BUZ_NM"].unique().tolist()),
        "warning": "Only three observable next-month transitions exist. Metrics are prototype diagnostics, not deployment evidence.",
    }
    joblib.dump(bundle, output_dir / "risk_multilabel.joblib")
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    history_columns = (["STRD_YYMM", "target_month"] + KEYS + NUMERIC + LABELS)
    panel.loc[eligible, history_columns].to_csv(
        output_dir / "historical_observations.csv", index=False, encoding="utf-8-sig")
    report = {"holdout_target_month": holdout_month, "train_rows": int(train_mask.sum()),
              "test_rows": int(test_mask.sum()), "thresholds": thresholds, "metrics": metrics,
              "limitations": ["2026-01~06 only", "single-month temporal holdout",
                              "competition label available only for exact supply mappings",
                              "region-industry risk; never an individual-store closure probability"]}
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def predict_frame(frame: pd.DataFrame, artifact: Path = ARTIFACT_ROOT / "risk_multilabel.joblib") -> pd.DataFrame:
    bundle = joblib.load(artifact)
    result = frame.copy()
    for label, estimator in bundle["models"].items():
        result[f"p_{label}"] = _predict_probability(
            estimator, frame[bundle["categorical"] + bundle["numeric"]])
        if label == "competition_pressure":
            applicable = result["TP_BUZ_NM"].isin(bundle.get("competition_industries", []))
            result.loc[~applicable, f"p_{label}"] = np.nan
    probability_columns = [f"p_{x}" for x in bundle["models"]]
    result["complex_risk"] = (result[probability_columns].ge(.5).sum(axis=1) >= 2).astype(int)
    return result
