#!/usr/bin/env python3
"""Compare snapshot and 3-month temporal multilabel models with rolling-origin tests."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.dataset import build_monthly_panel, derive_labels
from risk_assistant.temporal import SNAPSHOT_NUMERIC, add_temporal_features


LABELS = ["demand_decline", "ticket_pressure", "customer_concentration"]
CATEGORICAL = ["SIDO_NM", "TP_BUZ_NM"]
def preprocessor(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", RobustScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def estimator(kind: str):
    if kind == "logistic":
        base = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    else:
        base = RandomForestClassifier(n_estimators=250, min_samples_leaf=8,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    return MultiOutputClassifier(base)


def score(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    result = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
    }
    auc, ap = [], []
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size > 1:
            auc.append(roc_auc_score(y_true[:, index], probabilities[:, index]))
            ap.append(average_precision_score(y_true[:, index], probabilities[:, index]))
    result["macro_roc_auc"] = float(np.mean(auc))
    result["macro_average_precision"] = float(np.mean(ap))
    return result


def fit_predict(frame, train_mask, test_mask, numeric, kind):
    prep = preprocessor(numeric)
    x_train = prep.fit_transform(frame.loc[train_mask, CATEGORICAL + numeric])
    x_test = prep.transform(frame.loc[test_mask, CATEGORICAL + numeric])
    y_train = frame.loc[train_mask, LABELS].astype(int).to_numpy()
    y_test = frame.loc[test_mask, LABELS].astype(int).to_numpy()
    model = estimator(kind)
    model.fit(x_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probabilities = np.column_stack([m.predict_proba(x_test)[:, 1] for m in model.estimators_])
    predicted = (probabilities >= .5).astype(int)
    return score(y_test, predicted, probabilities), prep, model


def main():
    np.seterr(all="ignore")
    raw, temporal_numeric = add_temporal_features(build_monthly_panel())
    sequence_ready = raw["log_amt_lag2"].notna() & raw["target_month"].notna()
    all_eligible = raw["target_month"].notna()
    labeled, definitions = derive_labels(raw, all_eligible)
    months = sorted(int(x) for x in labeled.loc[sequence_ready, "target_month"].dropna().unique())
    folds = months[1:]  # first available target month supplies the initial training period
    configurations = [
        ("snapshot_logistic", SNAPSHOT_NUMERIC, "logistic"),
        ("temporal_3m_logistic", temporal_numeric, "logistic"),
        ("temporal_3m_random_forest", temporal_numeric, "forest"),
    ]
    rows = []
    for test_month in folds:
        train_mask = sequence_ready & (labeled["target_month"] < test_month)
        test_mask = sequence_ready & (labeled["target_month"] == test_month)
        for name, numeric, kind in configurations:
            values, prep, model = fit_predict(labeled, train_mask, test_mask, numeric, kind)
            rows.append({"test_target_month": test_month, "model": name,
                         "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()), **values})
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "artifacts" / "temporal_model_benchmark.csv", index=False, encoding="utf-8-sig")
    summary = results.groupby("model").agg(
        folds=("test_target_month", "nunique"), micro_f1_mean=("micro_f1", "mean"),
        macro_f1_mean=("macro_f1", "mean"), macro_auc_mean=("macro_roc_auc", "mean"),
        macro_ap_mean=("macro_average_precision", "mean")).reset_index()
    summary.to_csv(ROOT / "artifacts" / "temporal_model_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "unit": "month x region x industry", "history_window_months": 3,
        "forecast_horizon_months": 1, "rolling_origin_test_months": folds,
        "sequence_numeric_features": temporal_numeric, "label_definition": definitions,
        "warning": "Two rolling-origin folds from six months are architecture evidence, not deployment validation.",
    }
    (ROOT / "artifacts" / "temporal_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # After honest backtesting, refit on every observable t->t+1 pair for the next forecast.
    production_prep = preprocessor(temporal_numeric)
    production_x = production_prep.fit_transform(labeled.loc[sequence_ready, CATEGORICAL + temporal_numeric])
    production_y = labeled.loc[sequence_ready, LABELS].astype(int).to_numpy()
    production_model = estimator("logistic")
    production_model.fit(production_x, production_y)
    production_bundle = {
        "preprocessor": production_prep, "model": production_model, "labels": LABELS,
        "numeric": temporal_numeric, "categorical": CATEGORICAL, "history_months": 3,
        "trained_through_target_month": int(labeled.loc[sequence_ready, "target_month"].max()),
        "forecast_from_observation_month": int(labeled.loc[sequence_ready, "STRD_YYMM"].max()),
        "purpose": "region-industry next-month risk, never store closure survival",
    }
    joblib.dump(production_bundle, ROOT / "artifacts" / "temporal_3m_multilabel.joblib")
    print(results.to_string(index=False))
    print("\nMEAN\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
