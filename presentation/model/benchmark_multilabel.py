#!/usr/bin/env python3
"""Compare a forced one-problem classifier with genuine multilabel alternatives."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.dataset import build_monthly_panel, derive_labels
from risk_assistant.modeling import CATEGORICAL, NUMERIC, make_estimator


COMMON_LABELS = ["demand_decline", "ticket_pressure", "customer_concentration"]
OUT = ROOT / "artifacts"


def metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    multi = y_true.sum(axis=1) >= 2
    recovered = ((y_true[multi] == 1) & (y_pred[multi] == 1)).sum()
    positives = (y_true[multi] == 1).sum()
    return {
        "model": name,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "multi_positive_recall": recovered / positives if positives else np.nan,
    }


def main():
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
    np.seterr(all="ignore")
    OUT.mkdir(exist_ok=True)
    raw = build_monthly_panel()
    eligible = raw["target_month"].notna() & raw["amt_log_growth_2m"].notna()
    holdout_month = int(raw.loc[eligible, "target_month"].max())
    train_mask = eligible & (raw["target_month"] < holdout_month)
    test_mask = eligible & (raw["target_month"] == holdout_month)
    panel, definitions = derive_labels(raw, train_mask)
    x_train = panel.loc[train_mask, CATEGORICAL + NUMERIC]
    x_test = panel.loc[test_mask, CATEGORICAL + NUMERIC]
    y_train = panel.loc[train_mask, COMMON_LABELS].astype(int).to_numpy()
    y_test = panel.loc[test_mask, COMMON_LABELS].astype(int).to_numpy()

    # Fit one shared transform so every benchmark sees exactly the same features.
    pipeline = make_estimator()
    preprocessor = pipeline.named_steps["preprocess"]
    z_train = preprocessor.fit_transform(x_train)
    z_test = preprocessor.transform(x_test)

    results = []

    # Baseline explicitly enforces at most one problem. Multi-positive rows lose labels by design.
    forced_train = np.where(y_train.sum(axis=1) == 0, len(COMMON_LABELS), y_train.argmax(axis=1))
    forced = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    forced.fit(z_train, forced_train)
    forced_class = forced.predict(z_test)
    forced_pred = np.zeros_like(y_test)
    for row, klass in enumerate(forced_class):
        if klass < len(COMMON_LABELS):
            forced_pred[row, klass] = 1
    results.append(metrics(y_test, forced_pred, "forced_single_problem_logistic"))

    br_logistic = MultiOutputClassifier(LogisticRegression(
        max_iter=1500, class_weight="balanced", solver="liblinear"))
    br_logistic.fit(z_train, y_train)
    results.append(metrics(y_test, br_logistic.predict(z_test), "binary_relevance_logistic"))

    br_forest = MultiOutputClassifier(RandomForestClassifier(
        n_estimators=250, min_samples_leaf=8, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1))
    br_forest.fit(z_train, y_train)
    results.append(metrics(y_test, br_forest.predict(z_test), "binary_relevance_random_forest"))

    chain = ClassifierChain(LogisticRegression(
        max_iter=1500, class_weight="balanced", solver="liblinear"),
        order="random", random_state=42)
    chain.fit(z_train, y_train)
    results.append(metrics(y_test, chain.predict(z_test).astype(int), "classifier_chain_logistic"))

    result = pd.DataFrame(results).sort_values("micro_f1", ascending=False)
    result.to_csv(OUT / "multilabel_model_benchmark.csv", index=False, encoding="utf-8-sig")

    cardinality = pd.Series(y_test.sum(axis=1)).value_counts().sort_index()
    cooccurrence = pd.DataFrame(y_test.T @ y_test, index=COMMON_LABELS, columns=COMMON_LABELS)
    multi_mask = y_test.sum(axis=1) >= 2
    evidence = {
        "holdout_target_month": holdout_month,
        "labels": COMMON_LABELS,
        "holdout_rows": int(len(y_test)),
        "rows_with_no_problem": int((y_test.sum(axis=1) == 0).sum()),
        "rows_with_one_problem": int((y_test.sum(axis=1) == 1).sum()),
        "rows_with_multiple_problems": int(multi_mask.sum()),
        "multiple_problem_share": float(multi_mask.mean()),
        "mean_label_cardinality": float(y_test.sum(axis=1).mean()),
        "forced_single_theoretical_recall_ceiling_on_multi_rows": float(
            multi_mask.sum() / y_test[multi_mask].sum()) if multi_mask.any() else None,
        "label_cardinality_counts": {str(k): int(v) for k, v in cardinality.items()},
        "cooccurrence": cooccurrence.to_dict(),
        "label_definition": definitions,
        "interpretation": "Co-occurrence establishes target structure; model scores alone do not prove ontology.",
    }
    (OUT / "multilabel_necessity.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"preprocessor": preprocessor, "model": br_logistic, "labels": COMMON_LABELS},
                OUT / "benchmark_best_interpretable_multilabel.joblib")
    print(result.to_string(index=False))
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
