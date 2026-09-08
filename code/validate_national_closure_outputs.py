#!/usr/bin/env python3
"""Validate the reconstructed closure panel and write checks plus a file manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "STRD_YYMM",
    "SIDO_NM",
    "CCG_NM",
    "TPBUZ_NM",
    "source",
    "mapping_confidence",
    "active_count_at_start",
    "active_count_at_end",
    "opening_store_count",
    "closure_store_count",
    "net_store_change",
    "closure_rate_start",
    "closure_rate_avg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="dataset/외부데이터/폐업_전국",
        help="Package root",
    )
    args = parser.parse_args()
    root = Path(args.root)
    panel_path = root / "processed/202601_202606_national_closure_panel_bc_mapping.csv"
    panel = pd.read_csv(panel_path)

    key = ["STRD_YYMM", "SIDO_NM", "CCG_NM", "TPBUZ_NM", "source"]
    expected_end = (
        panel["active_count_at_start"]
        + panel["opening_store_count"]
        - panel["closure_store_count"]
    )
    expected_start_rate = np.where(
        panel["active_count_at_start"] > 0,
        panel["closure_store_count"] / panel["active_count_at_start"],
        np.nan,
    )
    expected_avg_rate = np.where(
        (panel["active_count_at_start"] + panel["active_count_at_end"]) > 0,
        panel["closure_store_count"]
        / ((panel["active_count_at_start"] + panel["active_count_at_end"]) / 2),
        np.nan,
    )

    checks = [
        ("panel_exists", panel_path.exists(), str(panel_path)),
        ("required_columns", REQUIRED_COLUMNS.issubset(panel.columns), str(sorted(REQUIRED_COLUMNS - set(panel.columns)))),
        ("months_are_202601_202606", sorted(panel["STRD_YYMM"].unique().tolist()) == list(range(202601, 202607)), str(sorted(panel["STRD_YYMM"].unique()))),
        ("mapping_only_high_medium", set(panel["mapping_confidence"].unique()) <= {"HIGH", "MEDIUM"}, str(sorted(panel["mapping_confidence"].unique()))),
        ("panel_key_unique", not panel.duplicated(key).any(), str(int(panel.duplicated(key).sum()))),
        ("accounting_identity", bool(np.allclose(panel["active_count_at_end"], expected_end, equal_nan=True)), str(int((panel["active_count_at_end"] != expected_end).sum()))),
        ("start_rate_formula", bool(np.allclose(panel["closure_rate_start"], expected_start_rate, equal_nan=True)), "closure/start"),
        ("average_rate_formula", bool(np.allclose(panel["closure_rate_avg"], expected_avg_rate, equal_nan=True)), "closure/average(start,end)"),
        ("no_negative_counts", bool((panel[["active_count_at_start", "active_count_at_end", "opening_store_count", "closure_store_count"]] >= 0).all().all()), "counts >= 0"),
        ("high_industry_count", panel.loc[panel.mapping_confidence.eq("HIGH"), "TPBUZ_NM"].nunique() == 3, str(panel.loc[panel.mapping_confidence.eq("HIGH"), "TPBUZ_NM"].nunique())),
        ("high_medium_industry_count", panel["TPBUZ_NM"].nunique() == 9, str(panel["TPBUZ_NM"].nunique())),
    ]
    check_df = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    check_df.to_csv(root / "audit/validation_checks.csv", index=False, encoding="utf-8-sig")

    manifest_rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "output_manifest.csv":
            continue
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        root / "metadata/output_manifest.csv", index=False, encoding="utf-8-sig"
    )

    if not check_df["passed"].all():
        raise SystemExit("Validation failed; inspect audit/validation_checks.csv")
    print(f"PASS: {len(check_df)} checks; {len(panel):,} panel rows")


if __name__ == "__main__":
    main()
