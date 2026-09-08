#!/usr/bin/env python3
"""Extract 2026 Q1-Q2 shop rent and vacancy tables from the official REB workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset/external_raw/reb_commercial/2026Q2_commercial_rent.xlsx"
OUT_DIR = ROOT / "dataset/external_raw/reb_commercial"

SHEETS = {
    "203": ("중대형상가", "vacancy_rate_pct"),
    "204": ("중대형상가", "rent_1000won_m2"),
    "303": ("소규모상가", "vacancy_rate_pct"),
    "304": ("소규모상가", "rent_1000won_m2"),
    "403": ("집합상가", "vacancy_rate_pct"),
    "404": ("집합상가", "rent_1000won_m2"),
}

SIDO_FULL = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
    "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
    "경남": "경상남도", "제주": "제주특별자치도", "전국": "전국",
}


def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for sheet, (property_type, metric) in SHEETS.items():
        frame = pd.read_excel(SOURCE, sheet_name=sheet, header=3)
        frame = frame[["지역구분(1)", "지역구분(2)", "2026.1Q", "2026.2Q"]].copy()
        frame.columns = ["area1", "area2", "2026Q1", "2026Q2"]
        frame["property_type"] = property_type
        frame["metric"] = metric
        pieces.append(frame)

    detail = pd.concat(pieces, ignore_index=True)
    detail["SIDO_NM"] = detail["area1"].map(SIDO_FULL)
    detail["qoq_change"] = detail["2026Q2"] - detail["2026Q1"]
    detail["qoq_change_pct"] = (detail["2026Q2"] / detail["2026Q1"] - 1) * 100
    detail = detail.dropna(subset=["SIDO_NM", "2026Q1", "2026Q2"])

    province = detail[(detail["area2"] == "계") & (detail["SIDO_NM"] != "전국")].copy()
    assert province.groupby(["property_type", "metric"]).size().eq(17).all()
    return detail, province


def main() -> None:
    detail, province = extract()
    detail.to_csv(OUT_DIR / "2026Q1_Q2_commercial_rent_vacancy_detail.csv", index=False, encoding="utf-8-sig")
    province.to_csv(OUT_DIR / "2026Q1_Q2_commercial_rent_vacancy_province.csv", index=False, encoding="utf-8-sig")
    print(f"saved {len(detail):,} detailed rows and {len(province):,} province rows")


if __name__ == "__main__":
    main()
