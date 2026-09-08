#!/usr/bin/env python3
"""Extract the Jan-Mar 2026 regional living-population table from the MOIS PDF.

The source PDF is an official Ministry of the Interior and Safety attachment.  It
contains nine pages per month for the 89 population-decline areas.  Extraction is
kept as code because the portal supplies the table as PDF rather than CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset/external_raw/mois_living_population/2026Q1_living_population_detail.pdf"
OUTPUT = ROOT / "dataset/external_raw/mois_living_population/202601_202603_living_population_by_region.csv"

# Zero-based page indexes.  Each monthly table occupies nine pages.
MONTH_PAGES = {202601: range(51, 60), 202602: range(60, 69), 202603: range(69, 78)}
SIDO_FULL = {
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
}


def number(value: str) -> int:
    return int(value.replace(",", ""))


def extract() -> pd.DataFrame:
    reader = PdfReader(SOURCE)
    records: list[dict] = []

    for month, pages in MONTH_PAGES.items():
        pending_total = None
        pending_row = None
        for page_no in pages:
            text = reader.pages[page_no].extract_text(extraction_mode="layout")
            for line in text.splitlines():
                total_match = re.match(r"^\s*계\s+([\d,]+)", line)
                if total_match:
                    pending_total = number(total_match.group(1))
                    continue

                resident_match = re.match(r"^\s*([가-힣]+)\s+주민등록인구\s+([\d,]+)", line)
                if resident_match:
                    pending_row = {
                        "STRD_YYMM": month,
                        "SIDO_NM": SIDO_FULL[resident_match.group(1)],
                        "living_population": pending_total,
                        "resident_population_pdf": number(resident_match.group(2)),
                    }
                    continue

                stay_match = re.match(r"^\s*(.+?)\s+체류인구\s+([\d,]+)", line)
                if stay_match and pending_row is not None:
                    pending_row["CCG_NM"] = stay_match.group(1).strip()
                    pending_row["stay_population"] = number(stay_match.group(2))
                    continue

                foreign_match = re.match(r"^\s*외국인\s+([\d,]+)", line)
                if foreign_match and pending_row is not None and "stay_population" in pending_row:
                    pending_row["registered_foreigner_population"] = number(foreign_match.group(1))
                    records.append(pending_row)
                    pending_row = None

    result = pd.DataFrame(records)
    expected_columns = [
        "STRD_YYMM",
        "SIDO_NM",
        "CCG_NM",
        "living_population",
        "resident_population_pdf",
        "stay_population",
        "registered_foreigner_population",
    ]
    result = result[expected_columns].sort_values(["STRD_YYMM", "SIDO_NM", "CCG_NM"])

    monthly_counts = result.groupby("STRD_YYMM").size().to_dict()
    assert monthly_counts == {202601: 89, 202602: 89, 202603: 89}, monthly_counts
    assert not result.duplicated(["STRD_YYMM", "SIDO_NM", "CCG_NM"]).any()
    components = (
        result["resident_population_pdf"]
        + result["stay_population"]
        + result["registered_foreigner_population"]
    )
    assert (components == result["living_population"]).all()
    return result


def main() -> None:
    result = extract()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved {len(result):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
