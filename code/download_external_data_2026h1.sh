#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dataset/외부데이터"
KMA_DIR="$OUT_DIR/기상청_기상"
FOREIGN_DIR="$OUT_DIR/법무부_등록외국인"

mkdir -p "$KMA_DIR" "$FOREIGN_DIR"

for month in 202601 202602 202603 202604 202605 202606; do
  curl -L -X POST \
    -d "pubType=SYNM&issDate=$month" \
    "https://data.kma.go.kr/data/common/pubXlsFileDownView.do" \
    -o "$KMA_DIR/${month}_기상월보.xls"
done

curl -L "https://www.immigration.go.kr/bbs/immigration/227/493620/download.do" \
  -o "$FOREIGN_DIR/202603_등록외국인_지역_국적.xlsx"
curl -L "https://www.immigration.go.kr/bbs/immigration/227/497455/download.do" \
  -o "$FOREIGN_DIR/202606_등록외국인_지역_국적.xlsx"

file "$KMA_DIR"/*.xls "$FOREIGN_DIR"/*.xlsx

