#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-dataset/외부데이터/폐업_전국}"
RAW_DIR="$ROOT_DIR/raw/localdata"

mkdir -p \
  "$RAW_DIR/general_restaurants" \
  "$RAW_DIR/rest_cafes" \
  "$RAW_DIR/bakery" \
  "$RAW_DIR/large_scale_retail" \
  "$RAW_DIR/food_retail" \
  "$RAW_DIR/other_candidates"

download_localdata() {
  local slug="$1"
  local target="$2"
  local temp_file="${target}.part"

  curl --fail --location --retry 3 \
    --user-agent "Mozilla/5.0" \
    --referer "https://file.localdata.go.kr/file/${slug}/info" \
    --cookie "JSESSIONID=public-download" \
    "https://file.localdata.go.kr/file/download/${slug}/info" \
    --output "$temp_file"

  if [[ ! -s "$temp_file" ]]; then
    echo "Empty download: $slug" >&2
    exit 1
  fi
  mv "$temp_file" "$target"
}

download_localdata general_restaurants "$RAW_DIR/general_restaurants/일반음식점.csv"
download_localdata rest_cafes "$RAW_DIR/rest_cafes/휴게음식점.csv"
download_localdata bakeries "$RAW_DIR/bakery/제과점.csv"
download_localdata large_scale_retail_stores "$RAW_DIR/large_scale_retail/생활_대규모점포.csv"
download_localdata other_food_retailers "$RAW_DIR/food_retail/식품판매업_기타.csv"
download_localdata over_the_counter_medicine_stores "$RAW_DIR/other_candidates/안전상비의약품판매업소.csv"
download_localdata tobacco_retailers "$RAW_DIR/other_candidates/담배소매업.csv"

echo "LOCALDATA nationwide files downloaded under: $RAW_DIR"
echo "Run: python3 scripts/build_national_closure_dataset.py"
