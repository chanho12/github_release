"""BC 공모전 데이터와 국세청 월간 생활업종 사업자 현황을 결합한다.

실행:
    python3 scripts/build_nts_closure_panel.py

출력:
    dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx

주의:
    - 국세청의 *****는 소수 값 마스킹이므로 0으로 대체하지 않는다.
    - '폐업강도(%)' = 당월 폐업사업자 / 전월 가동사업자 * 100이다.
      공식 통계명 '폐업률'이 아니므로 분석에서 명칭을 구분해야 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parents[1]
BC_PATH = ROOT / "dataset" / "ABP_CONTEST_DATA.csv"
OUTPUT_DIR = ROOT / "dataset" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "BC_국세청_폐업지표_202601_202606.xlsx"
NTS_SHEET = "3-2.생활업종 사업자 현황(지역별,업종별)"


NTS_COLUMNS = [
    "시도",
    "시군구",
    "국세청업종",
    "가동_당월",
    "가동_전월",
    "가동_전월대비_증감률",
    "가동_전년동월",
    "가동_전년동월대비_증감률",
    "신규_당월",
    "신규_전월",
    "신규_전월대비_증감률",
    "신규_전년동월",
    "신규_전년동월대비_증감률",
    "폐업_당월",
    "폐업_전월",
    "폐업_전월대비_증감률",
    "폐업_전년동월",
    "폐업_전년동월대비_증감률",
]

COUNT_COLUMNS = [
    "가동_당월",
    "가동_전월",
    "가동_전년동월",
    "신규_당월",
    "신규_전월",
    "신규_전년동월",
    "폐업_당월",
    "폐업_전월",
    "폐업_전년동월",
]

NTS_TO_ANALYSIS = {
    "편의점": "편의점",
    "슈퍼마켓": "슈퍼마켓",
    "한식음식점": "한식통합",
    "일식음식점": "일식회집",
    "중식음식점": "중국음식",
    "기타외국식음식점": "서양음식",
    "분식점": "스넥",
    "제과점": "제과점",
}

BC_TO_ANALYSIS = {
    4004: "대형할인점",
    4010: "편의점",
    4020: "슈퍼마켓",
    8001: "한식통합",
    8002: "한식통합",
    8003: "한식통합",
    8004: "일식회집",
    8005: "중국음식",
    8006: "서양음식",
    8021: "스넥",
    8301: "제과점",
}


def find_nts_files() -> list[Path]:
    files = sorted((ROOT / "dataset").glob("*/월간 지역 경제지표(2026년 *월).xlsx"))
    if not files:
        raise FileNotFoundError("dataset 하위 폴더에서 국세청 월간 지역 경제지표 파일을 찾지 못했습니다.")
    return files


def month_from_filename(path: Path) -> int:
    match = re.search(r"\((\d{4})년\s*(\d{1,2})월\)", path.name)
    if not match:
        raise ValueError(f"파일명에서 기준월을 읽을 수 없습니다: {path.name}")
    return int(match.group(1)) * 100 + int(match.group(2))


def read_nts_file(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=NTS_SHEET, header=[5, 6], dtype=object)
    if frame.shape[1] != len(NTS_COLUMNS):
        raise ValueError(f"{path.name}: 예상 18개 컬럼이지만 {frame.shape[1]}개입니다.")
    frame.columns = NTS_COLUMNS
    frame.insert(0, "기준년월", month_from_filename(path))
    frame.insert(1, "원본파일", path.name)

    for column in NTS_COLUMNS[3:]:
        raw = frame[column].astype("string").str.strip()
        if column in COUNT_COLUMNS:
            frame[f"{column}_마스킹"] = raw.eq("*****")
        frame[column] = pd.to_numeric(raw.replace({"*****": pd.NA, "-": pd.NA}), errors="coerce")

    denominator = frame["가동_전월"].astype("Float64").where(frame["가동_전월"].gt(0))
    frame["폐업강도_pct"] = frame["폐업_당월"].astype("Float64").div(denominator).mul(100)
    frame["신규강도_pct"] = frame["신규_당월"].astype("Float64").div(denominator).mul(100)
    frame["신규_폐업_차이"] = frame["신규_당월"] - frame["폐업_당월"]
    return frame


def load_nts() -> pd.DataFrame:
    return pd.concat([read_nts_file(path) for path in find_nts_files()], ignore_index=True)


def load_bc() -> pd.DataFrame:
    bc = pd.read_csv(BC_PATH, encoding="utf-8-sig")
    bc["TP_BUZ_NO"] = pd.to_numeric(bc["TP_BUZ_NO"], errors="raise").astype(int)
    bc["연결업종"] = bc["TP_BUZ_NO"].map(BC_TO_ANALYSIS)
    if bc["연결업종"].isna().any():
        missing = sorted(bc.loc[bc["연결업종"].isna(), "TP_BUZ_NO"].unique())
        raise ValueError(f"업종 매핑에 없는 BC 코드: {missing}")

    # 1~5월 파일은 광주·전남을 별도로, 6월 파일은 통합 명칭으로 제공한다.
    bc["연결시도"] = bc["SIDO_NM"]
    june_integrated = bc["STRD_YYMM"].eq(202606) & bc["SIDO_NM"].isin(["광주광역시", "전라남도"])
    bc.loc[june_integrated, "연결시도"] = "전남광주통합특별시"
    return (
        bc.groupby(["STRD_YYMM", "연결시도", "연결업종"], as_index=False)
        .agg(BC_매출액=("amt", "sum"), BC_결제건수=("cnt", "sum"))
        .rename(columns={"STRD_YYMM": "기준년월"})
    )


def build_integrated(ntsd: pd.DataFrame, bc: pd.DataFrame) -> pd.DataFrame:
    nts_target = ntsd.loc[
        ntsd["시군구"].eq("합계")
        & ntsd["시도"].ne("전국")
        & ntsd["국세청업종"].isin(NTS_TO_ANALYSIS)
    ].copy()
    nts_target["연결업종"] = nts_target["국세청업종"].map(NTS_TO_ANALYSIS)
    nts_target = nts_target.rename(columns={"시도": "연결시도"})

    keep = [
        "기준년월",
        "연결시도",
        "연결업종",
        "국세청업종",
        "가동_당월",
        "가동_전월",
        "신규_당월",
        "폐업_당월",
        "폐업강도_pct",
        "신규강도_pct",
        "신규_폐업_차이",
        "가동_당월_마스킹",
        "신규_당월_마스킹",
        "폐업_당월_마스킹",
    ]
    return (
        bc.merge(nts_target[keep], on=["기준년월", "연결시도", "연결업종"], how="left", validate="many_to_one")
        .sort_values(["기준년월", "연결시도", "연결업종"])
        .reset_index(drop=True)
    )


def mapping_table() -> pd.DataFrame:
    rows = [
        (4004, "대형할인점", "대형할인점", "", "연결 불가", "국세청 100대 생활업종에 직접 대응 업종 없음"),
        (4010, "편 의 점", "편의점", "편의점", "정확", ""),
        (4020, "슈퍼 마켓", "슈퍼마켓", "슈퍼마켓", "정확", ""),
        (8001, "일반한식", "한식통합", "한식음식점", "통합", "BC 8001·8002·8003을 합산"),
        (8002, "갈비전문점", "한식통합", "한식음식점", "통합", "BC 8001·8002·8003을 합산"),
        (8003, "한정식", "한식통합", "한식음식점", "통합", "BC 8001·8002·8003을 합산"),
        (8004, "일식회집", "일식회집", "일식음식점", "근사", "명칭 범위가 완전히 같지 않을 수 있음"),
        (8005, "중국음식", "중국음식", "중식음식점", "근사", "명칭 차이"),
        (8006, "서양음식", "서양음식", "기타외국식음식점", "포괄", "국세청 범위가 더 넓음"),
        (8021, "스넥", "스넥", "분식점", "근사", "명칭 차이"),
        (8301, "제 과 점", "제과점", "제과점", "정확", ""),
    ]
    return pd.DataFrame(rows, columns=["BC업종코드", "BC업종명", "연결업종", "국세청업종", "매핑수준", "주의사항"])


def quality_table(ntsd: pd.DataFrame, integrated: pd.DataFrame) -> pd.DataFrame:
    local = ntsd.loc[ntsd["시군구"].ne("합계")]
    province = ntsd.loc[ntsd["시군구"].eq("합계") & ntsd["시도"].ne("전국")]
    target_province = province.loc[province["국세청업종"].isin(NTS_TO_ANALYSIS)]
    rows = [
        ("국세청 원본 행수", len(ntsd), "6개월 합계"),
        ("기준월 수", ntsd["기준년월"].nunique(), ""),
        ("국세청 업종 수", ntsd["국세청업종"].nunique(), "'업종 전체' 포함"),
        ("시도 대상업종 폐업 마스킹률(%)", target_province["폐업_당월_마스킹"].mean() * 100, "주 분석 단위"),
        ("시군구 전체업종 폐업 마스킹률(%)", local["폐업_당월_마스킹"].mean() * 100, "세부 지역은 마스킹이 많음"),
        ("BC·국세청 통합 행수", len(integrated), "대형할인점 포함"),
        ("BC·국세청 폐업지표 연결률(%)", integrated["국세청업종"].notna().mean() * 100, "대형할인점은 미연결"),
    ]
    return pd.DataFrame(rows, columns=["항목", "값", "해석"])


def readme_table() -> pd.DataFrame:
    rows = [
        ("폐업강도_pct", "폐업_당월 / 가동_전월 * 100", "공식 폐업률이 아닌 분석용 지표"),
        ("신규강도_pct", "신규_당월 / 가동_전월 * 100", "월간 진입 규모"),
        ("마스킹 플래그", "원본값이 *****이면 TRUE", "결측값을 0으로 간주하지 말 것"),
        ("주 분석 단위", "기준년월 × 연결시도 × 연결업종", "시군구 단위는 폐업값 마스킹이 매우 많음"),
        ("지역 주의", "202606에만 광주광역시+전라남도→전남광주통합특별시", "국세청은 1~5월 별도, 6월 통합 명칭을 사용"),
        ("업종 주의", "대형할인점", "국세청 생활업종에 직접 대응 없음"),
    ]
    return pd.DataFrame(rows, columns=["항목", "정의/처리", "주의"])


def style_workbook(writer: pd.ExcelWriter) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in worksheet.columns:
            sample = list(column_cells)[:200]
            width = min(max(len(str(cell.value)) if cell.value is not None else 0 for cell in sample) + 2, 32)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max(width, 10)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ntsd = load_nts()
    bc = load_bc()
    integrated = build_integrated(ntsd, bc)

    province = ntsd.loc[ntsd["시군구"].eq("합계")].copy()
    local = ntsd.loc[ntsd["시군구"].ne("합계")].copy()

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        integrated.to_excel(writer, sheet_name="BC_국세청_시도통합", index=False)
        province.to_excel(writer, sheet_name="국세청_시도_101업종", index=False)
        local.to_excel(writer, sheet_name="국세청_시군구_101업종", index=False)
        mapping_table().to_excel(writer, sheet_name="BC업종_매핑", index=False)
        quality_table(ntsd, integrated).to_excel(writer, sheet_name="품질_요약", index=False)
        readme_table().to_excel(writer, sheet_name="README", index=False)
        style_workbook(writer)

    print(f"생성 완료: {OUTPUT_PATH}")
    print(f"국세청 원본: {len(ntsd):,}행, BC·국세청 통합: {len(integrated):,}행")


if __name__ == "__main__":
    main()
