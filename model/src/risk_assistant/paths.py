from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
PRESENTATION_ROOT = APP_ROOT.parent
DATA_ROOT = PRESENTATION_ROOT / "data"
BC_PATH = DATA_ROOT / "BC카드_제공데이터" / "ABP_CONTEST_DATA.csv"
EXTERNAL_ROOT = DATA_ROOT / "외부데이터"
OUTPUT_ROOT = DATA_ROOT / "분석결과"
ARTIFACT_ROOT = APP_ROOT / "artifacts"
