#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.inference import RegionRiskPredictor
from risk_assistant.service import RiskAssistant


if __name__ == "__main__":
    profile = RegionRiskPredictor().predict("충청북도", "충주시", "편의점")
    result = RiskAssistant().diagnose(profile)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
