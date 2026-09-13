#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.modeling import train


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False, indent=2))
