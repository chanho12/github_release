"""BC region-industry risk diagnosis and conversational assistance prototype."""

from pathlib import Path

from dotenv import load_dotenv


# `model/.env` is loaded automatically; exported variables keep priority.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

__version__ = "0.1.0"
