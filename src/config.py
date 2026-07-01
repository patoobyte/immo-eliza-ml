# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = ROOT / "data" / "raw" / "listings_raw_20260629.csv"
DATA_AUDITED = ROOT / "data" / "raw" / "listings_raw_20260629_audited.csv"
DATA_CLN = ROOT / "data" / "clean" / "listings_clean.csv"
RANDOM_SEED = 13
