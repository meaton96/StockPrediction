from pathlib import Path
from datetime import date
# config.py lives at project_root/src/models/config.py
# parents[2] -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

TRAIN_CUTOFF = '2019-01-01'
VALIDATE_CUTOFF = '2022-01-01'

