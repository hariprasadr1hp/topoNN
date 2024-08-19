import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_DATA_DIR = Path("data/")
PROJECT_RAW_DATA_DIR = PROJECT_DATA_DIR / "raw"
PROJECT_PROCESSED_DATA_DIR = PROJECT_DATA_DIR / "processed"


def create_dir_if_not_exists(fpath: Path) -> None:
    if not os.path.exists(fpath):
        os.makedirs(fpath)
