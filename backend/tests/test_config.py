"""Tests for settings: data paths must resolve against the project root,
not the current working directory (the API runs from backend/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR, Settings


def test_relative_data_paths_resolve_against_project_root():
    s = Settings(data_raw_dir="./data/raw", data_processed_dir="./data/processed",
                 data_index_dir="./data/index")
    root = BASE_DIR.parent
    assert Path(s.data_index_dir).is_absolute()
    assert Path(s.data_index_dir) == root / "data" / "index"
    assert Path(s.data_raw_dir) == root / "data" / "raw"


def test_absolute_paths_are_kept():
    s = Settings(data_index_dir="D:/somewhere/index")
    assert Path(s.data_index_dir) == Path("D:/somewhere/index")
