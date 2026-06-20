"""Gold-set loaders for the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from clew.config import get_settings


def _gold_path(name: str) -> Path:
    return get_settings().gold_dir / name


def load_er_pairs() -> dict:
    return json.loads(_gold_path("er_pairs.json").read_text())


def load_extraction_gold() -> dict:
    return json.loads(_gold_path("extraction_13d.json").read_text())
