"""Project-wide path constants, resolved from this file's location.

Importing these lets every module work no matter which directory
the script is launched from.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
EXPLAIN_DIR = OUTPUTS_DIR / "explainability"
