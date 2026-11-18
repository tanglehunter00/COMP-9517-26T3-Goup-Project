"""Portable locations for reproducing the frozen 5090 D report."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN_DIR=Path(os.environ.get('COMP9517_RUN_DIR',ROOT/'runs/5090d_batch8_seed42_20261117')).resolve()
DATA_ROOT=Path(os.environ.get('COMP9517_DATA_ROOT',ROOT/'Yolo Improeved/work/beach-bird/Beach Bird Detection Dataset v1.0')).resolve()
OUTPUT_DIR=Path(os.environ.get('COMP9517_REPORT_OUTPUT',ROOT/'outputs')).resolve()
