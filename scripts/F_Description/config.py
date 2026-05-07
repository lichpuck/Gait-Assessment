"""Configuration for the F_Description module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_E_DIR = ROOT / "outputs" / "E_Extraction"
OUTPUT_DIR = ROOT / "outputs" / "F_Description"


@dataclass(frozen=True)
class DescriptionConfig:
    input_e_dir: Path = INPUT_E_DIR
    output_dir: Path = OUTPUT_DIR
    language: str = "zh-CN"

    walk_features_csv_name: str = "walk_features.csv"
    turn_features_csv_name: str = "turn_features.csv"

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / str(subset_name)
