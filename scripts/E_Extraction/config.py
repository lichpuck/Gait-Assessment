"""Configuration for the E_Extraction module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_C_DIR = ROOT / "outputs" / "C_Representation"
INPUT_D_DIR = ROOT / "outputs" / "D_Segmentation"
OUTPUT_DIR = ROOT / "outputs" / "E_Extraction"

WALK_FEATURES_CSV_NAME = "walk_features.csv"
TURN_FEATURES_CSV_NAME = "turn_features.csv"


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    if fps <= 0.0:
        return int(minimum)
    return max(int(round(float(seconds) * float(fps))), int(minimum))


@dataclass(frozen=True)
class ExtractionConfig:
    input_c_dir: Path = INPUT_C_DIR
    input_d_dir: Path = INPUT_D_DIR
    output_dir: Path = OUTPUT_DIR

    walk_features_csv_name: str = WALK_FEATURES_CSV_NAME
    turn_features_csv_name: str = TURN_FEATURES_CSV_NAME

    robust_range_low_percentile: float = 5.0
    robust_range_high_percentile: float = 95.0
    asymmetry_epsilon: float = 1e-6

    pre_turn_hesitation_speed_mps: float = 0.10
    pre_turn_hesitation_max_lookback_sec: float = 1.00

    reorientation_onset_fraction: float = 0.10
    reorientation_min_total_angle_deg: float = 15.0
    turn_radius_min_angle_deg: float = 5.0

    allow_b_pose_fallback: bool = True

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / str(subset_name)