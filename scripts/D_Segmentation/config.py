"""Configuration for the independent D_Segmentation module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PRIMARY_LABELS = (
    "stand_to_sit",
    "sit",
    "sit_to_stand",
    "turn",
    "walk",
    "adjust",
)
PRIMARY_LABEL_TO_INDEX = {label: index for index, label in enumerate(PRIMARY_LABELS)}
AUXILIARY_LABELS = ("hesitation",)

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs" / "C_Representation"
OUTPUT_DIR = ROOT / "outputs" / "D_Segmentation"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
SUBSET_SUMMARY_NAME = "segmentation_summary.csv"


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    if fps <= 0:
        return int(minimum)
    return max(int(round(float(seconds) * float(fps))), int(minimum))


@dataclass(frozen=True)
class SegmentationConfig:
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    mplconfigdir: Path = MPLCONFIGDIR
    subset_summary_name: str = SUBSET_SUMMARY_NAME

    posture_sigma_sec: float = 0.20
    speed_sigma_sec: float = 0.12
    heading_sigma_sec: float = 0.18

    pelvis_height_low_percentile: float = 2.0
    pelvis_height_high_percentile: float = 98.0
    pelvis_height_min_span_m: float = 0.16

    walk_speed_turn_backfill_mps: float = 0.05
    hesitation_walk_speed_mps: float = 0.10
    hesitation_turn_speed_deg_s: float = 10.0
    hesitation_min_duration_sec: float = 0.20
    turn_speed_end_deg_s: float = 10.0
    segment_valid_frame_ratio_min: float = 0.80
    segment_quality_score_min: float = 0.60
    segment_hesitation_exclusion_ratio: float = 0.50

    diagnostic_dpi: int = 160
    diagnostic_generate_by_default: bool = False

    def subset_input_dir(self, subset_name: str) -> Path:
        return Path(self.input_dir) / subset_name

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / subset_name
