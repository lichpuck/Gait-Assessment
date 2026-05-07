"""Configuration for the A_Audition Joint3D conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "raw_data"
MODEL_ROOT = ROOT / "body_models"
SMPL_MODEL_PATH = MODEL_ROOT / "smpl" / "SMPL_NEUTRAL.pkl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "A_Audition"
SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "audition_summary.csv"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"

CANONICAL_AXES = {
    "forward": "+X",
    "lateral": "+Y",
    "left": "+Y",
    "up": "+Z",
    "handedness": "right",
}


@dataclass(frozen=True)
class AuditionConfig:
    raw_data_dir: Path = RAW_DATA_DIR
    model_root: Path = MODEL_ROOT
    smpl_model_path: Path = SMPL_MODEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    summary_csv: Path = SUMMARY_CSV
    mplconfigdir: Path = MPLCONFIGDIR

    min_duration_sec: float = 3.0
    smpl_batch_size: int = 256
    diagnostic_dpi: int = 160

    robust_range_low_percentile: float = 5.0
    robust_range_high_percentile: float = 95.0
    min_forward_axis_margin: float = 0.05
    min_vertical_alignment: float = 0.55
    min_lateral_alignment: float = 0.55
    min_forward_segment_displacement_m: float = 0.05
    smoothing_window_frames: int = 9

    support_speed_floor_mps: float = 0.035
    support_speed_ceiling_mps: float = 0.22
    support_height_tolerance_m: float = 0.055

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / subset_name
