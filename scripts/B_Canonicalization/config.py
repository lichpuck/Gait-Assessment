"""Configuration for the simplified B canonicalization stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "body_models"
SMPL_MODEL_PATH = MODEL_ROOT / "smpl" / "SMPL_NEUTRAL.pkl"
DEFAULT_INPUT_DIR = ROOT / "outputs" / "A_Audition"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "B_Canonicalization"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"

CANONICAL_AXES = {
    "forward": "+X",
    "left": "+Y",
    "up": "+Z",
    "handedness": "right",
}


@dataclass(frozen=True)
class CanonicalizationConfig:
    """Paths and numeric defaults for sequence-level rigid canonicalization."""

    model_root: Path = MODEL_ROOT
    smpl_model_path: Path = SMPL_MODEL_PATH
    input_dir: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    mplconfigdir: Path = MPLCONFIGDIR
    smpl_batch_size: int = 16

    support_window_sec: float = 0.5
    support_window_stride_sec: float = 0.5
    low_point_fraction: float = 0.15
    min_support_cloud_points: int = 6
    floor_percentile: float = 5.0
    floor_max_tilt_deg: float = 60.0
    floor_max_median_abs_residual_m: float = 0.08
    ground_percentile: float = 5.0

    body_axis_frame_count: int = 5
    heading_pca_min_variance: float = 0.90
    heading_min_robust_range_m: float = 0.20
    heading_min_net_path_ratio: float = 0.15

    range_low_percentile: float = 5.0
    range_high_percentile: float = 95.0
    yaw_search_step_deg: float = 0.25

    scale_method: str = "smpl_beta0_long_bone_scale"
    scale_bone_pairs: tuple[tuple[str, str], ...] = (
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )
    scale_min_valid_bones: int = 2
    scale_clip_min: float = 0.75
    scale_clip_max: float = 1.35

    floor_rank_tol: float = 1e-8
    normal_eps: float = 1e-10
    diagnostic_dpi: int = 160
