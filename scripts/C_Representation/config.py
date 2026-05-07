"""Configuration for CARE-PD C_Representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "outputs" / "B_Canonicalization"
OUTPUT_DIR = ROOT / "outputs" / "C_Representation"
MODULE_VERSION = "2.0.0"

CANONICAL_AXES = {
    "forward": "+X",
    "left": "+Y",
    "up": "+Z",
    "handedness": "right",
}

REQUIRED_B_NPZ_KEYS = ("pose_raw", "trans_raw", "joints_can", "trans_can")
OUTPUT_ARTIFACTS = (".npz", ".json")
STRICT_NPZ_FIELDS = (
    "fps",
    "time_s",
    "frame_index",
    "valid_frame_mask",
    "joints_can",
    "trans_can",
    "root_pos_m",
    "root_velocity_mps",
    "root_speed_xy_mps",
    "root_acceleration_mps2",
    "pelvis_height_m",
    "pelvis_vertical_velocity_mps",
    "heading_deg",
    "heading_unwrapped_deg",
    "yaw_rate_deg_s",
    "yaw_acceleration_deg_s2",
    "left_foot_pos_m",
    "right_foot_pos_m",
    "left_foot_velocity_mps",
    "right_foot_velocity_mps",
    "left_foot_speed_mps",
    "right_foot_speed_mps",
    "left_foot_height_m",
    "right_foot_height_m",
    "left_foot_contact_prob",
    "right_foot_contact_prob",
    "left_foot_contact",
    "right_foot_contact",
    "contact_confidence",
    "left_heel_strike",
    "right_heel_strike",
    "left_toe_off",
    "right_toe_off",
    "left_gait_phase",
    "right_gait_phase",
    "gait_phase_global",
    "trunk_forward_flexion_deg",
    "trunk_lateral_lean_deg",
    "trunk_lean_angle_deg",
    "pelvis_pitch_deg",
    "pelvis_roll_deg",
    "pelvis_yaw_deg",
    "trunk_pitch_deg",
    "trunk_roll_deg",
    "trunk_yaw_deg",
    "joint_nan_mask",
    "velocity_outlier_mask",
    "representation_quality_score",
)


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    if fps <= 0:
        return int(minimum)
    return max(int(round(float(seconds) * float(fps))), int(minimum))


@dataclass(frozen=True)
class RepresentationConfig:
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR

    heading_min_horizontal_norm_m: float = 1e-5
    heading_velocity_fallback_min_mps: float = 0.03

    contact_floor_percentile: float = 5.0
    contact_height_threshold_m: float = 0.08
    contact_speed_threshold_mps: float = 0.30
    contact_height_weight: float = 0.60
    contact_speed_weight: float = 0.40
    contact_binary_threshold: float = 0.50
    min_contact_duration_sec: float = 0.08
    min_swing_duration_sec: float = 0.08

    max_root_speed_mps: float = 4.0
    max_foot_speed_mps: float = 8.0
    contact_degenerate_std_min: float = 0.02

    def subset_input_dir(self, subset_name: str) -> Path:
        return Path(self.input_dir) / subset_name

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / subset_name

    def to_parameters(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_dir"] = str(self.input_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload
