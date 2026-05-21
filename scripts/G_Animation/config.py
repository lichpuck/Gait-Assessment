"""Configuration for CARE-PD G_Animation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_B_DIR = ROOT / "outputs" / "B_Canonicalization"
INPUT_C_DIR = ROOT / "outputs" / "C_Representation"
INPUT_D_DIR = ROOT / "outputs" / "D_Segmentation"
INPUT_F_DIR = ROOT / "outputs" / "F_Description"
OUTPUT_DIR = ROOT / "outputs" / "G_Animation"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
MODULE_VERSION = "1.0.0"

LABEL_COLORS = {
    "stand_to_sit": "#8c564b",
    "sit": "#4c78a8",
    "sit_to_stand": "#f58518",
    "turn": "#e45756",
    "walk": "#54a24b",
    "adjust": "#b279a2",
}


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    if fps <= 0.0:
        return int(minimum)
    return max(int(round(float(seconds) * float(fps))), int(minimum))


@dataclass(frozen=True)
class AnimationConfig:
    input_b_dir: Path = INPUT_B_DIR
    input_c_dir: Path = INPUT_C_DIR
    input_d_dir: Path = INPUT_D_DIR
    input_f_dir: Path = INPUT_F_DIR
    output_dir: Path = OUTPUT_DIR
    mplconfigdir: Path = MPLCONFIGDIR

    module_version: str = MODULE_VERSION
    manifest_suffix: str = ".json"
    video_suffix: str = ".mp4"

    render_dpi: int = 140
    figure_width_in: float = 15.0
    figure_height_in: float = 9.5
    skeleton_line_width: float = 2.2
    root_trail_line_width: float = 1.1
    curve_line_width: float = 1.25

    camera_elev_deg: float = 18.0
    camera_azim_deg: float = -64.0
    axis_length_m: float = 0.30
    skeleton_padding_ratio: float = 0.12

    coordination_window_sec: float = 0.80
    coordination_min_window_frames: int = 7
    output_fps: float | None = None

    def subset_output_dir(self, subset_name: str) -> Path:
        return Path(self.output_dir) / str(subset_name)