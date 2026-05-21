"""Framewise metric derivation for CARE-PD G_Animation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.C_Representation.numeric_utils import finite_difference

from .config import AnimationConfig, seconds_to_frames
from .io_utils import AnimationSequenceInputs


@dataclass(frozen=True)
class MetricSeries:
    key: str
    title: str
    unit: str
    values: np.ndarray


@dataclass(frozen=True)
class AnimationMetrics:
    time_s: np.ndarray
    frame_index: np.ndarray
    valid_frame_mask: np.ndarray
    series: tuple[MetricSeries, ...]

    def by_key(self) -> dict[str, MetricSeries]:
        return {item.key: item for item in self.series}


def _as_vector(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(arrays[key], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{key} must be 1D, got shape {values.shape}")
    return values.astype(np.float32, copy=False)


def _as_xyz(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(arrays[key], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{key} must have shape (T, 3), got {values.shape}")
    return values.astype(np.float32, copy=False)


def _rolling_correlation(left: np.ndarray, right: np.ndarray, window_frames: int) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float32)
    right_values = np.asarray(right, dtype=np.float32)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("rolling correlation expects matching 1D arrays")

    frame_count = int(left_values.shape[0])
    result = np.zeros(frame_count, dtype=np.float32)
    if frame_count == 0:
        return result

    half_window = max(int(window_frames) // 2, 1)
    minimum_samples = max(min(int(window_frames), frame_count) // 2, 3)
    for frame_index in range(frame_count):
        start = max(0, frame_index - half_window)
        end = min(frame_count, frame_index + half_window + 1)
        if end - start < minimum_samples:
            continue
        x_window = left_values[start:end]
        y_window = right_values[start:end]
        x_centered = x_window - np.mean(x_window)
        y_centered = y_window - np.mean(y_window)
        denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        if denominator <= 1e-8:
            continue
        result[frame_index] = float(np.clip(np.dot(x_centered, y_centered) / denominator, -1.0, 1.0))
    return result.astype(np.float32, copy=False)


def build_animation_metrics(
    inputs: AnimationSequenceInputs,
    *,
    config: AnimationConfig | None = None,
) -> AnimationMetrics:
    config = config or AnimationConfig()
    arrays = inputs.c_sequence.arrays

    time_s = _as_vector(arrays, "time_s")
    frame_index = np.asarray(arrays["frame_index"], dtype=np.int32)
    valid_frame_mask = np.asarray(arrays["valid_frame_mask"], dtype=bool)
    fps = float(inputs.fps)
    coordination_window_frames = max(
        seconds_to_frames(
            config.coordination_window_sec,
            fps,
            minimum=config.coordination_min_window_frames,
        ),
        int(config.coordination_min_window_frames),
    )

    trunk_lean = np.abs(_as_vector(arrays, "trunk_lean_angle_deg"))
    pelvis_roll = np.abs(_as_vector(arrays, "pelvis_roll_deg"))
    left_gait_phase = _as_vector(arrays, "left_gait_phase")
    right_gait_phase = _as_vector(arrays, "right_gait_phase")
    left_foot_speed = _as_vector(arrays, "left_foot_speed_mps")
    right_foot_speed = _as_vector(arrays, "right_foot_speed_mps")
    root_speed_xy = _as_vector(arrays, "root_speed_xy_mps")
    root_acceleration = _as_xyz(arrays, "root_acceleration_mps2")

    symmetry = np.abs(left_gait_phase - right_gait_phase).astype(np.float32, copy=False)
    coordination = _rolling_correlation(left_foot_speed, right_foot_speed, coordination_window_frames)
    jerk = finite_difference(root_acceleration, fps)
    control = np.linalg.norm(jerk, axis=1).astype(np.float32, copy=False)

    series = (
        MetricSeries(
            key="stability",
            title="稳定性：躯干倾斜角度",
            unit="deg",
            values=trunk_lean,
        ),
        MetricSeries(
            key="balance",
            title="平衡性：骨盆侧倾角度",
            unit="deg",
            values=pelvis_roll,
        ),
        MetricSeries(
            key="symmetry",
            title="对称性：左右脚相位差",
            unit="phase",
            values=symmetry,
        ),
        MetricSeries(
            key="coordination",
            title="协调性：左右脚速度一致性",
            unit="corr",
            values=coordination,
        ),
        MetricSeries(
            key="mobility",
            title="移动能力：前进速度",
            unit="m/s",
            values=root_speed_xy,
        ),
        MetricSeries(
            key="control",
            title="控制能力：运动平滑度",
            unit="m/s^3",
            values=control,
        ),
    )
    return AnimationMetrics(
        time_s=time_s.astype(np.float32, copy=False),
        frame_index=frame_index.astype(np.int32, copy=False),
        valid_frame_mask=valid_frame_mask.astype(bool, copy=False),
        series=series,
    )