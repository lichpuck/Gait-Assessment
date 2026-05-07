"""Signal extraction for D_Segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import SegmentationConfig, seconds_to_frames


@dataclass(frozen=True)
class SignalSet:
    fps: float
    time_sec: np.ndarray
    pelvis_speed_mps: np.ndarray
    pelvis_height_norm: np.ndarray
    foot_speed_mean_mps: np.ndarray
    heading_smooth_deg: np.ndarray
    turn_angle_from_start_deg: np.ndarray
    turn_speed_deg_s: np.ndarray
    distance_from_start_m: np.ndarray
    pelvis_height_m: np.ndarray


def smooth_array(values: np.ndarray, sigma_frames: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if sigma_frames <= 0 or array.shape[0] <= 1:
        return array
    return gaussian_filter1d(array, sigma=sigma_frames, axis=0, mode="nearest")


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] < 2:
        return np.zeros_like(array, dtype=np.float32)
    return (np.gradient(array, axis=0) * float(fps)).astype(np.float32)


def wrap_to_180_deg(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return ((array + 180.0) % 360.0 - 180.0).astype(np.float32)


def smooth_heading_deg(heading_deg: np.ndarray, fps: float, config: SegmentationConfig) -> np.ndarray:
    heading = np.asarray(heading_deg, dtype=np.float32)
    if heading.size == 0:
        return heading
    sigma = seconds_to_frames(config.heading_sigma_sec, fps, minimum=0)
    unwrapped_rad = np.unwrap(np.deg2rad(heading.astype(np.float64)))
    smoothed_rad = smooth_array(unwrapped_rad.astype(np.float32), sigma)
    return wrap_to_180_deg(np.rad2deg(smoothed_rad).astype(np.float32))


def smooth_unwrapped_heading_deg(heading_unwrapped_deg: np.ndarray, fps: float, config: SegmentationConfig) -> np.ndarray:
    heading = np.asarray(heading_unwrapped_deg, dtype=np.float32)
    if heading.size == 0:
        return heading
    sigma = seconds_to_frames(config.heading_sigma_sec, fps, minimum=0)
    return smooth_array(heading, sigma).astype(np.float32)


def compute_signals(arrays: dict[str, np.ndarray], fps: float, config: SegmentationConfig) -> SignalSet:
    root_pos_raw = np.asarray(arrays["root_pos_m"], dtype=np.float32)
    root_speed_xy_raw = np.asarray(arrays["root_speed_xy_mps"], dtype=np.float32)
    left_foot_speed_raw = np.asarray(arrays["left_foot_speed_mps"], dtype=np.float32)
    right_foot_speed_raw = np.asarray(arrays["right_foot_speed_mps"], dtype=np.float32)
    heading_unwrapped_raw = np.asarray(arrays["heading_unwrapped_deg"], dtype=np.float32)
    yaw_rate_raw = np.asarray(arrays["yaw_rate_deg_s"], dtype=np.float32)
    pelvis_height_raw = np.asarray(arrays["pelvis_height_m"], dtype=np.float32)
    frame_count = int(root_pos_raw.shape[0])

    speed_sigma = seconds_to_frames(config.speed_sigma_sec, fps, minimum=0)
    heading_sigma = seconds_to_frames(config.heading_sigma_sec, fps, minimum=0)

    root_pos = np.asarray(root_pos_raw, dtype=np.float32)
    pelvis_speed_mps = smooth_array(root_speed_xy_raw, speed_sigma)

    left_foot_speed = smooth_array(left_foot_speed_raw, speed_sigma)
    right_foot_speed = smooth_array(right_foot_speed_raw, speed_sigma)
    foot_speed_mean_mps = (0.5 * (left_foot_speed + right_foot_speed)).astype(np.float32)

    low = float(np.nanpercentile(pelvis_height_raw, config.pelvis_height_low_percentile)) if frame_count else 0.0
    high = float(np.nanpercentile(pelvis_height_raw, config.pelvis_height_high_percentile)) if frame_count else 0.0
    height_span = max(high - low, 1e-4)
    if height_span < config.pelvis_height_min_span_m:
        pelvis_height_norm = np.full(frame_count, 0.5, dtype=np.float32)
    else:
        pelvis_height_norm = np.clip((pelvis_height_raw - low) / height_span, 0.0, 1.0).astype(np.float32)

    heading_unwrapped_smooth_deg = smooth_unwrapped_heading_deg(heading_unwrapped_raw, fps, config)
    if frame_count:
        heading_smooth_deg = wrap_to_180_deg(heading_unwrapped_smooth_deg)
        turn_angle_from_start_deg = np.abs(
            wrap_to_180_deg(heading_unwrapped_smooth_deg - float(heading_unwrapped_smooth_deg[0]))
        ).astype(np.float32)
        turn_speed_deg_s = np.abs(smooth_array(yaw_rate_raw, heading_sigma)).astype(np.float32)
    else:
        heading_smooth_deg = np.zeros((0,), dtype=np.float32)
        turn_angle_from_start_deg = np.zeros((0,), dtype=np.float32)
        turn_speed_deg_s = np.zeros((0,), dtype=np.float32)

    xy_delta = root_pos[:, :2] - root_pos[:1, :2]
    distance_from_start_m = np.linalg.norm(xy_delta, axis=1).astype(np.float32)
    time_sec = (np.arange(frame_count, dtype=np.float32) / float(fps)).astype(np.float32)

    return SignalSet(
        fps=float(fps),
        time_sec=time_sec,
        pelvis_speed_mps=pelvis_speed_mps.astype(np.float32),
        pelvis_height_norm=pelvis_height_norm.astype(np.float32),
        foot_speed_mean_mps=foot_speed_mean_mps.astype(np.float32),
        heading_smooth_deg=heading_smooth_deg.astype(np.float32),
        turn_angle_from_start_deg=turn_angle_from_start_deg.astype(np.float32),
        turn_speed_deg_s=turn_speed_deg_s.astype(np.float32),
        distance_from_start_m=distance_from_start_m.astype(np.float32),
        pelvis_height_m=pelvis_height_raw.astype(np.float32),
    )
