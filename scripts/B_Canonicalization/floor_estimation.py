"""Support-point cloud construction and floor-plane fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CanonicalizationConfig
from .joints_schema import JOINT_INDEX, validate_joints


SUPPORT_JOINT_NAMES = ("left_foot", "left_ankle", "right_foot", "right_ankle")


class FloorEstimationError(ValueError):
    """Raised when a sequence cannot support a reliable global floor fit."""


@dataclass(frozen=True)
class FloorEstimationResult:
    floor_normal_before: np.ndarray
    floor_normal_after: np.ndarray
    R_floor: np.ndarray
    plane_coefficients: np.ndarray
    residuals: np.ndarray
    support_points: np.ndarray
    support_joint_indices: np.ndarray
    robust_support_cloud: np.ndarray
    metrics: dict[str, float | int | bool | dict[str, int]]
    warnings: tuple[str, ...]


def normalize_vector(vector: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < eps:
        raise FloorEstimationError("Cannot normalize a degenerate floor normal.")
    return array / norm


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_between_vectors(source: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    source_vec = normalize_vector(source, eps=eps)
    target_vec = normalize_vector(target, eps=eps)
    cross = np.cross(source_vec, target_vec)
    dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
    if dot > 1.0 - eps:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + eps:
        axis = np.cross(source_vec, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < eps:
            axis = np.cross(source_vec, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = normalize_vector(axis, eps=eps)
        k = _skew(axis)
        return np.eye(3, dtype=np.float64) + 2.0 * (k @ k)
    k = _skew(cross)
    return np.eye(3, dtype=np.float64) + k + k @ k * (1.0 / (1.0 + dot))


def compute_support_points(joints_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    joints = validate_joints(joints_3d, name="joints_3d")
    candidates = np.stack([joints[:, JOINT_INDEX[name]] for name in SUPPORT_JOINT_NAMES], axis=1)
    selected = np.argmin(candidates[:, :, 2], axis=1)
    support_points = candidates[np.arange(candidates.shape[0]), selected]
    counts = {
        f"{name}_selected_count": int(np.count_nonzero(selected == index))
        for index, name in enumerate(SUPPORT_JOINT_NAMES)
    }
    return support_points.astype(np.float32, copy=False), selected.astype(np.int16, copy=False), counts


def build_robust_support_cloud(
    support_points: np.ndarray,
    fps: float,
    config: CanonicalizationConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    support = np.asarray(support_points, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3:
        raise FloorEstimationError(f"support_points must have shape (T, 3), got {support.shape}")
    finite = np.all(np.isfinite(support), axis=1)
    if not np.any(finite):
        raise FloorEstimationError("No finite support points are available.")

    window_frames = max(1, int(round(config.support_window_sec * fps)))
    stride_frames = max(1, int(round(config.support_window_stride_sec * fps)))
    selected_chunks: list[np.ndarray] = []
    nonempty_windows = 0
    for start in range(0, support.shape[0], stride_frames):
        stop = min(start + window_frames, support.shape[0])
        window = support[start:stop]
        window = window[np.all(np.isfinite(window), axis=1)]
        if window.size == 0:
            continue
        nonempty_windows += 1
        keep_count = max(1, int(np.ceil(window.shape[0] * config.low_point_fraction)))
        order = np.argsort(window[:, 2], kind="mergesort")
        selected_chunks.append(window[order[:keep_count]])

    if not selected_chunks:
        raise FloorEstimationError("No robust support points remained after windowing.")
    cloud = np.concatenate(selected_chunks, axis=0)
    if cloud.shape[0] < config.min_support_cloud_points:
        raise FloorEstimationError(
            f"Only {cloud.shape[0]} robust support points remained; "
            f"minimum is {config.min_support_cloud_points}."
        )
    metrics = {
        "support_point_count": int(support.shape[0]),
        "finite_support_point_count": int(np.count_nonzero(finite)),
        "window_frames": int(window_frames),
        "stride_frames": int(stride_frames),
        "window_count": int(nonempty_windows),
        "low_point_fraction": float(config.low_point_fraction),
        "robust_support_cloud_count": int(cloud.shape[0]),
    }
    return cloud.astype(np.float32, copy=False), metrics


def fit_floor_plane_z(support_cloud: np.ndarray, config: CanonicalizationConfig) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.asarray(support_cloud, dtype=np.float64)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if cloud.shape[0] < config.min_support_cloud_points:
        raise FloorEstimationError("Insufficient robust support points for floor fitting.")
    design = np.column_stack([cloud[:, 0], cloud[:, 1], np.ones(cloud.shape[0], dtype=np.float64)])
    rank = int(np.linalg.matrix_rank(design, tol=config.floor_rank_tol))
    if rank < 3:
        raise FloorEstimationError("Robust support cloud is degenerate in XY; cannot fit Z=aX+bY+c.")
    coeffs, _residual_sum, _rank, _s = np.linalg.lstsq(design, cloud[:, 2], rcond=None)
    if not np.all(np.isfinite(coeffs)):
        raise FloorEstimationError("Floor plane coefficients are non-finite.")
    residuals = cloud[:, 2] - (design @ coeffs)
    if not np.all(np.isfinite(residuals)):
        raise FloorEstimationError("Floor plane residuals are non-finite.")
    return coeffs.astype(np.float64), residuals.astype(np.float64)


def estimate_floor(joints_3d: np.ndarray, fps: float, config: CanonicalizationConfig) -> FloorEstimationResult:
    support_points, selected_indices, selected_counts = compute_support_points(joints_3d)
    robust_cloud, cloud_metrics = build_robust_support_cloud(support_points, fps, config)
    coeffs, residuals = fit_floor_plane_z(robust_cloud, config)

    a, b, _c = [float(value) for value in coeffs]
    floor_normal_before = normalize_vector(np.array([-a, -b, 1.0], dtype=np.float64), eps=config.normal_eps)
    if floor_normal_before[2] < 0:
        floor_normal_before = -floor_normal_before
    R_floor = rotation_between_vectors(floor_normal_before, np.array([0.0, 0.0, 1.0]), eps=config.normal_eps)
    if not np.allclose(R_floor @ R_floor.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R_floor), 1.0, atol=1e-6):
        raise FloorEstimationError("Floor leveling rotation is not a valid right-handed rotation.")
    floor_normal_after = R_floor @ floor_normal_before
    floor_normal_after = floor_normal_after / max(float(np.linalg.norm(floor_normal_after)), config.normal_eps)

    abs_res = np.abs(residuals)
    metrics: dict[str, float | int | bool | dict[str, int]] = {
        **cloud_metrics,
        "support_joint_selected_counts": selected_counts,
        "plane_rank_ok": True,
        "residual_mean_abs_m": float(np.mean(abs_res)),
        "residual_median_abs_m": float(np.median(abs_res)),
        "residual_max_abs_m": float(np.max(abs_res)),
        "tilt_before_deg": float(np.degrees(np.arccos(np.clip(floor_normal_before[2], -1.0, 1.0)))),
        "tilt_after_deg": float(np.degrees(np.arccos(np.clip(floor_normal_after[2], -1.0, 1.0)))),
    }
    return FloorEstimationResult(
        floor_normal_before=floor_normal_before.astype(np.float32),
        floor_normal_after=floor_normal_after.astype(np.float32),
        R_floor=R_floor.astype(np.float32),
        plane_coefficients=coeffs.astype(np.float32),
        residuals=residuals.astype(np.float32),
        support_points=support_points.astype(np.float32, copy=False),
        support_joint_indices=selected_indices,
        robust_support_cloud=robust_cloud.astype(np.float32, copy=False),
        metrics=metrics,
        warnings=(),
    )

