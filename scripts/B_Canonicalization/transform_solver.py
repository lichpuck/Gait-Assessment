"""Rigid transform composition for B canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import CanonicalizationConfig
from .floor_estimation import FloorEstimationError, estimate_floor
from .joints_schema import ROBUST_SCALE_BONE_PAIRS, JOINT_INDEX, bone_lengths, bone_pair_labels, validate_joints
from .smpl_forward import smpl_to_joints


SUPPORT_JOINT_NAMES = ("left_foot", "left_ankle", "right_foot", "right_ankle")


@dataclass(frozen=True)
class BodyFrameAlignmentResult:
    R_body: np.ndarray
    forward_axis: np.ndarray
    left_axis: np.ndarray
    up_axis: np.ndarray
    frames_used: int
    aggregation: str
    determinant: float
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class YawAlignmentResult:
    R_yaw: np.ndarray
    enabled: bool
    skipped_reason: str | None
    yaw_angle_rad: float
    yaw_angle_deg: float
    forward_axis_body: np.ndarray
    pca_variance_ratio: float
    robust_range_m: float
    projection_p5_m: float
    projection_p95_m: float
    net_displacement_m: float
    path_length_m: float
    net_path_ratio: float
    metrics: dict[str, float | int | str | bool | np.ndarray]


@dataclass(frozen=True)
class GroundAlignmentResult:
    ground_z: float
    percentile: float
    sample_count: int
    support_joint_names: tuple[str, ...]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class ScaleAlignmentResult:
    enabled: bool
    method: str
    skipped_reason: str | None
    s_global: float
    s_global_raw: float
    clip_applied: bool
    clip_range: tuple[float, float]
    aggregation: str
    bones_used: tuple[str, ...]
    bone_count_valid: int
    quality_flag: str
    target_bone_lengths_m: dict[str, float]
    observed_bone_lengths_m: dict[str, float | None]
    per_bone_ratios: dict[str, float | None]
    metrics: dict[str, float | int | str | bool | tuple[str, ...] | dict[str, float | None] | None]


@dataclass(frozen=True)
class FloorAlignmentResult:
    R_floor: np.ndarray
    enabled: bool
    skipped_reason: str | None
    floor_normal_before: np.ndarray
    floor_normal_after: np.ndarray
    plane_coefficients: np.ndarray
    tilt_before_deg: float
    tilt_after_deg: float
    residual_median_abs_m: float
    robust_support_cloud_count: int
    support_joint_names: tuple[str, ...]
    metrics: dict[str, float | int | str | bool | dict[str, int] | None]


@dataclass(frozen=True)
class TransformResult:
    R_global: np.ndarray
    t_global: np.ndarray
    R_body: np.ndarray
    R_floor: np.ndarray
    R_yaw: np.ndarray
    s_global: float
    body: BodyFrameAlignmentResult
    scale: ScaleAlignmentResult
    floor: FloorAlignmentResult
    yaw: YawAlignmentResult
    ground: GroundAlignmentResult
    metrics: dict[str, float | bool]


def _validate_rotation(name: str, rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} is not orthonormal")
    det = float(np.linalg.det(matrix))
    if not np.isclose(det, 1.0, atol=1e-5):
        raise ValueError(f"{name} must be right-handed with determinant 1, got {det:.8f}")
    return matrix


def _normalize_vector(name: str, vector: np.ndarray, eps: float) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"Cannot normalize degenerate {name} vector.")
    return array / norm


def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray | None = None,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    transformed = np.einsum("ij,...j->...i", np.asarray(R, dtype=np.float64), np.asarray(points, dtype=np.float64))
    transformed = float(scale) * transformed
    if t is not None:
        transformed = transformed + np.asarray(t, dtype=np.float64).reshape(3)
    return transformed.astype(np.float32, copy=False)


@lru_cache(maxsize=8)
def _target_scale_bone_lengths(
    model_root: str,
    smpl_model_path: str,
    bone_pairs: tuple[tuple[str, str], ...],
    batch_size: int,
) -> tuple[dict[str, float], str, tuple[str, ...]]:
    pose = np.zeros((1, 72), dtype=np.float32)
    trans = np.zeros((1, 3), dtype=np.float32)
    beta = np.zeros(10, dtype=np.float32)
    result = smpl_to_joints(
        pose,
        trans,
        beta,
        model_root=model_root,
        smpl_model_path=smpl_model_path,
        batch_size=max(int(batch_size), 1),
    )
    labels = bone_pair_labels(bone_pairs)
    lengths = np.asarray(bone_lengths(result.joints, bone_pairs), dtype=np.float64)
    target = {label: float(lengths[0, index]) for index, label in enumerate(labels)}
    return target, str(result.backend_used), tuple(result.backend_notes)


def solve_scale_alignment(joints_body: np.ndarray, config: CanonicalizationConfig) -> ScaleAlignmentResult:
    joints = validate_joints(joints_body, name="joints_body")
    bone_pairs = tuple(config.scale_bone_pairs) or ROBUST_SCALE_BONE_PAIRS
    bone_labels = bone_pair_labels(bone_pairs)
    clip_range = (float(config.scale_clip_min), float(config.scale_clip_max))
    target_lengths, backend_used, backend_notes = _target_scale_bone_lengths(
        str(config.model_root),
        str(config.smpl_model_path),
        bone_pairs,
        int(config.smpl_batch_size),
    )

    observed_samples = np.asarray(bone_lengths(joints, bone_pairs), dtype=np.float64)
    observed_lengths: dict[str, float | None] = {}
    per_bone_ratios: dict[str, float | None] = {}
    valid_labels: list[str] = []
    valid_ratios: list[float] = []
    for index, label in enumerate(bone_labels):
        samples = observed_samples[:, index]
        finite = np.isfinite(samples) & (samples > config.normal_eps)
        if not np.any(finite):
            observed_lengths[label] = None
            per_bone_ratios[label] = None
            continue

        observed_value = float(np.median(samples[finite]))
        ratio = float(target_lengths[label] / observed_value) if observed_value > config.normal_eps else float("nan")
        observed_lengths[label] = observed_value
        per_bone_ratios[label] = ratio if np.isfinite(ratio) and ratio > config.normal_eps else None
        if np.isfinite(ratio) and ratio > config.normal_eps:
            valid_labels.append(label)
            valid_ratios.append(ratio)

    valid_count = len(valid_ratios)
    min_valid_bones = min(max(int(config.scale_min_valid_bones), 1), len(bone_labels))
    metrics = {
        "target_backend_used": backend_used,
        "target_backend_notes": backend_notes,
        "bone_count_total": int(len(bone_labels)),
        "bone_count_valid": int(valid_count),
        "minimum_valid_bones": int(min_valid_bones),
        "target_bone_lengths_m": target_lengths,
        "observed_bone_lengths_m": observed_lengths,
        "per_bone_ratios": per_bone_ratios,
    }
    if valid_count < min_valid_bones:
        return ScaleAlignmentResult(
            enabled=False,
            method=str(config.scale_method),
            skipped_reason="insufficient_valid_bones",
            s_global=1.0,
            s_global_raw=float("nan"),
            clip_applied=False,
            clip_range=clip_range,
            aggregation="median",
            bones_used=tuple(valid_labels),
            bone_count_valid=int(valid_count),
            quality_flag="insufficient_valid_bones",
            target_bone_lengths_m=target_lengths,
            observed_bone_lengths_m=observed_lengths,
            per_bone_ratios=per_bone_ratios,
            metrics=metrics,
        )

    s_global_raw = float(np.median(np.asarray(valid_ratios, dtype=np.float64)))
    if not np.isfinite(s_global_raw) or s_global_raw <= config.normal_eps:
        return ScaleAlignmentResult(
            enabled=False,
            method=str(config.scale_method),
            skipped_reason="invalid_scale_ratio",
            s_global=1.0,
            s_global_raw=s_global_raw,
            clip_applied=False,
            clip_range=clip_range,
            aggregation="median",
            bones_used=tuple(valid_labels),
            bone_count_valid=int(valid_count),
            quality_flag="invalid_scale_ratio",
            target_bone_lengths_m=target_lengths,
            observed_bone_lengths_m=observed_lengths,
            per_bone_ratios=per_bone_ratios,
            metrics=metrics,
        )

    s_global = float(np.clip(s_global_raw, clip_range[0], clip_range[1]))
    clip_applied = not np.isclose(s_global, s_global_raw)
    quality_flag = "clipped" if clip_applied else "ok"
    return ScaleAlignmentResult(
        enabled=True,
        method=str(config.scale_method),
        skipped_reason=None,
        s_global=s_global,
        s_global_raw=s_global_raw,
        clip_applied=bool(clip_applied),
        clip_range=clip_range,
        aggregation="median",
        bones_used=tuple(valid_labels),
        bone_count_valid=int(valid_count),
        quality_flag=quality_flag,
        target_bone_lengths_m=target_lengths,
        observed_bone_lengths_m=observed_lengths,
        per_bone_ratios=per_bone_ratios,
        metrics=metrics,
    )


def estimate_body_frame(joints_3d: np.ndarray, config: CanonicalizationConfig) -> BodyFrameAlignmentResult:
    joints = validate_joints(joints_3d, name="joints_3d")
    frames_used = min(max(int(config.body_axis_frame_count), 1), joints.shape[0])
    window = joints[:frames_used]

    left_hip = window[:, JOINT_INDEX["left_hip"]]
    right_hip = window[:, JOINT_INDEX["right_hip"]]
    left_shoulder = window[:, JOINT_INDEX["left_shoulder"]]
    right_shoulder = window[:, JOINT_INDEX["right_shoulder"]]

    mid_hip = (left_hip + right_hip) * 0.5
    mid_shoulder = (left_shoulder + right_shoulder) * 0.5
    up_raw = np.median(mid_shoulder - mid_hip, axis=0)
    up = _normalize_vector("torso-up", up_raw, config.normal_eps)

    left_candidates = np.concatenate(
        [left_hip - right_hip, left_shoulder - right_shoulder],
        axis=0,
    )
    left_raw = np.median(left_candidates, axis=0)
    left_orthogonal = left_raw - float(np.dot(left_raw, up)) * up
    left = _normalize_vector("body-left", left_orthogonal, config.normal_eps)
    forward = _normalize_vector("body-forward", np.cross(left, up), config.normal_eps)

    R_body = np.stack([forward, left, up], axis=0)
    R_body = _validate_rotation("R_body", R_body)
    det = float(np.linalg.det(R_body))
    metrics = {
        "up_raw_norm": float(np.linalg.norm(up_raw)),
        "left_raw_norm": float(np.linalg.norm(left_raw)),
        "left_orthogonal_norm": float(np.linalg.norm(left_orthogonal)),
        "forward_norm": float(np.linalg.norm(forward)),
    }
    return BodyFrameAlignmentResult(
        R_body=R_body.astype(np.float32),
        forward_axis=forward.astype(np.float32),
        left_axis=left.astype(np.float32),
        up_axis=up.astype(np.float32),
        frames_used=int(frames_used),
        aggregation="componentwise_median",
        determinant=det,
        metrics=metrics,
    )


def _yaw_rotation_to_plus_x(forward_axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(forward_axis, dtype=np.float64).reshape(2)
    angle = float(np.arctan2(axis[1], axis[0]))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def solve_yaw_alignment(
    trans_body: np.ndarray,
    config: CanonicalizationConfig,
    fallback_forward_xy: np.ndarray | None = None,
) -> YawAlignmentResult:
    trans = np.asarray(trans_body, dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans_body must have shape (T, 3), got {trans.shape}")
    finite = np.all(np.isfinite(trans[:, :2]), axis=1)
    if not np.any(finite):
        raise ValueError("No finite root XY samples are available for yaw alignment.")

    xy = trans[finite, :2]
    centered = xy - np.median(xy, axis=0)
    if xy.shape[0] < 2:
        covariance = np.zeros((2, 2), dtype=np.float64)
    else:
        covariance = np.cov(centered.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Root XY covariance for yaw alignment is non-finite.")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal = eigenvectors[:, order[0]] if order.size else np.array([1.0, 0.0], dtype=np.float64)
    principal_norm = float(np.linalg.norm(principal))
    if principal_norm < config.normal_eps:
        principal = np.array([1.0, 0.0], dtype=np.float64)
    else:
        principal = principal / principal_norm

    projection = xy @ principal
    p5, p95 = np.percentile(projection, [config.range_low_percentile, config.range_high_percentile])
    robust_range = float(p95 - p5)
    total_variance = float(np.sum(eigenvalues))
    variance_ratio = float(eigenvalues[0] / total_variance) if total_variance > config.normal_eps else 0.0

    diffs = np.diff(xy, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1))) if diffs.size else 0.0
    displacement = xy[-1] - xy[0]
    net_displacement = float(np.linalg.norm(displacement))
    net_path_ratio = net_displacement / max(path_length, config.normal_eps)

    skipped_reason: str | None = None
    if variance_ratio < config.heading_pca_min_variance:
        skipped_reason = "pca_variance_below_threshold"
    elif robust_range < config.heading_min_robust_range_m:
        skipped_reason = "robust_range_below_threshold"

    use_net_sign = net_path_ratio >= config.heading_min_net_path_ratio and net_displacement > config.normal_eps
    fallback_axis = np.array([1.0, 0.0], dtype=np.float64)
    if fallback_forward_xy is not None:
        fallback_candidate = np.asarray(fallback_forward_xy, dtype=np.float64).reshape(2)
        fallback_norm = float(np.linalg.norm(fallback_candidate))
        if np.isfinite(fallback_norm) and fallback_norm >= config.normal_eps:
            fallback_axis = fallback_candidate / fallback_norm

    if use_net_sign:
        if float(np.dot(principal, displacement)) < 0.0:
            principal = -principal
            p5, p95 = -p95, -p5
    elif float(np.dot(principal, fallback_axis)) < 0.0:
        principal = -principal
        p5, p95 = -p95, -p5

    enabled = skipped_reason is None
    R_yaw = _yaw_rotation_to_plus_x(principal) if enabled else np.eye(3, dtype=np.float64)
    yaw_angle = float(np.arctan2(principal[1], principal[0])) if enabled else 0.0
    metrics = {
        "method": "pelvis_xy_pca",
        "enabled": bool(enabled),
        "variance_threshold": float(config.heading_pca_min_variance),
        "robust_range_threshold_m": float(config.heading_min_robust_range_m),
        "net_path_ratio_threshold": float(config.heading_min_net_path_ratio),
        "used_net_displacement_sign": bool(use_net_sign),
        "fallback_forward_xy": fallback_axis.astype(np.float32),
        "range_metric": f"P{config.range_high_percentile:g}-P{config.range_low_percentile:g}",
    }
    return YawAlignmentResult(
        R_yaw=R_yaw.astype(np.float32),
        enabled=bool(enabled),
        skipped_reason=skipped_reason,
        yaw_angle_rad=yaw_angle,
        yaw_angle_deg=float(np.degrees(yaw_angle)),
        forward_axis_body=np.array([principal[0], principal[1], 0.0], dtype=np.float32),
        pca_variance_ratio=variance_ratio,
        robust_range_m=robust_range,
        projection_p5_m=float(p5),
        projection_p95_m=float(p95),
        net_displacement_m=net_displacement,
        path_length_m=path_length,
        net_path_ratio=float(net_path_ratio),
        metrics=metrics,
    )


def ground_joint_points(joints_3d: np.ndarray) -> np.ndarray:
    joints = validate_joints(joints_3d, name="joints_3d")
    return np.stack([joints[:, JOINT_INDEX[name]] for name in SUPPORT_JOINT_NAMES], axis=1).astype(np.float32)


def per_frame_lowest_ground_points(ground_points: np.ndarray) -> np.ndarray:
    points = np.asarray(ground_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (len(SUPPORT_JOINT_NAMES), 3):
        raise ValueError(
            f"ground_points must have shape (T, {len(SUPPORT_JOINT_NAMES)}, 3), got {points.shape}"
        )
    selected = np.argmin(points[:, :, 2], axis=1)
    return points[np.arange(points.shape[0]), selected].astype(np.float32)


def solve_floor_alignment(
    joints_body: np.ndarray,
    fps: float,
    config: CanonicalizationConfig,
) -> FloorAlignmentResult:
    thresholds = {
        "floor_max_tilt_deg": float(config.floor_max_tilt_deg),
        "floor_max_median_abs_residual_m": float(config.floor_max_median_abs_residual_m),
        "min_support_cloud_points": int(config.min_support_cloud_points),
    }
    try:
        floor = estimate_floor(joints_body, fps, config)
    except FloorEstimationError as error:
        metrics: dict[str, float | int | str | bool | dict[str, int] | None] = {
            "method": "low_foot_ankle_support_plane",
            "enabled": False,
            "skipped_reason": "floor_estimation_failed",
            "error": str(error),
            **thresholds,
        }
        return FloorAlignmentResult(
            R_floor=np.eye(3, dtype=np.float32),
            enabled=False,
            skipped_reason="floor_estimation_failed",
            floor_normal_before=np.full(3, np.nan, dtype=np.float32),
            floor_normal_after=np.full(3, np.nan, dtype=np.float32),
            plane_coefficients=np.full(3, np.nan, dtype=np.float32),
            tilt_before_deg=float("nan"),
            tilt_after_deg=float("nan"),
            residual_median_abs_m=float("nan"),
            robust_support_cloud_count=0,
            support_joint_names=SUPPORT_JOINT_NAMES,
            metrics=metrics,
        )

    tilt_before = float(floor.metrics["tilt_before_deg"])
    residual_median = float(floor.metrics["residual_median_abs_m"])
    skipped_reason: str | None = None
    if tilt_before > config.floor_max_tilt_deg:
        skipped_reason = "floor_tilt_above_threshold"
    elif residual_median > config.floor_max_median_abs_residual_m:
        skipped_reason = "floor_median_residual_above_threshold"

    enabled = skipped_reason is None
    R_floor = _validate_rotation("R_floor", floor.R_floor if enabled else np.eye(3, dtype=np.float64))
    normal_before = np.asarray(floor.floor_normal_before, dtype=np.float64)
    normal_after = R_floor @ normal_before
    normal_norm = float(np.linalg.norm(normal_after))
    if normal_norm > config.normal_eps:
        normal_after = normal_after / normal_norm
    tilt_after = float(np.degrees(np.arccos(np.clip(normal_after[2], -1.0, 1.0))))
    metrics = {
        **floor.metrics,
        "method": "low_foot_ankle_support_plane",
        "enabled": bool(enabled),
        "skipped_reason": skipped_reason,
        **thresholds,
    }
    return FloorAlignmentResult(
        R_floor=R_floor.astype(np.float32),
        enabled=bool(enabled),
        skipped_reason=skipped_reason,
        floor_normal_before=floor.floor_normal_before.astype(np.float32),
        floor_normal_after=normal_after.astype(np.float32),
        plane_coefficients=floor.plane_coefficients.astype(np.float32),
        tilt_before_deg=tilt_before,
        tilt_after_deg=tilt_after,
        residual_median_abs_m=residual_median,
        robust_support_cloud_count=int(floor.metrics["robust_support_cloud_count"]),
        support_joint_names=SUPPORT_JOINT_NAMES,
        metrics=metrics,
    )


def solve_transform(
    joints_3d: np.ndarray,
    trans_canonical: np.ndarray,
    fps: float,
    config: CanonicalizationConfig,
) -> TransformResult:
    joints = validate_joints(joints_3d, name="joints_3d")
    trans = np.asarray(trans_canonical, dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans_canonical must have shape (T, 3), got {trans.shape}")
    if trans.shape[0] != joints.shape[0]:
        raise ValueError(f"joints/trans frame counts differ: {joints.shape[0]} vs {trans.shape[0]}")
    if not np.all(np.isfinite(trans)):
        raise ValueError("trans_canonical contains non-finite values")

    body = estimate_body_frame(joints, config)
    R_body64 = _validate_rotation("R_body", body.R_body)
    joints_body = transform_points(joints, R_body64)
    trans_body = transform_points(trans, R_body64)
    scale = solve_scale_alignment(joints_body, config)
    s_global = float(scale.s_global)
    joints_body_scaled = (s_global * np.asarray(joints_body, dtype=np.float64)).astype(np.float32, copy=False)
    trans_body_scaled = (s_global * np.asarray(trans_body, dtype=np.float64)).astype(np.float32, copy=False)

    floor = solve_floor_alignment(joints_body_scaled, fps, config)
    R_floor64 = _validate_rotation("R_floor", floor.R_floor)
    trans_floor = transform_points(trans_body_scaled, R_floor64)
    body_forward_floor_xy = (R_floor64 @ np.array([1.0, 0.0, 0.0], dtype=np.float64))[:2]
    yaw = solve_yaw_alignment(trans_floor, config, fallback_forward_xy=body_forward_floor_xy)
    R_yaw64 = _validate_rotation("R_yaw", yaw.R_yaw)
    R_global = _validate_rotation("R_global", R_yaw64 @ R_floor64 @ R_body64)

    rotated_trans = transform_points(trans, R_global, scale=s_global)
    rotated_ground_points = transform_points(ground_joint_points(joints), R_global, scale=s_global)
    finite_ground_z = rotated_ground_points[np.isfinite(rotated_ground_points[:, :, 2]), 2]
    if finite_ground_z.size == 0:
        raise ValueError("No finite rotated foot/ankle Z samples are available for vertical translation.")
    ground_z = float(np.percentile(finite_ground_z, config.ground_percentile))

    t_global = np.zeros(3, dtype=np.float64)
    t_global[:2] = -np.asarray(rotated_trans[0, :2], dtype=np.float64)
    t_global[2] = -ground_z

    ground = GroundAlignmentResult(
        ground_z=ground_z,
        percentile=float(config.ground_percentile),
        sample_count=int(finite_ground_z.size),
        support_joint_names=SUPPORT_JOINT_NAMES,
        metrics={
            "method": "foot_ankle_all_samples_low_percentile",
            "finite_z_sample_count": int(finite_ground_z.size),
        },
    )
    metrics = {
        "det_R_global": float(np.linalg.det(R_global)),
        "scale_enabled": bool(scale.enabled),
        "s_global": float(s_global),
        "floor_enabled": bool(floor.enabled),
        "floor_tilt_before_deg": float(floor.tilt_before_deg),
        "floor_tilt_after_deg": float(floor.tilt_after_deg),
        "anchor_first_root_xy": True,
        "ground_percentile": float(config.ground_percentile),
        "ground_z_before_translation": ground_z,
    }
    return TransformResult(
        R_global=R_global.astype(np.float32),
        t_global=t_global.astype(np.float32),
        R_body=R_body64.astype(np.float32),
        R_floor=R_floor64.astype(np.float32),
        R_yaw=R_yaw64.astype(np.float32),
        s_global=float(s_global),
        body=body,
        scale=scale,
        floor=floor,
        yaw=yaw,
        ground=ground,
        metrics=metrics,
    )
