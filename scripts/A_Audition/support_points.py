"""Support-point candidates for A_Audition diagnostics and downstream review."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.B_Canonicalization.joints_schema import JOINT_INDEX, validate_joints

from .config import AuditionConfig


@dataclass(frozen=True)
class SupportPointResult:
    support_points: np.ndarray
    metrics: dict[str, float]
    warnings: tuple[str, ...]


def compute_support_points(
    joints_3d: np.ndarray,
    fps: float,
    config: AuditionConfig,
) -> SupportPointResult:
    joints = validate_joints(joints_3d, name="joints_3d")
    del fps, config

    joint_names = ("left_ankle", "right_ankle", "left_foot", "right_foot")
    candidates = np.stack([joints[:, JOINT_INDEX[name]] for name in joint_names], axis=1)
    selected_indices = np.argmin(candidates[:, :, 2], axis=1)
    support_points = candidates[np.arange(candidates.shape[0]), selected_indices]

    finite_mask = np.all(np.isfinite(support_points), axis=1)
    finite_ratio = float(np.mean(finite_mask)) if finite_mask.size else 0.0

    warnings: list[str] = []
    if finite_ratio < 1.0:
        warnings.append("support_points_nonfinite")

    counts = {
        f"{name}_selected_count": int(np.count_nonzero(selected_indices == index))
        for index, name in enumerate(joint_names)
    }
    ratios = {
        f"{name}_selected_ratio": float(np.mean(selected_indices == index)) if selected_indices.size else 0.0
        for index, name in enumerate(joint_names)
    }
    metrics = {
        "finite_ratio": finite_ratio,
        "min_z": float(np.min(support_points[:, 2])) if support_points.size else 0.0,
        "max_z": float(np.max(support_points[:, 2])) if support_points.size else 0.0,
        **counts,
        **ratios,
    }
    return SupportPointResult(
        support_points=support_points.astype(np.float32, copy=False),
        metrics=metrics,
        warnings=tuple(warnings),
    )
