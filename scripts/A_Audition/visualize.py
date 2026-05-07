"""Visualization helpers for A_Audition semantic-coordinate diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

from .config import MPLCONFIGDIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.B_Canonicalization.joints_schema import JOINT_INDEX, validate_joints

from .config import AuditionConfig


def create_semantic_plot(
    joints_3d: np.ndarray,
    trans_canonical: np.ndarray,
    support_points: np.ndarray,
    output_path: str | Path,
    title: str,
    config: AuditionConfig,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    joints = validate_joints(joints_3d, name="joints_3d")
    trans = np.asarray(trans_canonical, dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans_canonical must have shape (T, 3), got {trans.shape}")
    if joints.shape[0] != trans.shape[0]:
        raise ValueError(f"joints/trans frame counts differ: {joints.shape[0]} vs {trans.shape[0]}")
    support = np.asarray(support_points, dtype=np.float64)
    if support.shape != trans.shape:
        raise ValueError(f"support_points must have shape {trans.shape}, got {support.shape}")

    pelvis = joints[:, JOINT_INDEX["pelvis"]]
    neck = joints[:, JOINT_INDEX["neck"]]
    left_hip = joints[:, JOINT_INDEX["left_hip"]]
    right_hip = joints[:, JOINT_INDEX["right_hip"]]
    left_shoulder = joints[:, JOINT_INDEX["left_shoulder"]]
    right_shoulder = joints[:, JOINT_INDEX["right_shoulder"]]

    body_left = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    torso_up = neck - pelvis
    facing = np.cross(body_left, torso_up)
    facing[:, 2] = 0.0
    facing_norm = np.linalg.norm(facing, axis=1)
    valid_facing = facing_norm > 1e-8
    cosine = np.full(facing.shape[0], np.nan, dtype=np.float64)
    cosine[valid_facing] = np.clip(facing[valid_facing, 0] / facing_norm[valid_facing], -1.0, 1.0)
    facing_angle_deg = np.rad2deg(np.arccos(cosine))
    frame_index = np.arange(trans.shape[0], dtype=np.int64)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle(title)

    axes[0].plot(trans[:, 0], trans[:, 1], color="#1f77b4", linewidth=1.2, label="root")
    axes[0].scatter(trans[0, 0], trans[0, 1], color="#2ca02c", s=28, label="start", zorder=3)
    axes[0].scatter(trans[-1, 0], trans[-1, 1], color="#d62728", s=28, label="end", zorder=3)
    axes[0].set_xlabel("X (forward, m)")
    axes[0].set_ylabel("Y (subject-left, m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(trans[:, 0], trans[:, 2], color="#1f77b4", linewidth=1.1, label="root XZ")
    axes[1].plot(support[:, 0], support[:, 2], color="#ff7f0e", linewidth=0.9, label="support XZ")
    axes[1].scatter(trans[0, 0], trans[0, 2], color="#2ca02c", s=22, zorder=3)
    axes[1].scatter(trans[-1, 0], trans[-1, 2], color="#d62728", s=22, zorder=3)
    axes[1].set_xlabel("X (forward, m)")
    axes[1].set_ylabel("Z (up, m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    axes[2].plot(frame_index, facing_angle_deg, color="#9467bd", linewidth=1.0, label="|facing vs +X|")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Angle (deg)")
    axes[2].set_ylim(0.0, 180.0)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper left")

    fig.savefig(output, dpi=config.diagnostic_dpi)
    plt.close(fig)
    return output
