"""Diagnostic plotting for simplified B canonicalization."""

from __future__ import annotations

from pathlib import Path
import os

from .config import MPLCONFIGDIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import CanonicalizationConfig
from .joints_schema import JOINT_INDEX, validate_joints


def body_facing_angle_deg(joints_can: np.ndarray) -> np.ndarray:
    joints = validate_joints(joints_can, name="joints_can")
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
    norm = np.linalg.norm(facing[:, :2], axis=1)
    angle = np.full(joints.shape[0], np.nan, dtype=np.float64)
    valid = norm > 1e-8
    angle[valid] = np.degrees(np.arccos(np.clip(facing[valid, 0] / norm[valid], -1.0, 1.0)))
    return angle.astype(np.float32, copy=False)


def create_diagnostic_plot(
    joints_can: np.ndarray,
    trans_can: np.ndarray,
    support_points_can: np.ndarray,
    fps: float,
    title: str,
    output_path: str | Path,
    config: CanonicalizationConfig,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joints = validate_joints(joints_can, name="joints_can")
    trans = np.asarray(trans_can, dtype=np.float64)
    support = np.asarray(support_points_can, dtype=np.float64)
    if trans.shape != (joints.shape[0], 3):
        raise ValueError(f"trans_can must have shape {(joints.shape[0], 3)}, got {trans.shape}")
    if support.shape != trans.shape:
        raise ValueError(f"support_points_can must have shape {trans.shape}, got {support.shape}")

    time_sec = np.arange(joints.shape[0], dtype=np.float64) / float(fps)
    facing_angle = body_facing_angle_deg(joints)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(title)

    axes[0].plot(time_sec, trans[:, 0], color="#1f77b4", linewidth=1.0, label="root X")
    axes[0].plot(time_sec, trans[:, 1], color="#ff7f0e", linewidth=1.0, label="root Y")
    axes[0].axhline(0.0, color="#666666", linewidth=0.7, alpha=0.5)
    axes[0].set_ylabel("Root XY (m)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("root X/Y over time")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_sec, trans[:, 2], color="#1f77b4", linewidth=1.0, label="root Z")
    axes[1].plot(time_sec, support[:, 2], color="#2ca02c", linewidth=0.85, label="support Z")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Z (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("root/support Z over time")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_sec, facing_angle, color="#9467bd", linewidth=1.0, label="|facing vs +X|")
    axes[2].set_ylim(0.0, 180.0)
    axes[2].set_ylabel("Angle (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("body facing angle to +X")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.25)

    fig.savefig(output, dpi=config.diagnostic_dpi)
    plt.close(fig)
    return output

