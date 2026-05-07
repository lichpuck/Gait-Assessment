"""SMPL 24-joint schema helpers for representation features."""

from __future__ import annotations

import numpy as np


JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

PELVIS = JOINT_INDEX["pelvis"]
LEFT_HIP = JOINT_INDEX["left_hip"]
RIGHT_HIP = JOINT_INDEX["right_hip"]
LEFT_ANKLE = JOINT_INDEX["left_ankle"]
RIGHT_ANKLE = JOINT_INDEX["right_ankle"]
LEFT_FOOT = JOINT_INDEX["left_foot"]
RIGHT_FOOT = JOINT_INDEX["right_foot"]
NECK = JOINT_INDEX["neck"]
HEAD = JOINT_INDEX["head"]
LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]


def validate_joints(joints: np.ndarray, *, name: str = "joints_can") -> np.ndarray:
    array = np.asarray(joints, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (24, 3):
        raise ValueError(f"{name} must have shape (T, 24, 3), got {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 frames")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.float32, copy=False)


def joint_schema_payload() -> dict[str, object]:
    return {
        "layout": "SMPL_24",
        "joint_names": list(JOINT_NAMES),
        "joint_index": dict(JOINT_INDEX),
        "key_joints": {
            "pelvis": PELVIS,
            "left_hip": LEFT_HIP,
            "right_hip": RIGHT_HIP,
            "left_ankle": LEFT_ANKLE,
            "right_ankle": RIGHT_ANKLE,
            "left_foot": LEFT_FOOT,
            "right_foot": RIGHT_FOOT,
            "neck": NECK,
            "head": HEAD,
            "left_shoulder": LEFT_SHOULDER,
            "right_shoulder": RIGHT_SHOULDER,
        },
    }


def support_points_by_side(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = validate_joints(joints)
    left_foot = array[:, LEFT_FOOT]
    right_foot = array[:, RIGHT_FOOT]
    left_ankle = array[:, LEFT_ANKLE]
    right_ankle = array[:, RIGHT_ANKLE]
    left = np.where((left_foot[:, 2] <= left_ankle[:, 2])[:, None], left_foot, left_ankle)
    right = np.where((right_foot[:, 2] <= right_ankle[:, 2])[:, None], right_foot, right_ankle)
    return left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)
