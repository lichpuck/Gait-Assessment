"""Joint names and helpers for the SMPL 24-joint layout."""

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

JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

LEFT_ANKLE = JOINT_INDEX["left_ankle"]
RIGHT_ANKLE = JOINT_INDEX["right_ankle"]
LEFT_FOOT = JOINT_INDEX["left_foot"]
RIGHT_FOOT = JOINT_INDEX["right_foot"]
ROBUST_SCALE_BONE_PAIRS = (
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def validate_joints(joints: np.ndarray, *, name: str = "joints") -> np.ndarray:
    array = np.asarray(joints, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (24, 3):
        raise ValueError(f"{name} must have shape (T, 24, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def get_joint(joints: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(joints)[:, JOINT_INDEX[name]]


def get_feet(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return get_joint(joints, "left_foot"), get_joint(joints, "right_foot")


def get_ankles(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return get_joint(joints, "left_ankle"), get_joint(joints, "right_ankle")


def bone_pair_labels(bone_pairs: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(f"{parent}-{child}" for parent, child in bone_pairs)


def bone_lengths(joints: np.ndarray, bone_pairs: tuple[tuple[str, str], ...]) -> np.ndarray:
    array = validate_joints(joints, name="joints")
    lengths = []
    for parent, child in bone_pairs:
        parent_index = JOINT_INDEX[parent]
        child_index = JOINT_INDEX[child]
        delta = array[:, child_index] - array[:, parent_index]
        lengths.append(np.linalg.norm(delta, axis=1))
    return np.stack(lengths, axis=1).astype(np.float32, copy=False)


def support_points_by_side(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_foot, right_foot = get_feet(joints)
    left_ankle, right_ankle = get_ankles(joints)
    left = np.where((left_foot[:, 2] <= left_ankle[:, 2])[:, None], left_foot, left_ankle)
    right = np.where((right_foot[:, 2] <= right_ankle[:, 2])[:, None], right_foot, right_ankle)
    return left, right

