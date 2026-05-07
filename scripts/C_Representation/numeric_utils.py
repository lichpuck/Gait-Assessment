"""Numeric helpers for C_Representation."""

from __future__ import annotations

import numpy as np


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] <= 1 or fps <= 0:
        return np.zeros_like(array, dtype=np.float32)
    derivative = np.zeros_like(array, dtype=np.float32)
    derivative[1:-1] = (array[2:] - array[:-2]) * (0.5 * float(fps))
    derivative[0] = (array[1] - array[0]) * float(fps)
    derivative[-1] = (array[-1] - array[-2]) * float(fps)
    return derivative.astype(np.float32, copy=False)


def horizontal(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    result = np.array(array, dtype=np.float32, copy=True)
    result[..., 2] = 0.0
    return result


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    valid = norms[..., 0] > float(eps)
    normalized = np.zeros_like(array, dtype=np.float32)
    np.divide(array, np.maximum(norms, eps), out=normalized, where=norms > 0.0)
    return normalized.astype(np.float32, copy=False), valid.astype(bool, copy=False)


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) + 180.0) % 360.0 - 180.0).astype(np.float32, copy=False)


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(values, dtype=np.float32))
    return np.rad2deg(np.unwrap(radians)).astype(np.float32, copy=False)


def heading_from_horizontal_vectors(vectors: np.ndarray) -> np.ndarray:
    array, _ = normalize_vectors(horizontal(vectors))
    return wrap_degrees(np.rad2deg(np.arctan2(array[:, 1], array[:, 0])).astype(np.float32, copy=False))


def axis_angle_to_matrix(axis_angles: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angles, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"axis_angles must have shape (T, 3), got {vectors.shape}")

    frame_count = vectors.shape[0]
    matrices = np.zeros((frame_count, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1.0
    matrices[:, 1, 1] = 1.0
    matrices[:, 2, 2] = 1.0

    angles = np.linalg.norm(vectors, axis=1)
    valid = angles > 1e-8
    if not np.any(valid):
        return matrices

    axis = np.zeros_like(vectors, dtype=np.float32)
    axis[valid] = vectors[valid] / angles[valid, None]
    x = axis[valid, 0]
    y = axis[valid, 1]
    z = axis[valid, 2]
    c = np.cos(angles[valid]).astype(np.float32)
    s = np.sin(angles[valid]).astype(np.float32)
    one_c = (1.0 - c).astype(np.float32)

    rot = np.empty((int(np.count_nonzero(valid)), 3, 3), dtype=np.float32)
    rot[:, 0, 0] = c + x * x * one_c
    rot[:, 0, 1] = x * y * one_c - z * s
    rot[:, 0, 2] = x * z * one_c + y * s
    rot[:, 1, 0] = y * x * one_c + z * s
    rot[:, 1, 1] = c + y * y * one_c
    rot[:, 1, 2] = y * z * one_c - x * s
    rot[:, 2, 0] = z * x * one_c - y * s
    rot[:, 2, 1] = z * y * one_c + x * s
    rot[:, 2, 2] = c + z * z * one_c
    matrices[valid] = rot
    return matrices.astype(np.float32, copy=False)


def matrix_to_euler_xyz_deg(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return roll-X, pitch-Y, yaw-Z degrees from body-to-world rotation matrices."""
    rot = np.asarray(matrices, dtype=np.float32)
    if rot.ndim != 3 or rot.shape[1:] != (3, 3):
        raise ValueError(f"matrices must have shape (T, 3, 3), got {rot.shape}")

    pitch = np.arcsin(np.clip(-rot[:, 2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    regular = np.abs(cos_pitch) > 1e-6

    roll = np.zeros(rot.shape[0], dtype=np.float32)
    yaw = np.zeros(rot.shape[0], dtype=np.float32)
    roll[regular] = np.arctan2(rot[regular, 2, 1], rot[regular, 2, 2])
    yaw[regular] = np.arctan2(rot[regular, 1, 0], rot[regular, 0, 0])
    yaw[~regular] = np.arctan2(-rot[~regular, 0, 1], rot[~regular, 1, 1])

    return (
        wrap_degrees(np.rad2deg(roll).astype(np.float32, copy=False)),
        wrap_degrees(np.rad2deg(pitch).astype(np.float32, copy=False)),
        wrap_degrees(np.rad2deg(yaw).astype(np.float32, copy=False)),
    )


def nearest_valid_fill(values: np.ndarray, valid_mask: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.shape[0] == 0 or np.all(valid):
        return array.astype(np.float32, copy=True)

    filled = np.array(array, dtype=np.float32, copy=True)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        if fallback is None:
            return filled
        fallback_array = np.asarray(fallback, dtype=np.float32)
        filled[...] = fallback_array
        return filled

    first = int(valid_indices[0])
    filled[:first] = filled[first]
    last = first
    for idx in range(first + 1, array.shape[0]):
        if valid[idx]:
            last = idx
            continue
        next_candidates = valid_indices[valid_indices > idx]
        if next_candidates.size == 0:
            filled[idx] = filled[last]
            continue
        next_idx = int(next_candidates[0])
        filled[idx] = filled[last] if (idx - last) <= (next_idx - idx) else filled[next_idx]
    return filled.astype(np.float32, copy=False)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, values.shape[0] - 1))
    return runs


def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if min_length <= 1:
        return result
    for start, end in true_runs(result):
        if end - start + 1 < min_length:
            result[start : end + 1] = False
    return result


def fill_short_false_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if min_length <= 1:
        return result
    for start, end in true_runs(~result):
        if end - start + 1 < min_length:
            result[start : end + 1] = True
    return result


def mask_onsets(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    previous = np.concatenate([np.zeros(1, dtype=bool), values[:-1]])
    return (values & ~previous).astype(np.uint8)


def mask_offsets(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    previous = np.concatenate([np.zeros(1, dtype=bool), values[:-1]])
    return (~values & previous).astype(np.uint8)
