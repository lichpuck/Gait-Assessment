"""Infer raw coordinate-axis semantics and build a semantic rotation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.B_Canonicalization.joints_schema import JOINT_INDEX, validate_joints

from .config import AuditionConfig


AXIS_NAMES = ("X", "Y", "Z")


class AxisSemanticsError(ValueError):
    """Raised when a sequence cannot be assigned a valid semantic frame."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class AxisSemanticsResult:
    R_total: np.ndarray
    forward_raw_axis: int
    lateral_raw_axis: int
    vertical_raw_axis: int
    forward_sign: float
    lateral_sign: float
    vertical_sign: float
    raw_axis_semantics: dict[str, str]
    raw_ranges: dict[str, float]
    robust_ranges: dict[str, float]
    range_order: tuple[str, ...]
    vertical_alignment: float
    lateral_alignment: float
    forward_axis_margin: float
    forward_segment: dict[str, object]
    warnings: tuple[str, ...]
    quality_flags: dict[str, bool]

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.R_total))


def apply_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.einsum("...i,ji->...j", array, rot)


def _validate_trans(trans_raw: np.ndarray) -> np.ndarray:
    trans = np.asarray(trans_raw, dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans_raw must have shape (T, 3), got {trans.shape}")
    if not np.all(np.isfinite(trans)):
        raise ValueError("trans_raw contains non-finite values")
    return trans


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0 or window <= 1 or values.size < window:
        return values
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _robust_ranges(trans: np.ndarray, config: AuditionConfig) -> tuple[np.ndarray, np.ndarray]:
    low = float(config.robust_range_low_percentile)
    high = float(config.robust_range_high_percentile)
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"Invalid robust range percentiles: {low}, {high}")
    raw = np.ptp(trans, axis=0)
    robust = np.percentile(trans, high, axis=0) - np.percentile(trans, low, axis=0)
    return raw.astype(np.float64), robust.astype(np.float64)


def _dominant_axis(vector: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(vector, dtype=np.float64).reshape(3)
    abs_values = np.abs(values)
    order = np.argsort(abs_values)[::-1]
    axis = int(order[0])
    norm = float(np.linalg.norm(values))
    alignment = float(abs_values[axis] / norm) if norm > 1e-8 and np.isfinite(norm) else 0.0
    sign = 1.0 if values[axis] >= 0.0 else -1.0
    return axis, sign, alignment


def _axis_margin(values: np.ndarray, axis: int) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    if ordered.size < 2:
        return 1.0
    denom = max(float(ordered[0]), 1e-8)
    return float((ordered[0] - ordered[1]) / denom)


def _axis_margin_among(values: np.ndarray, axis: int, candidates: tuple[int, ...]) -> float:
    selected = float(np.asarray(values, dtype=np.float64)[int(axis)])
    others = [float(np.asarray(values, dtype=np.float64)[int(item)]) for item in candidates if int(item) != int(axis)]
    if not others:
        return 1.0
    runner_up = max(others)
    denom = max(selected, 1e-8)
    return float((selected - runner_up) / denom)


def _longest_monotonic_segment(
    projection: np.ndarray,
    config: AuditionConfig,
) -> dict[str, object]:
    smoothed = _moving_average(projection, config.smoothing_window_frames)
    if smoothed.size < 2:
        return {
            "start_frame": 0,
            "end_frame": 0,
            "signed_displacement": 0.0,
            "abs_displacement": 0.0,
            "length_frames": int(smoothed.size),
            "direction": "positive",
            "method": "longest_monotonic_smoothed_projection",
        }

    diffs = np.diff(smoothed)
    eps = max(float(np.nanpercentile(np.abs(diffs), 20)) * 0.25, 1e-6)
    signs = np.zeros_like(diffs, dtype=np.int8)
    signs[diffs > eps] = 1
    signs[diffs < -eps] = -1

    best: dict[str, object] | None = None
    for target_sign in (1, -1):
        start = 0
        while start < signs.size:
            while start < signs.size and signs[start] not in (0, target_sign):
                start += 1
            if start >= signs.size:
                break
            end = start
            while end < signs.size and signs[end] in (0, target_sign):
                end += 1
            start_frame = int(start)
            end_frame = int(end)
            displacement = float(smoothed[end_frame] - smoothed[start_frame])
            candidate = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "signed_displacement": displacement,
                "abs_displacement": abs(displacement),
                "length_frames": int(end_frame - start_frame + 1),
                "direction": "positive" if displacement >= 0.0 else "negative",
                "method": "longest_monotonic_smoothed_projection",
            }
            if best is None:
                best = candidate
            else:
                best_key = (float(best["abs_displacement"]), int(best["length_frames"]))
                candidate_key = (float(candidate["abs_displacement"]), int(candidate["length_frames"]))
                if candidate_key > best_key:
                    best = candidate
            start = end

    if best is None:
        start_frame = 0
        end_frame = int(smoothed.size - 1)
        displacement = float(smoothed[end_frame] - smoothed[start_frame])
        best = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "signed_displacement": displacement,
            "abs_displacement": abs(displacement),
            "length_frames": int(smoothed.size),
            "direction": "positive" if displacement >= 0.0 else "negative",
            "method": "fallback_full_sequence_projection",
        }
    return best


def _basis_row(raw_axis: int, sign: float) -> np.ndarray:
    row = np.zeros(3, dtype=np.float64)
    row[int(raw_axis)] = 1.0 if sign >= 0.0 else -1.0
    return row


def estimate_axis_semantics(
    joints_raw: np.ndarray,
    trans_raw: np.ndarray,
    config: AuditionConfig,
) -> AxisSemanticsResult:
    joints = validate_joints(joints_raw, name="joints_raw")
    trans = _validate_trans(trans_raw)
    if joints.shape[0] != trans.shape[0]:
        raise ValueError(f"joints/trans frame counts differ: {joints.shape[0]} vs {trans.shape[0]}")

    raw_range_values, robust_range_values = _robust_ranges(trans, config)
    range_order_indices = tuple(int(i) for i in np.argsort(robust_range_values)[::-1])
    initial_forward_axis = range_order_indices[0]

    pelvis = joints[:, JOINT_INDEX["pelvis"]]
    head = joints[:, JOINT_INDEX["head"]]
    neck = joints[:, JOINT_INDEX["neck"]]
    left_hip = joints[:, JOINT_INDEX["left_hip"]]
    right_hip = joints[:, JOINT_INDEX["right_hip"]]

    vertical_vector = np.median(head - pelvis, axis=0)
    vertical_source = "pelvis_to_head"
    if not np.all(np.isfinite(vertical_vector)) or float(np.linalg.norm(vertical_vector)) < 1e-8:
        vertical_vector = np.median(neck - pelvis, axis=0)
        vertical_source = "pelvis_to_neck"
    vertical_axis, vertical_sign, vertical_alignment = _dominant_axis(vertical_vector)

    horizontal_candidates = tuple(index for index in range_order_indices if index != vertical_axis)
    forward_axis = int(horizontal_candidates[0])
    lateral_axis = int(horizontal_candidates[1])
    forward_margin = _axis_margin_among(robust_range_values, forward_axis, horizontal_candidates)
    axis_conflict_resolved = initial_forward_axis == vertical_axis

    lateral_vector = np.median(left_hip - right_hip, axis=0)
    lateral_component = float(lateral_vector[lateral_axis])
    lateral_sign = 1.0 if lateral_component >= 0.0 else -1.0
    lateral_norm = float(np.linalg.norm(lateral_vector))
    lateral_alignment = abs(lateral_component) / lateral_norm if lateral_norm > 1e-8 and np.isfinite(lateral_norm) else 0.0

    projected = trans[:, forward_axis]
    segment = _longest_monotonic_segment(projected, config)
    segment_forward_sign = 1.0 if float(segment["signed_displacement"]) >= 0.0 else -1.0
    forward_sign = segment_forward_sign

    rotation = np.stack(
        [
            _basis_row(forward_axis, forward_sign),
            _basis_row(lateral_axis, lateral_sign),
            _basis_row(vertical_axis, vertical_sign),
        ],
        axis=0,
    )
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        forward_sign *= -1.0
        rotation = np.stack(
            [
                _basis_row(forward_axis, forward_sign),
                _basis_row(lateral_axis, lateral_sign),
                _basis_row(vertical_axis, vertical_sign),
            ],
            axis=0,
        )
        determinant = float(np.linalg.det(rotation))
        if determinant <= 0.0:
            raise AxisSemanticsError(
                "handedness_conflict_failed",
                f"signed axis mapping is not right-handed after forward flip; det={determinant:.3f}",
            )

    warnings: list[str] = []
    if axis_conflict_resolved:
        warnings.append("axis_conflict_resolved_by_vertical_priority")
    if forward_sign != segment_forward_sign:
        warnings.append("forward_sign_flipped_for_right_handed_frame")
    if forward_margin < config.min_forward_axis_margin:
        warnings.append("forward_axis_low_margin")
    if vertical_alignment < config.min_vertical_alignment:
        warnings.append("vertical_axis_low_alignment")
    if lateral_alignment < config.min_lateral_alignment:
        warnings.append("lateral_axis_low_alignment")
    if float(segment["abs_displacement"]) < config.min_forward_segment_displacement_m:
        warnings.append("forward_segment_displacement_small")

    raw_axis_semantics = {
        AXIS_NAMES[forward_axis]: f"{'+' if forward_sign > 0 else '-'}forward",
        AXIS_NAMES[lateral_axis]: f"{'+' if lateral_sign > 0 else '-'}subject_left",
        AXIS_NAMES[vertical_axis]: f"{'+' if vertical_sign > 0 else '-'}vertical_up",
    }
    segment = {
        **segment,
        "raw_axis": AXIS_NAMES[forward_axis],
        "initial_forward_raw_axis": AXIS_NAMES[initial_forward_axis],
        "axis_conflict_resolved": axis_conflict_resolved,
        "segment_forward_sign": "positive" if segment_forward_sign > 0 else "negative_flip",
        "applied_sign": "positive" if forward_sign > 0 else "negative_flip",
    }

    quality_flags = {
        "axis_conflict_resolved_by_vertical_priority": axis_conflict_resolved,
        "forward_sign_flipped_for_right_handed_frame": forward_sign != segment_forward_sign,
        "low_forward_axis_margin": forward_margin < config.min_forward_axis_margin,
        "low_vertical_alignment": vertical_alignment < config.min_vertical_alignment,
        "low_lateral_alignment": lateral_alignment < config.min_lateral_alignment,
        "low_forward_segment_displacement": float(segment["abs_displacement"])
        < config.min_forward_segment_displacement_m,
        "rotation_is_right_handed": True,
    }

    return AxisSemanticsResult(
        R_total=rotation.astype(np.float32, copy=False),
        forward_raw_axis=forward_axis,
        lateral_raw_axis=lateral_axis,
        vertical_raw_axis=vertical_axis,
        forward_sign=forward_sign,
        lateral_sign=lateral_sign,
        vertical_sign=vertical_sign,
        raw_axis_semantics=raw_axis_semantics,
        raw_ranges={AXIS_NAMES[i]: float(raw_range_values[i]) for i in range(3)},
        robust_ranges={AXIS_NAMES[i]: float(robust_range_values[i]) for i in range(3)},
        range_order=tuple(AXIS_NAMES[i] for i in range_order_indices),
        vertical_alignment=vertical_alignment,
        lateral_alignment=lateral_alignment,
        forward_axis_margin=forward_margin,
        forward_segment={**segment, "vertical_source": vertical_source},
        warnings=tuple(warnings),
        quality_flags=quality_flags,
    )
