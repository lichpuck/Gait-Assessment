"""Segment-level feature extraction for walk and turn segments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from scripts.C_Representation.joints_schema import JOINT_INDEX
from scripts.C_Representation.numeric_utils import axis_angle_to_matrix, matrix_to_euler_xyz_deg, true_runs, unwrap_degrees, wrap_degrees

from .config import ExtractionConfig, seconds_to_frames
from .io_utils import ExtractionSequence, MotionSegment


SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)

PELVIS = JOINT_INDEX["pelvis"]
LEFT_HIP = JOINT_INDEX["left_hip"]
RIGHT_HIP = JOINT_INDEX["right_hip"]
LEFT_KNEE = JOINT_INDEX["left_knee"]
RIGHT_KNEE = JOINT_INDEX["right_knee"]
LEFT_ANKLE = JOINT_INDEX["left_ankle"]
RIGHT_ANKLE = JOINT_INDEX["right_ankle"]
LEFT_FOOT = JOINT_INDEX["left_foot"]
RIGHT_FOOT = JOINT_INDEX["right_foot"]
NECK = JOINT_INDEX["neck"]
SPINE3 = JOINT_INDEX["spine3"]
HEAD = JOINT_INDEX["head"]
LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
LEFT_WRIST = JOINT_INDEX["left_wrist"]
RIGHT_WRIST = JOINT_INDEX["right_wrist"]

COMMON_ROW_COLUMNS = (
    "subset",
    "subject_id",
    "trial_id",
    "sequence_name",
    "segment_id",
    "label",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "source_rule",
    "confidence",
    "quality_status",
    "quality_valid_frame_ratio",
    "quality_representation_score_mean",
    "quality_hesitation_overlap_ratio",
    "quality_reasons",
)

WALK_FEATURE_COLUMNS = (
    "gait_speed_mps",
    "cadence_steps_per_min",
    "step_time_mean_sec",
    "stride_time_mean_sec",
    "stance_time_mean_sec",
    "swing_time_mean_sec",
    "double_support_time_mean_sec",
    "double_support_percent",
    "step_time_variability_cv_percent",
    "stride_time_variability_cv_percent",
    "left_step_length_mean_m",
    "right_step_length_mean_m",
    "step_length_mean_m",
    "step_length_asymmetry_percent",
    "left_stride_length_mean_m",
    "right_stride_length_mean_m",
    "stride_length_mean_m",
    "stride_length_asymmetry_percent",
    "left_hip_rom_deg",
    "right_hip_rom_deg",
    "hip_rom_mean_deg",
    "hip_rom_asymmetry_percent",
    "left_knee_rom_deg",
    "right_knee_rom_deg",
    "knee_rom_mean_deg",
    "knee_rom_asymmetry_percent",
    "left_ankle_rom_deg",
    "right_ankle_rom_deg",
    "ankle_rom_mean_deg",
    "ankle_rom_asymmetry_percent",
    "left_arm_swing_amplitude_deg",
    "right_arm_swing_amplitude_deg",
    "arm_swing_amplitude_mean_deg",
    "arm_swing_asymmetry_percent",
    "trunk_flexion_mean_deg",
    "trunk_flexion_rom_deg",
    "pelvis_rotation_rom_deg",
    "trunk_ml_sway_m",
)

TURN_FEATURE_COLUMNS = (
    "turn_angle_deg",
    "turn_duration_sec",
    "turn_step_count",
    "mean_step_time_during_turn_sec",
    "pre_turn_hesitation_time_sec",
    "mean_turn_angular_velocity_deg_s",
    "peak_turn_angular_velocity_deg_s",
    "turn_angular_velocity_variability_cv_percent",
    "turn_path_radius_m",
    "turn_path_compactness_deg_per_m",
    "mean_step_length_during_turn_m",
    "trunk_yaw_rom_during_turn_deg",
    "pelvis_yaw_rom_during_turn_deg",
    "head_trunk_pelvis_reorientation_delay_sec",
    "trunk_pelvis_reorientation_delay_sec",
    "head_trunk_reorientation_delay_sec",
    "en_bloc_index",
    "trunk_lateral_lean_during_turn_deg",
    "pelvis_ml_excursion_during_turn_m",
)


WALK_FEATURE_DEFINITIONS = {
    "gait_speed_mps": "Segment horizontal pelvis path length divided by segment duration.",
    "cadence_steps_per_min": "Heel-strike count within the segment divided by duration and scaled to steps per minute.",
    "step_time_mean_sec": "Mean interval between consecutive contralateral heel strikes within the segment.",
    "stride_time_mean_sec": "Mean interval between consecutive ipsilateral heel strikes within the segment.",
    "stance_time_mean_sec": "Mean heel-strike to next toe-off duration across both feet.",
    "swing_time_mean_sec": "Mean toe-off to next heel-strike duration across both feet.",
    "double_support_time_mean_sec": "Mean contiguous duration of frames where both feet are in contact.",
    "double_support_percent": "Percentage of segment frames labeled as double support.",
    "step_time_variability_cv_percent": "Coefficient of variation of step times, expressed as percent.",
    "stride_time_variability_cv_percent": "Coefficient of variation of stride times, expressed as percent.",
    "left_step_length_mean_m": "Mean left step length from heel-strike foot separation projected on +X progression.",
    "right_step_length_mean_m": "Mean right step length from heel-strike foot separation projected on +X progression.",
    "step_length_mean_m": "Mean of left and right step lengths.",
    "step_length_asymmetry_percent": "Normalized absolute left-right difference in mean step length, expressed as percent.",
    "left_stride_length_mean_m": "Mean left stride length from successive left heel-strike foot positions along +X.",
    "right_stride_length_mean_m": "Mean right stride length from successive right heel-strike foot positions along +X.",
    "stride_length_mean_m": "Mean of left and right stride lengths.",
    "stride_length_asymmetry_percent": "Normalized absolute left-right difference in mean stride length, expressed as percent.",
    "left_hip_rom_deg": "Robust sagittal-plane hip excursion, defined as P95-P5 of thigh angle relative to trunk-down.",
    "right_hip_rom_deg": "Robust sagittal-plane hip excursion, defined as P95-P5 of thigh angle relative to trunk-down.",
    "hip_rom_mean_deg": "Mean of left and right hip ROM values.",
    "hip_rom_asymmetry_percent": "Normalized absolute left-right difference in hip ROM, expressed as percent.",
    "left_knee_rom_deg": "Robust sagittal-plane knee flexion excursion, defined as P95-P5 of knee flexion angle.",
    "right_knee_rom_deg": "Robust sagittal-plane knee flexion excursion, defined as P95-P5 of knee flexion angle.",
    "knee_rom_mean_deg": "Mean of left and right knee ROM values.",
    "knee_rom_asymmetry_percent": "Normalized absolute left-right difference in knee ROM, expressed as percent.",
    "left_ankle_rom_deg": "Robust sagittal-plane ankle relative-angle excursion, defined as P95-P5 of foot-shank angle.",
    "right_ankle_rom_deg": "Robust sagittal-plane ankle relative-angle excursion, defined as P95-P5 of foot-shank angle.",
    "ankle_rom_mean_deg": "Mean of left and right ankle ROM values.",
    "ankle_rom_asymmetry_percent": "Normalized absolute left-right difference in ankle ROM, expressed as percent.",
    "left_arm_swing_amplitude_deg": "Robust sagittal excursion of the shoulder-to-wrist vector relative to trunk-down, P95-P5.",
    "right_arm_swing_amplitude_deg": "Robust sagittal excursion of the shoulder-to-wrist vector relative to trunk-down, P95-P5.",
    "arm_swing_amplitude_mean_deg": "Mean of left and right arm swing amplitudes.",
    "arm_swing_asymmetry_percent": "Normalized absolute left-right difference in arm swing amplitude, expressed as percent.",
    "trunk_flexion_mean_deg": "Mean of C_Representation trunk_forward_flexion_deg over valid frames.",
    "trunk_flexion_rom_deg": "Robust range P95-P5 of trunk_forward_flexion_deg over valid frames.",
    "pelvis_rotation_rom_deg": "Robust range P95-P5 of pelvis_yaw_deg over valid frames.",
    "trunk_ml_sway_m": "Standard deviation of detrended neck lateral displacement over the segment.",
}


TURN_FEATURE_DEFINITIONS = {
    "turn_angle_deg": "Absolute heading change from segment start to end using heading_unwrapped_deg.",
    "turn_duration_sec": "Segment duration copied from D_Segmentation.",
    "turn_step_count": "Heel-strike count within the turn segment.",
    "mean_step_time_during_turn_sec": "Mean contralateral heel-strike interval inside the turn segment.",
    "pre_turn_hesitation_time_sec": "Contiguous low-speed duration immediately before turn onset, using root_speed_xy_mps < 0.10 m/s within a 1 s lookback.",
    "mean_turn_angular_velocity_deg_s": "Mean absolute yaw_rate_deg_s during the turn.",
    "peak_turn_angular_velocity_deg_s": "Peak absolute yaw_rate_deg_s during the turn.",
    "turn_angular_velocity_variability_cv_percent": "Coefficient of variation of absolute yaw_rate_deg_s, expressed as percent.",
    "turn_path_radius_m": "Pelvis horizontal path length divided by turn angle in radians.",
    "turn_path_compactness_deg_per_m": "Turn angle divided by pelvis horizontal path length.",
    "mean_step_length_during_turn_m": "Mean horizontal foot separation at heel strikes within the turn.",
    "trunk_yaw_rom_during_turn_deg": "Robust range P95-P5 of trunk_yaw_deg during the turn.",
    "pelvis_yaw_rom_during_turn_deg": "Robust range P95-P5 of pelvis_yaw_deg during the turn.",
    "head_trunk_pelvis_reorientation_delay_sec": "Head yaw onset time minus pelvis yaw onset time during the turn, using a 10% turn-angle onset criterion.",
    "trunk_pelvis_reorientation_delay_sec": "Trunk yaw onset time minus pelvis yaw onset time during the turn.",
    "head_trunk_reorientation_delay_sec": "Head yaw onset time minus trunk yaw onset time during the turn.",
    "en_bloc_index": "One minus the mean absolute pelvis-trunk-head yaw separation normalized by total turn angle, clipped to [0, 1].",
    "trunk_lateral_lean_during_turn_deg": "Mean absolute trunk_lateral_lean_deg during the turn.",
    "pelvis_ml_excursion_during_turn_m": "Robust range P95-P5 of pelvis lateral position during the turn.",
}


@dataclass(frozen=True)
class PoseOrientationSignals:
    available: bool
    pelvis_yaw_unwrapped_deg: np.ndarray | None
    trunk_yaw_unwrapped_deg: np.ndarray | None
    head_yaw_unwrapped_deg: np.ndarray | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SegmentFeatureResult:
    row: dict[str, Any]
    features: dict[str, float]
    missing_features: dict[str, str]


def _segment_slice(segment: MotionSegment) -> slice:
    return slice(int(segment.start_frame), int(segment.end_frame) + 1)


def _valid_segment_mask(sequence: ExtractionSequence, segment: MotionSegment) -> np.ndarray:
    return np.asarray(sequence.arrays["valid_frame_mask"][_segment_slice(segment)], dtype=bool)


def _nanfloat(value: float | np.ndarray | None = None) -> float:
    if value is None:
        return float("nan")
    scalar = float(value)
    return scalar if math.isfinite(scalar) else float("nan")


def _masked_series(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.shape[0] != valid.shape[0]:
        raise ValueError("mask length mismatch")
    finite = np.isfinite(array)
    return array[valid & finite].astype(np.float32, copy=False)


def _robust_range(values: np.ndarray, config: ExtractionConfig) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(
        np.percentile(finite, config.robust_range_high_percentile)
        - np.percentile(finite, config.robust_range_low_percentile)
    )


def _coefficient_of_variation_percent(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    mean = float(np.mean(finite))
    if not math.isfinite(mean) or abs(mean) < 1e-8:
        return float("nan")
    return float(np.std(finite) / abs(mean) * 100.0)


def _normalized_asymmetry(left: float, right: float, config: ExtractionConfig) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    denominator = max((abs(left) + abs(right)) * 0.5, config.asymmetry_epsilon)
    return float(abs(left - right) / denominator * 100.0)


def _ordered_heel_strikes(sequence: ExtractionSequence, segment: MotionSegment) -> list[tuple[int, str]]:
    valid_frames = np.asarray(sequence.arrays["valid_frame_mask"], dtype=bool)
    left_frames = np.flatnonzero(np.asarray(sequence.arrays["left_heel_strike"], dtype=bool))
    right_frames = np.flatnonzero(np.asarray(sequence.arrays["right_heel_strike"], dtype=bool))
    events = [
        (int(frame), "left")
        for frame in left_frames
        if segment.start_frame <= int(frame) <= segment.end_frame and bool(valid_frames[int(frame)])
    ]
    events.extend(
        (int(frame), "right")
        for frame in right_frames
        if segment.start_frame <= int(frame) <= segment.end_frame and bool(valid_frames[int(frame)])
    )
    return sorted(events, key=lambda item: (item[0], item[1]))


def _ipsilateral_event_frames(sequence: ExtractionSequence, segment: MotionSegment, event_key: str) -> np.ndarray:
    valid_frames = np.asarray(sequence.arrays["valid_frame_mask"], dtype=bool)
    frames = np.flatnonzero(np.asarray(sequence.arrays[event_key], dtype=bool))
    filtered = [
        int(frame)
        for frame in frames
        if segment.start_frame <= int(frame) <= segment.end_frame and bool(valid_frames[int(frame)])
    ]
    return np.asarray(filtered, dtype=np.int32)


def _contralateral_intervals_sec(events: list[tuple[int, str]], fps: float) -> np.ndarray:
    intervals: list[float] = []
    for (frame_a, side_a), (frame_b, side_b) in zip(events[:-1], events[1:]):
        if side_a == side_b:
            continue
        delta = (int(frame_b) - int(frame_a)) / float(fps)
        if delta > 0.0:
            intervals.append(delta)
    return np.asarray(intervals, dtype=np.float32)


def _same_side_intervals_sec(frames: np.ndarray, fps: float) -> np.ndarray:
    if frames.size < 2:
        return np.asarray([], dtype=np.float32)
    deltas = np.diff(np.asarray(frames, dtype=np.int32)).astype(np.float32) / float(fps)
    return deltas[deltas > 0.0].astype(np.float32, copy=False)


def _phase_durations_sec(start_frames: np.ndarray, stop_frames: np.ndarray, fps: float) -> np.ndarray:
    starts = np.asarray(start_frames, dtype=np.int32)
    stops = np.asarray(stop_frames, dtype=np.int32)
    durations: list[float] = []
    for start in starts:
        candidates = stops[stops > start]
        if candidates.size == 0:
            continue
        duration = (int(candidates[0]) - int(start)) / float(fps)
        if duration > 0.0:
            durations.append(duration)
    return np.asarray(durations, dtype=np.float32)


def _double_support_durations_sec(gait_phase: np.ndarray, fps: float) -> np.ndarray:
    mask = np.asarray(gait_phase, dtype=np.int32) == 3
    return np.asarray([(end - start + 1) / float(fps) for start, end in true_runs(mask)], dtype=np.float32)


def _path_length_xy(root_xy: np.ndarray, valid_mask: np.ndarray) -> float:
    points = np.asarray(root_xy, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.shape[0] < 2:
        return float("nan")
    diffs = np.diff(points, axis=0)
    valid_pairs = valid[1:] & valid[:-1]
    if not np.any(valid_pairs):
        return float("nan")
    return float(np.sum(np.linalg.norm(diffs[valid_pairs], axis=1)))


def _sagittal_orientation_deg(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    return np.rad2deg(np.arctan2(array[..., 0], -array[..., 2])).astype(np.float32, copy=False)


def _relative_sagittal_angle_deg(vectors: np.ndarray, reference_vectors: np.ndarray) -> np.ndarray:
    return wrap_degrees(_sagittal_orientation_deg(vectors) - _sagittal_orientation_deg(reference_vectors))


def _project_sagittal(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    projected = np.zeros_like(array, dtype=np.float32)
    projected[..., 0] = array[..., 0]
    projected[..., 2] = array[..., 2]
    return projected.astype(np.float32, copy=False)


def _internal_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    vector_a = _project_sagittal(first)
    vector_b = _project_sagittal(second)
    dot = np.sum(vector_a * vector_b, axis=-1)
    norm_a = np.linalg.norm(vector_a, axis=-1)
    norm_b = np.linalg.norm(vector_b, axis=-1)
    denominator = np.maximum(norm_a * norm_b, 1e-8)
    cosine = np.clip(dot / denominator, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float32, copy=False)


def _hip_angle_series_deg(joints: np.ndarray, side: str) -> np.ndarray:
    hip_index = LEFT_HIP if side == "left" else RIGHT_HIP
    knee_index = LEFT_KNEE if side == "left" else RIGHT_KNEE
    thigh = joints[:, knee_index] - joints[:, hip_index]
    trunk_down = joints[:, PELVIS] - joints[:, NECK]
    return _relative_sagittal_angle_deg(thigh, trunk_down)


def _knee_flexion_series_deg(joints: np.ndarray, side: str) -> np.ndarray:
    hip_index = LEFT_HIP if side == "left" else RIGHT_HIP
    knee_index = LEFT_KNEE if side == "left" else RIGHT_KNEE
    ankle_index = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    thigh = joints[:, hip_index] - joints[:, knee_index]
    shank = joints[:, ankle_index] - joints[:, knee_index]
    return (180.0 - _internal_angle_deg(thigh, shank)).astype(np.float32, copy=False)


def _ankle_relative_series_deg(joints: np.ndarray, side: str) -> np.ndarray:
    knee_index = LEFT_KNEE if side == "left" else RIGHT_KNEE
    ankle_index = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    foot_index = LEFT_FOOT if side == "left" else RIGHT_FOOT
    shank = joints[:, knee_index] - joints[:, ankle_index]
    foot = joints[:, foot_index] - joints[:, ankle_index]
    return _relative_sagittal_angle_deg(foot, shank)


def _arm_swing_series_deg(joints: np.ndarray, side: str) -> np.ndarray:
    shoulder_index = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    wrist_index = LEFT_WRIST if side == "left" else RIGHT_WRIST
    arm = joints[:, wrist_index] - joints[:, shoulder_index]
    trunk_down = joints[:, PELVIS] - joints[:, NECK]
    return _relative_sagittal_angle_deg(arm, trunk_down)


def _step_lengths_walk_m(sequence: ExtractionSequence, segment: MotionSegment) -> tuple[np.ndarray, np.ndarray]:
    left_frames = _ipsilateral_event_frames(sequence, segment, "left_heel_strike")
    right_frames = _ipsilateral_event_frames(sequence, segment, "right_heel_strike")
    left_foot = np.asarray(sequence.arrays["left_foot_pos_m"], dtype=np.float32)
    right_foot = np.asarray(sequence.arrays["right_foot_pos_m"], dtype=np.float32)
    left_lengths = np.asarray([abs(float(left_foot[frame, 0] - right_foot[frame, 0])) for frame in left_frames], dtype=np.float32)
    right_lengths = np.asarray([abs(float(right_foot[frame, 0] - left_foot[frame, 0])) for frame in right_frames], dtype=np.float32)
    return left_lengths, right_lengths


def _stride_lengths_walk_m(sequence: ExtractionSequence, segment: MotionSegment) -> tuple[np.ndarray, np.ndarray]:
    left_frames = _ipsilateral_event_frames(sequence, segment, "left_heel_strike")
    right_frames = _ipsilateral_event_frames(sequence, segment, "right_heel_strike")
    left_foot = np.asarray(sequence.arrays["left_foot_pos_m"], dtype=np.float32)
    right_foot = np.asarray(sequence.arrays["right_foot_pos_m"], dtype=np.float32)
    left_lengths = np.asarray(
        [abs(float(left_foot[next_frame, 0] - left_foot[frame, 0])) for frame, next_frame in zip(left_frames[:-1], left_frames[1:])],
        dtype=np.float32,
    )
    right_lengths = np.asarray(
        [abs(float(right_foot[next_frame, 0] - right_foot[frame, 0])) for frame, next_frame in zip(right_frames[:-1], right_frames[1:])],
        dtype=np.float32,
    )
    return left_lengths, right_lengths


def _step_lengths_turn_m(sequence: ExtractionSequence, segment: MotionSegment) -> np.ndarray:
    events = _ordered_heel_strikes(sequence, segment)
    left_foot = np.asarray(sequence.arrays["left_foot_pos_m"], dtype=np.float32)
    right_foot = np.asarray(sequence.arrays["right_foot_pos_m"], dtype=np.float32)
    lengths: list[float] = []
    for frame, _side in events:
        lengths.append(float(np.linalg.norm(left_foot[frame, :2] - right_foot[frame, :2])))
    return np.asarray(lengths, dtype=np.float32)


def _mean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _first_onset_time_sec(
    yaw_unwrapped_deg: np.ndarray | None,
    time_sec: np.ndarray,
    config: ExtractionConfig,
) -> float:
    if yaw_unwrapped_deg is None:
        return float("nan")
    yaw = np.asarray(yaw_unwrapped_deg, dtype=np.float32)
    if yaw.size == 0 or time_sec.size != yaw.size:
        return float("nan")
    total_angle = float(abs(yaw[-1] - yaw[0]))
    if not math.isfinite(total_angle) or total_angle < config.reorientation_min_total_angle_deg:
        return float("nan")
    threshold = config.reorientation_onset_fraction * total_angle
    progress = np.abs(yaw - yaw[0])
    indices = np.flatnonzero(progress >= threshold)
    if indices.size == 0:
        return float("nan")
    return float(time_sec[int(indices[0])] - time_sec[0])


def build_pose_orientation_signals(sequence: ExtractionSequence, config: ExtractionConfig) -> PoseOrientationSignals:
    warnings: list[str] = []
    if not config.allow_b_pose_fallback:
        return PoseOrientationSignals(False, None, None, None, tuple(warnings))
    if sequence.pose_raw is None:
        warnings.append("pose_raw_unavailable_for_head_orientation_features")
        return PoseOrientationSignals(False, None, None, None, tuple(warnings))
    if sequence.r_global is None or sequence.r_total is None:
        warnings.append("canonical_rotation_metadata_unavailable_for_head_orientation_features")
        return PoseOrientationSignals(False, None, None, None, tuple(warnings))

    frame_count = sequence.pose_raw.shape[0]
    pose = np.asarray(sequence.pose_raw, dtype=np.float32).reshape(frame_count, 24, 3)
    local_rot = axis_angle_to_matrix(pose.reshape(-1, 3)).reshape(frame_count, 24, 3, 3)
    global_rot = np.zeros((frame_count, 24, 3, 3), dtype=np.float32)
    canonical_root = np.asarray(sequence.r_global @ sequence.r_total, dtype=np.float32)
    global_rot[:, 0] = np.einsum("ij,tjk->tik", canonical_root, local_rot[:, 0])
    for joint_index in range(1, 24):
        parent = int(SMPL_PARENTS[joint_index])
        global_rot[:, joint_index] = np.einsum("tij,tjk->tik", global_rot[:, parent], local_rot[:, joint_index])

    pelvis_yaw = unwrap_degrees(matrix_to_euler_xyz_deg(global_rot[:, PELVIS])[2])
    trunk_yaw = unwrap_degrees(matrix_to_euler_xyz_deg(global_rot[:, SPINE3])[2])
    head_yaw = unwrap_degrees(matrix_to_euler_xyz_deg(global_rot[:, HEAD])[2])
    return PoseOrientationSignals(True, pelvis_yaw, trunk_yaw, head_yaw, tuple(warnings))


def _common_row(sequence: ExtractionSequence, segment: MotionSegment) -> dict[str, Any]:
    return {
        "subset": sequence.subset_name,
        "subject_id": sequence.subject_id,
        "trial_id": sequence.trial_id,
        "sequence_name": sequence.stem,
        "segment_id": segment.segment_id,
        "label": segment.label,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "start_time_sec": segment.start_time_sec,
        "end_time_sec": segment.end_time_sec,
        "duration_sec": segment.duration_sec,
        "source_rule": segment.source_rule,
        "confidence": segment.confidence,
        "quality_status": segment.quality_status,
        "quality_valid_frame_ratio": segment.quality_valid_frame_ratio,
        "quality_representation_score_mean": segment.quality_representation_score_mean,
        "quality_hesitation_overlap_ratio": segment.quality_hesitation_overlap_ratio,
        "quality_reasons": "; ".join(segment.quality_reasons),
    }


def extract_walk_segment_features(sequence: ExtractionSequence, segment: MotionSegment, config: ExtractionConfig) -> SegmentFeatureResult:
    sl = _segment_slice(segment)
    joints = np.asarray(sequence.arrays["joints_can"][sl], dtype=np.float32)
    valid_mask = _valid_segment_mask(sequence, segment)
    root_pos = np.asarray(sequence.arrays["root_pos_m"][sl], dtype=np.float32)
    trunk_flexion = _masked_series(sequence.arrays["trunk_forward_flexion_deg"][sl], valid_mask)
    pelvis_yaw = _masked_series(unwrap_degrees(sequence.arrays["pelvis_yaw_deg"][sl]), valid_mask)
    neck_y = np.asarray(joints[:, NECK, 1], dtype=np.float32)

    heel_events = _ordered_heel_strikes(sequence, segment)
    step_times = _contralateral_intervals_sec(heel_events, sequence.fps)
    left_hs = _ipsilateral_event_frames(sequence, segment, "left_heel_strike")
    right_hs = _ipsilateral_event_frames(sequence, segment, "right_heel_strike")
    left_to = _ipsilateral_event_frames(sequence, segment, "left_toe_off")
    right_to = _ipsilateral_event_frames(sequence, segment, "right_toe_off")
    stride_times = np.concatenate([
        _same_side_intervals_sec(left_hs, sequence.fps),
        _same_side_intervals_sec(right_hs, sequence.fps),
    ]).astype(np.float32, copy=False)
    stance_times = np.concatenate([
        _phase_durations_sec(left_hs, left_to, sequence.fps),
        _phase_durations_sec(right_hs, right_to, sequence.fps),
    ]).astype(np.float32, copy=False)
    swing_times = np.concatenate([
        _phase_durations_sec(left_to, left_hs, sequence.fps),
        _phase_durations_sec(right_to, right_hs, sequence.fps),
    ]).astype(np.float32, copy=False)
    double_support = _double_support_durations_sec(np.asarray(sequence.arrays["gait_phase_global"][sl]), sequence.fps)

    left_step_lengths, right_step_lengths = _step_lengths_walk_m(sequence, segment)
    left_stride_lengths, right_stride_lengths = _stride_lengths_walk_m(sequence, segment)

    left_hip_rom = _robust_range(_masked_series(_hip_angle_series_deg(joints, "left"), valid_mask), config)
    right_hip_rom = _robust_range(_masked_series(_hip_angle_series_deg(joints, "right"), valid_mask), config)
    left_knee_rom = _robust_range(_masked_series(_knee_flexion_series_deg(joints, "left"), valid_mask), config)
    right_knee_rom = _robust_range(_masked_series(_knee_flexion_series_deg(joints, "right"), valid_mask), config)
    left_ankle_rom = _robust_range(_masked_series(_ankle_relative_series_deg(joints, "left"), valid_mask), config)
    right_ankle_rom = _robust_range(_masked_series(_ankle_relative_series_deg(joints, "right"), valid_mask), config)
    left_arm_swing = _robust_range(_masked_series(_arm_swing_series_deg(joints, "left"), valid_mask), config)
    right_arm_swing = _robust_range(_masked_series(_arm_swing_series_deg(joints, "right"), valid_mask), config)

    duration_sec = float(segment.duration_sec)
    path_length = _path_length_xy(root_pos[:, :2], valid_mask)
    double_support_mask = np.asarray(sequence.arrays["gait_phase_global"][sl], dtype=np.int32) == 3
    valid_double_support = double_support_mask & valid_mask

    if neck_y.size >= 2:
        trend = np.linspace(float(neck_y[0]), float(neck_y[-1]), neck_y.size, dtype=np.float32)
        trunk_ml_sway = float(np.std(neck_y - trend))
    else:
        trunk_ml_sway = float("nan")

    features = {
        "gait_speed_mps": float(path_length / duration_sec) if math.isfinite(path_length) and duration_sec > 0.0 else float("nan"),
        "cadence_steps_per_min": float(len(heel_events) / duration_sec * 60.0) if duration_sec > 0.0 else float("nan"),
        "step_time_mean_sec": _mean_or_nan(step_times),
        "stride_time_mean_sec": _mean_or_nan(stride_times),
        "stance_time_mean_sec": _mean_or_nan(stance_times),
        "swing_time_mean_sec": _mean_or_nan(swing_times),
        "double_support_time_mean_sec": _mean_or_nan(double_support),
        "double_support_percent": float(np.mean(valid_double_support.astype(np.float32)) * 100.0) if valid_mask.size else float("nan"),
        "step_time_variability_cv_percent": _coefficient_of_variation_percent(step_times),
        "stride_time_variability_cv_percent": _coefficient_of_variation_percent(stride_times),
        "left_step_length_mean_m": _mean_or_nan(left_step_lengths),
        "right_step_length_mean_m": _mean_or_nan(right_step_lengths),
        "step_length_mean_m": _mean_or_nan(np.concatenate([left_step_lengths, right_step_lengths])),
        "step_length_asymmetry_percent": _normalized_asymmetry(_mean_or_nan(left_step_lengths), _mean_or_nan(right_step_lengths), config),
        "left_stride_length_mean_m": _mean_or_nan(left_stride_lengths),
        "right_stride_length_mean_m": _mean_or_nan(right_stride_lengths),
        "stride_length_mean_m": _mean_or_nan(np.concatenate([left_stride_lengths, right_stride_lengths])),
        "stride_length_asymmetry_percent": _normalized_asymmetry(_mean_or_nan(left_stride_lengths), _mean_or_nan(right_stride_lengths), config),
        "left_hip_rom_deg": left_hip_rom,
        "right_hip_rom_deg": right_hip_rom,
        "hip_rom_mean_deg": _mean_or_nan(np.asarray([left_hip_rom, right_hip_rom], dtype=np.float32)),
        "hip_rom_asymmetry_percent": _normalized_asymmetry(left_hip_rom, right_hip_rom, config),
        "left_knee_rom_deg": left_knee_rom,
        "right_knee_rom_deg": right_knee_rom,
        "knee_rom_mean_deg": _mean_or_nan(np.asarray([left_knee_rom, right_knee_rom], dtype=np.float32)),
        "knee_rom_asymmetry_percent": _normalized_asymmetry(left_knee_rom, right_knee_rom, config),
        "left_ankle_rom_deg": left_ankle_rom,
        "right_ankle_rom_deg": right_ankle_rom,
        "ankle_rom_mean_deg": _mean_or_nan(np.asarray([left_ankle_rom, right_ankle_rom], dtype=np.float32)),
        "ankle_rom_asymmetry_percent": _normalized_asymmetry(left_ankle_rom, right_ankle_rom, config),
        "left_arm_swing_amplitude_deg": left_arm_swing,
        "right_arm_swing_amplitude_deg": right_arm_swing,
        "arm_swing_amplitude_mean_deg": _mean_or_nan(np.asarray([left_arm_swing, right_arm_swing], dtype=np.float32)),
        "arm_swing_asymmetry_percent": _normalized_asymmetry(left_arm_swing, right_arm_swing, config),
        "trunk_flexion_mean_deg": _mean_or_nan(trunk_flexion),
        "trunk_flexion_rom_deg": _robust_range(trunk_flexion, config),
        "pelvis_rotation_rom_deg": _robust_range(pelvis_yaw, config),
        "trunk_ml_sway_m": trunk_ml_sway,
    }

    missing_features: dict[str, str] = {}
    if not math.isfinite(features["step_time_mean_sec"]):
        missing_features["step_time_mean_sec"] = "insufficient_alternating_heel_strikes"
        missing_features["step_time_variability_cv_percent"] = "insufficient_alternating_heel_strikes"
    if not math.isfinite(features["stride_time_mean_sec"]):
        missing_features["stride_time_mean_sec"] = "insufficient_ipsilateral_heel_strikes"
        missing_features["stride_time_variability_cv_percent"] = "insufficient_ipsilateral_heel_strikes"
    if not math.isfinite(features["stance_time_mean_sec"]):
        missing_features["stance_time_mean_sec"] = "insufficient_heel_strike_to_toe_off_pairs"
    if not math.isfinite(features["swing_time_mean_sec"]):
        missing_features["swing_time_mean_sec"] = "insufficient_toe_off_to_heel_strike_pairs"
    if not math.isfinite(features["double_support_time_mean_sec"]):
        missing_features["double_support_time_mean_sec"] = "no_double_support_frames"
    if not math.isfinite(features["left_step_length_mean_m"]):
        missing_features["left_step_length_mean_m"] = "no_left_heel_strikes"
    if not math.isfinite(features["right_step_length_mean_m"]):
        missing_features["right_step_length_mean_m"] = "no_right_heel_strikes"
    if not math.isfinite(features["left_stride_length_mean_m"]):
        missing_features["left_stride_length_mean_m"] = "insufficient_left_heel_strikes"
    if not math.isfinite(features["right_stride_length_mean_m"]):
        missing_features["right_stride_length_mean_m"] = "insufficient_right_heel_strikes"

    row = _common_row(sequence, segment)
    row.update(features)
    return SegmentFeatureResult(row=row, features=features, missing_features=missing_features)


def extract_turn_segment_features(
    sequence: ExtractionSequence,
    segment: MotionSegment,
    config: ExtractionConfig,
    pose_signals: PoseOrientationSignals,
) -> SegmentFeatureResult:
    sl = _segment_slice(segment)
    valid_mask = _valid_segment_mask(sequence, segment)
    time_sec = np.asarray(sequence.arrays["time_s"][sl], dtype=np.float32)
    root_pos = np.asarray(sequence.arrays["root_pos_m"][sl], dtype=np.float32)
    heading_unwrapped = np.asarray(sequence.arrays["heading_unwrapped_deg"][sl], dtype=np.float32)
    yaw_rate = np.abs(np.asarray(sequence.arrays["yaw_rate_deg_s"][sl], dtype=np.float32))
    pelvis_yaw = _masked_series(unwrap_degrees(sequence.arrays["pelvis_yaw_deg"][sl]), valid_mask)
    trunk_yaw = _masked_series(unwrap_degrees(sequence.arrays["trunk_yaw_deg"][sl]), valid_mask)
    trunk_lateral_lean = np.abs(_masked_series(sequence.arrays["trunk_lateral_lean_deg"][sl], valid_mask))
    root_speed = np.asarray(sequence.arrays["root_speed_xy_mps"], dtype=np.float32)

    heel_events = _ordered_heel_strikes(sequence, segment)
    step_times = _contralateral_intervals_sec(heel_events, sequence.fps)
    step_lengths_turn = _step_lengths_turn_m(sequence, segment)

    turn_angle = float(abs(heading_unwrapped[-1] - heading_unwrapped[0])) if heading_unwrapped.size else float("nan")
    duration_sec = float(segment.duration_sec)
    path_length = _path_length_xy(root_pos[:, :2], valid_mask)

    lookback_frames = seconds_to_frames(config.pre_turn_hesitation_max_lookback_sec, sequence.fps, minimum=1)
    pre_start = max(0, segment.start_frame - lookback_frames)
    pre_speed = np.asarray(root_speed[pre_start : segment.start_frame], dtype=np.float32)
    if pre_speed.size:
        low_speed = pre_speed < float(config.pre_turn_hesitation_speed_mps)
        pre_turn_hesitation = 0.0
        run_length = 0
        for value in low_speed[::-1]:
            if not bool(value):
                break
            run_length += 1
        pre_turn_hesitation = run_length / float(sequence.fps)
    else:
        pre_turn_hesitation = float("nan")

    turn_radius = float(path_length / math.radians(turn_angle)) if math.isfinite(path_length) and math.isfinite(turn_angle) and turn_angle >= config.turn_radius_min_angle_deg else float("nan")
    turn_compactness = float(turn_angle / path_length) if math.isfinite(path_length) and path_length > 1e-8 and math.isfinite(turn_angle) else float("nan")

    head_pelvis_delay = float("nan")
    trunk_pelvis_delay = float("nan")
    head_trunk_delay = float("nan")
    en_bloc_index = float("nan")
    if pose_signals.available:
        pelvis_yaw_pose = pose_signals.pelvis_yaw_unwrapped_deg[sl] if pose_signals.pelvis_yaw_unwrapped_deg is not None else None
        trunk_yaw_pose = pose_signals.trunk_yaw_unwrapped_deg[sl] if pose_signals.trunk_yaw_unwrapped_deg is not None else None
        head_yaw_pose = pose_signals.head_yaw_unwrapped_deg[sl] if pose_signals.head_yaw_unwrapped_deg is not None else None
        pelvis_onset = _first_onset_time_sec(pelvis_yaw_pose, time_sec, config)
        trunk_onset = _first_onset_time_sec(trunk_yaw_pose, time_sec, config)
        head_onset = _first_onset_time_sec(head_yaw_pose, time_sec, config)
        if math.isfinite(head_onset) and math.isfinite(pelvis_onset):
            head_pelvis_delay = float(head_onset - pelvis_onset)
        if math.isfinite(trunk_onset) and math.isfinite(pelvis_onset):
            trunk_pelvis_delay = float(trunk_onset - pelvis_onset)
        if math.isfinite(head_onset) and math.isfinite(trunk_onset):
            head_trunk_delay = float(head_onset - trunk_onset)
        if pelvis_yaw_pose is not None and trunk_yaw_pose is not None and head_yaw_pose is not None and math.isfinite(turn_angle) and turn_angle >= config.reorientation_min_total_angle_deg:
            separation = (
                np.abs(head_yaw_pose - trunk_yaw_pose)
                + np.abs(trunk_yaw_pose - pelvis_yaw_pose)
                + np.abs(head_yaw_pose - pelvis_yaw_pose)
            ) / 3.0
            en_bloc_index = float(np.clip(1.0 - np.mean(separation) / max(turn_angle, 1e-6), 0.0, 1.0))

    features = {
        "turn_angle_deg": turn_angle,
        "turn_duration_sec": duration_sec,
        "turn_step_count": float(len(heel_events)),
        "mean_step_time_during_turn_sec": _mean_or_nan(step_times),
        "pre_turn_hesitation_time_sec": float(pre_turn_hesitation),
        "mean_turn_angular_velocity_deg_s": _mean_or_nan(yaw_rate[valid_mask]),
        "peak_turn_angular_velocity_deg_s": float(np.max(yaw_rate[valid_mask])) if np.any(valid_mask) else float("nan"),
        "turn_angular_velocity_variability_cv_percent": _coefficient_of_variation_percent(yaw_rate[valid_mask]),
        "turn_path_radius_m": turn_radius,
        "turn_path_compactness_deg_per_m": turn_compactness,
        "mean_step_length_during_turn_m": _mean_or_nan(step_lengths_turn),
        "trunk_yaw_rom_during_turn_deg": _robust_range(trunk_yaw, config),
        "pelvis_yaw_rom_during_turn_deg": _robust_range(pelvis_yaw, config),
        "head_trunk_pelvis_reorientation_delay_sec": head_pelvis_delay,
        "trunk_pelvis_reorientation_delay_sec": trunk_pelvis_delay,
        "head_trunk_reorientation_delay_sec": head_trunk_delay,
        "en_bloc_index": en_bloc_index,
        "trunk_lateral_lean_during_turn_deg": _mean_or_nan(trunk_lateral_lean),
        "pelvis_ml_excursion_during_turn_m": _robust_range(_masked_series(root_pos[:, 1], valid_mask), config),
    }

    missing_features: dict[str, str] = {}
    if not math.isfinite(features["mean_step_time_during_turn_sec"]):
        missing_features["mean_step_time_during_turn_sec"] = "insufficient_alternating_heel_strikes"
    if not math.isfinite(features["turn_path_radius_m"]):
        missing_features["turn_path_radius_m"] = "turn_angle_below_radius_threshold_or_invalid_path"
    if not math.isfinite(features["mean_step_length_during_turn_m"]):
        missing_features["mean_step_length_during_turn_m"] = "no_valid_heel_strikes"
    if not pose_signals.available:
        missing_features["head_trunk_pelvis_reorientation_delay_sec"] = "missing_pose_orientation_signals"
        missing_features["trunk_pelvis_reorientation_delay_sec"] = "missing_pose_orientation_signals"
        missing_features["head_trunk_reorientation_delay_sec"] = "missing_pose_orientation_signals"
        missing_features["en_bloc_index"] = "missing_pose_orientation_signals"
    elif not math.isfinite(features["head_trunk_pelvis_reorientation_delay_sec"]):
        missing_features["head_trunk_pelvis_reorientation_delay_sec"] = "insufficient_turn_angle_for_reorientation_onset"
    if not math.isfinite(features["trunk_pelvis_reorientation_delay_sec"]):
        missing_features.setdefault("trunk_pelvis_reorientation_delay_sec", "insufficient_turn_angle_for_reorientation_onset")
    if not math.isfinite(features["head_trunk_reorientation_delay_sec"]):
        missing_features.setdefault("head_trunk_reorientation_delay_sec", "insufficient_turn_angle_for_reorientation_onset")
    if not math.isfinite(features["en_bloc_index"]):
        missing_features.setdefault("en_bloc_index", "insufficient_turn_angle_for_en_bloc_index")

    row = _common_row(sequence, segment)
    row.update(features)
    return SegmentFeatureResult(row=row, features=features, missing_features=missing_features)


def walk_feature_schema() -> dict[str, str]:
    return dict(WALK_FEATURE_DEFINITIONS)


def turn_feature_schema() -> dict[str, str]:
    return dict(TURN_FEATURE_DEFINITIONS)