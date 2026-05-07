"""Feature construction for the C_Representation analysis layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RepresentationConfig, seconds_to_frames
from .joints_schema import (
    LEFT_FOOT,
    LEFT_HIP,
    LEFT_SHOULDER,
    NECK,
    PELVIS,
    RIGHT_FOOT,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    support_points_by_side,
)
from .numeric_utils import (
    axis_angle_to_matrix,
    finite_difference,
    fill_short_false_runs,
    heading_from_horizontal_vectors,
    horizontal,
    matrix_to_euler_xyz_deg,
    mask_offsets,
    mask_onsets,
    nearest_valid_fill,
    normalize_vectors,
    remove_short_true_runs,
    unwrap_degrees,
)


@dataclass(frozen=True)
class FeatureBuildResult:
    arrays: dict[str, np.ndarray]
    quality_flags: dict[str, bool]
    quality_metrics: dict[str, object]
    gait_summary: dict[str, object]
    warnings: tuple[str, ...]


def _body_heading(joints: np.ndarray, root_velocity: np.ndarray, config: RepresentationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    pelvis = joints[:, PELVIS]
    neck = joints[:, NECK]
    left_hip = joints[:, LEFT_HIP]
    right_hip = joints[:, RIGHT_HIP]
    left_shoulder = joints[:, LEFT_SHOULDER]
    right_shoulder = joints[:, RIGHT_SHOULDER]

    body_left_raw = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    trunk_up_raw = neck - pelvis
    body_forward_raw = np.cross(body_left_raw, trunk_up_raw)
    body_forward_horizontal = horizontal(body_forward_raw)
    body_forward_axis, body_valid = normalize_vectors(
        body_forward_horizontal,
        eps=config.heading_min_horizontal_norm_m,
    )

    root_velocity_horizontal = horizontal(root_velocity)
    motion_axis, motion_valid = normalize_vectors(
        root_velocity_horizontal,
        eps=config.heading_velocity_fallback_min_mps,
    )

    heading_axis = np.array(body_forward_axis, dtype=np.float32, copy=True)
    fallback_mask = ~body_valid & motion_valid
    heading_axis[fallback_mask] = motion_axis[fallback_mask]
    source_valid = body_valid | fallback_mask
    warnings: list[str] = []

    heading_source_valid = source_valid.astype(bool, copy=True)
    if np.any(heading_source_valid):
        heading_axis = nearest_valid_fill(
            heading_axis,
            heading_source_valid,
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
    else:
        heading_axis[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        warnings.append("Body heading and motion fallback were degenerate for the whole sequence; +X was used.")

    heading_axis, _ = normalize_vectors(heading_axis, eps=config.heading_min_horizontal_norm_m)
    heading_deg = heading_from_horizontal_vectors(heading_axis)
    return heading_axis, heading_deg, heading_source_valid.astype(bool, copy=False), tuple(warnings)


def _contact_probability(
    support_points: np.ndarray,
    fps: float,
    floor_z: float,
    config: RepresentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_velocity = finite_difference(support_points, fps)
    support_speed_xy = np.linalg.norm(support_velocity[:, :2], axis=1)
    clearance = np.maximum(support_points[:, 2] - float(floor_z), 0.0)
    height_score = 1.0 - np.clip(clearance / max(config.contact_height_threshold_m, 1e-6), 0.0, 1.0)
    speed_score = 1.0 - np.clip(support_speed_xy / max(config.contact_speed_threshold_mps, 1e-6), 0.0, 1.0)
    prob = (
        config.contact_height_weight * height_score
        + config.contact_speed_weight * speed_score
    ).astype(np.float32)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)

    mask = prob >= float(config.contact_binary_threshold)
    mask = remove_short_true_runs(mask, seconds_to_frames(config.min_contact_duration_sec, fps, minimum=1))
    mask = fill_short_false_runs(mask, seconds_to_frames(config.min_swing_duration_sec, fps, minimum=1))
    return (
        prob.astype(np.float32, copy=False),
        mask.astype(bool, copy=False),
        mask_onsets(mask),
        mask_offsets(mask),
    )


def _contact_stability(left_contact: np.ndarray, right_contact: np.ndarray, left_hs: np.ndarray, right_hs: np.ndarray) -> float:
    left = np.asarray(left_contact, dtype=bool)
    right = np.asarray(right_contact, dtype=bool)
    left_events = np.flatnonzero(left_hs)
    right_events = np.flatnonzero(right_hs)

    def duty_score(mask: np.ndarray) -> float:
        duty = float(np.mean(mask.astype(np.float32))) if mask.size else 0.0
        return max(0.0, 1.0 - min(abs(duty - 0.60) / 0.60, 1.0))

    merged = sorted([(int(frame), "left") for frame in left_events] + [(int(frame), "right") for frame in right_events])
    if len(merged) < 2:
        alternation_score = 0.0
    else:
        alternating = sum(1 for (_, prev), (_, cur) in zip(merged[:-1], merged[1:]) if prev != cur)
        alternation_score = alternating / max(len(merged) - 1, 1)
    event_balance = 1.0 - abs(len(left_events) - len(right_events)) / max(len(left_events) + len(right_events), 1)
    return float(np.mean([duty_score(left), duty_score(right), alternation_score, event_balance]))


def _gait_phase(left_contact: np.ndarray, right_contact: np.ndarray) -> np.ndarray:
    left = np.asarray(left_contact, dtype=bool)
    right = np.asarray(right_contact, dtype=bool)
    phase = np.zeros(left.shape, dtype=np.uint8)
    phase[left & ~right] = 1
    phase[right & ~left] = 2
    phase[left & right] = 3
    return phase


def _canonical_root_orientation(
    pose_raw: np.ndarray,
    r_global: np.ndarray,
    r_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_raw = axis_angle_to_matrix(np.asarray(pose_raw, dtype=np.float32)[:, :3])
    canonical_transform = np.asarray(r_global, dtype=np.float32) @ np.asarray(r_total, dtype=np.float32)
    root_can = canonical_transform[None, :, :] @ root_raw
    roll, pitch, yaw = matrix_to_euler_xyz_deg(root_can)
    return pitch.astype(np.float32, copy=False), roll.astype(np.float32, copy=False), yaw.astype(np.float32, copy=False)


def _trunk_orientation(
    joints: np.ndarray,
    heading_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pelvis = joints[:, PELVIS]
    neck = joints[:, NECK]
    left_shoulder = joints[:, LEFT_SHOULDER]
    right_shoulder = joints[:, RIGHT_SHOULDER]

    trunk_axis, _ = normalize_vectors(neck - pelvis)
    vertical_component = np.clip(trunk_axis[:, 2], -1.0, 1.0)
    forward_component = np.sum(trunk_axis * heading_axis, axis=1)
    heading_left_axis = np.stack(
        [-heading_axis[:, 1], heading_axis[:, 0], np.zeros(heading_axis.shape[0], dtype=np.float32)],
        axis=1,
    )
    heading_left_axis, _ = normalize_vectors(heading_left_axis)
    lateral_component = np.sum(trunk_axis * heading_left_axis, axis=1)

    left_axis_raw = left_shoulder - right_shoulder
    left_axis_raw = left_axis_raw - np.sum(left_axis_raw * trunk_axis, axis=1, keepdims=True) * trunk_axis
    left_axis, left_valid = normalize_vectors(left_axis_raw)
    if np.any(left_valid):
        left_axis = nearest_valid_fill(left_axis, left_valid, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    else:
        left_axis[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    left_axis, _ = normalize_vectors(left_axis)

    forward_axis = np.cross(left_axis, trunk_axis)
    forward_axis, forward_valid = normalize_vectors(forward_axis)
    if np.any(forward_valid):
        forward_axis = nearest_valid_fill(forward_axis, forward_valid, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    else:
        forward_axis[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    forward_axis, _ = normalize_vectors(forward_axis)
    left_axis = np.cross(trunk_axis, forward_axis)
    left_axis, _ = normalize_vectors(left_axis)

    body_frame = np.stack([forward_axis, left_axis, trunk_axis], axis=2)
    trunk_roll, trunk_pitch, trunk_yaw = matrix_to_euler_xyz_deg(body_frame)

    trunk_forward_flexion = np.rad2deg(np.arctan2(forward_component, vertical_component)).astype(np.float32)
    trunk_lateral_lean = np.rad2deg(np.arctan2(lateral_component, vertical_component)).astype(np.float32)
    trunk_lean = np.rad2deg(np.arccos(vertical_component)).astype(np.float32)
    return (
        trunk_forward_flexion,
        trunk_lateral_lean,
        trunk_lean,
        trunk_pitch.astype(np.float32, copy=False),
        trunk_roll.astype(np.float32, copy=False),
        trunk_yaw.astype(np.float32, copy=False),
    )


def build_representation_features(
    joints_can: np.ndarray,
    trans_can: np.ndarray,
    pose_raw: np.ndarray,
    r_global: np.ndarray,
    r_total: np.ndarray,
    fps: float,
    config: RepresentationConfig,
) -> FeatureBuildResult:
    joints = np.asarray(joints_can, dtype=np.float32)
    trans = np.asarray(trans_can, dtype=np.float32)
    pose = np.asarray(pose_raw, dtype=np.float32)
    frame_count = int(joints.shape[0])
    warnings: list[str] = []

    pelvis = joints[:, PELVIS]
    root_pos = pelvis.astype(np.float32, copy=False)
    left_foot = joints[:, LEFT_FOOT]
    right_foot = joints[:, RIGHT_FOOT]
    left_support, right_support = support_points_by_side(joints)

    root_velocity = finite_difference(root_pos, fps)
    root_acceleration = finite_difference(root_velocity, fps)
    root_speed_xy = np.linalg.norm(root_velocity[:, :2], axis=1).astype(np.float32)
    left_foot_velocity = finite_difference(left_foot, fps)
    right_foot_velocity = finite_difference(right_foot, fps)
    left_foot_speed = np.linalg.norm(left_foot_velocity, axis=1).astype(np.float32)
    right_foot_speed = np.linalg.norm(right_foot_velocity, axis=1).astype(np.float32)
    pelvis_height = pelvis[:, 2].astype(np.float32, copy=False)
    pelvis_vertical_velocity = finite_difference(pelvis_height[:, None], fps)[:, 0].astype(np.float32)

    heading_axis, heading_deg, heading_source_valid, heading_warnings = _body_heading(joints, root_velocity, config)
    warnings.extend(heading_warnings)
    heading_unwrapped = unwrap_degrees(heading_deg)
    yaw_rate = finite_difference(heading_unwrapped[:, None], fps)[:, 0].astype(np.float32)
    yaw_acceleration = finite_difference(yaw_rate[:, None], fps)[:, 0].astype(np.float32)

    (
        trunk_forward_flexion,
        trunk_lateral_lean,
        trunk_lean,
        trunk_pitch,
        trunk_roll,
        trunk_yaw,
    ) = _trunk_orientation(joints, heading_axis)
    pelvis_pitch, pelvis_roll, pelvis_yaw = _canonical_root_orientation(pose, r_global, r_total)

    support_z = np.concatenate([left_support[:, 2], right_support[:, 2]])
    floor_z = float(np.percentile(support_z[np.isfinite(support_z)], config.contact_floor_percentile))
    left_prob, left_contact, left_hs, left_to = _contact_probability(left_support, fps, floor_z, config)
    right_prob, right_contact, right_hs, right_to = _contact_probability(right_support, fps, floor_z, config)
    gait_phase_global = _gait_phase(left_contact, right_contact)
    left_gait_phase = left_contact.astype(np.uint8, copy=False)
    right_gait_phase = right_contact.astype(np.uint8, copy=False)
    contact_confidence = np.stack(
        [
            np.clip(2.0 * np.abs(left_prob - float(config.contact_binary_threshold)), 0.0, 1.0),
            np.clip(2.0 * np.abs(right_prob - float(config.contact_binary_threshold)), 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    left_contact_ratio = float(np.mean(left_contact.astype(np.float32)))
    right_contact_ratio = float(np.mean(right_contact.astype(np.float32)))
    left_hs_count = int(np.count_nonzero(left_hs))
    right_hs_count = int(np.count_nonzero(right_hs))
    step_count = int(left_hs_count + right_hs_count)
    duration_sec = frame_count / float(fps)
    cadence = float(step_count / max(duration_sec, 1e-6) * 60.0)
    contact_stability = _contact_stability(left_contact, right_contact, left_hs, right_hs)

    if step_count < 2:
        warnings.append("Too few contact onsets were detected for stable gait rhythm estimates.")
    if np.std(left_prob) < config.contact_degenerate_std_min and np.std(right_prob) < config.contact_degenerate_std_min:
        warnings.append("Foot contact probabilities are nearly constant across the sequence.")
    if contact_stability < 0.45:
        warnings.append(f"Low contact stability score ({contact_stability:.3f}).")

    joint_nan_mask = ~np.all(np.isfinite(joints), axis=(1, 2))
    finite_frame_mask = (
        ~joint_nan_mask
        & np.all(np.isfinite(trans), axis=1)
        & np.all(np.isfinite(root_pos), axis=1)
        & np.all(np.isfinite(root_velocity), axis=1)
        & np.all(np.isfinite(root_acceleration), axis=1)
        & np.isfinite(heading_deg)
        & np.isfinite(heading_unwrapped)
        & np.isfinite(yaw_rate)
        & np.isfinite(yaw_acceleration)
        & np.isfinite(left_prob)
        & np.isfinite(right_prob)
        & np.isfinite(contact_confidence).all(axis=1)
        & np.isfinite(trunk_forward_flexion)
        & np.isfinite(trunk_lateral_lean)
        & np.isfinite(trunk_lean)
        & np.isfinite(pelvis_pitch)
        & np.isfinite(pelvis_roll)
        & np.isfinite(pelvis_yaw)
        & np.isfinite(trunk_pitch)
        & np.isfinite(trunk_roll)
        & np.isfinite(trunk_yaw)
    )
    velocity_outlier_mask = (
        (root_speed_xy > config.max_root_speed_mps)
        | (left_foot_speed > config.max_foot_speed_mps)
        | (right_foot_speed > config.max_foot_speed_mps)
    )
    speed_quality_mask = ~velocity_outlier_mask
    contact_quality_mask = np.isfinite(left_prob) & np.isfinite(right_prob) & np.isfinite(contact_confidence).all(axis=1)
    valid_frame_mask = finite_frame_mask & speed_quality_mask & heading_source_valid & contact_quality_mask

    root_speed_score = 1.0 - np.clip(root_speed_xy / max(config.max_root_speed_mps, 1e-6), 0.0, 1.0)
    left_speed_score = 1.0 - np.clip(left_foot_speed / max(config.max_foot_speed_mps, 1e-6), 0.0, 1.0)
    right_speed_score = 1.0 - np.clip(right_foot_speed / max(config.max_foot_speed_mps, 1e-6), 0.0, 1.0)
    speed_score = np.minimum(np.minimum(root_speed_score, left_speed_score), right_speed_score)
    representation_quality_score = np.mean(
        np.stack(
            [
                finite_frame_mask.astype(np.float32),
                speed_score.astype(np.float32),
                heading_source_valid.astype(np.float32),
                np.mean(contact_confidence, axis=1).astype(np.float32),
            ],
            axis=1,
        ),
        axis=1,
    )
    representation_quality_score = np.clip(representation_quality_score, 0.0, 1.0).astype(np.float32)

    arrays = {
        "fps": np.asarray(float(fps), dtype=np.float32),
        "time_s": (np.arange(frame_count, dtype=np.float32) / float(fps)).astype(np.float32),
        "frame_index": np.arange(frame_count, dtype=np.int32),
        "valid_frame_mask": valid_frame_mask.astype(bool, copy=False),
        "joints_can": joints.astype(np.float32, copy=False),
        "trans_can": trans.astype(np.float32, copy=False),
        "root_pos_m": root_pos.astype(np.float32, copy=False),
        "root_velocity_mps": root_velocity.astype(np.float32, copy=False),
        "root_speed_xy_mps": root_speed_xy.astype(np.float32, copy=False),
        "root_acceleration_mps2": root_acceleration.astype(np.float32, copy=False),
        "pelvis_height_m": pelvis_height,
        "pelvis_vertical_velocity_mps": pelvis_vertical_velocity,
        "heading_deg": heading_deg.astype(np.float32, copy=False),
        "heading_unwrapped_deg": heading_unwrapped.astype(np.float32, copy=False),
        "yaw_rate_deg_s": yaw_rate.astype(np.float32, copy=False),
        "yaw_acceleration_deg_s2": yaw_acceleration.astype(np.float32, copy=False),
        "left_foot_pos_m": left_foot.astype(np.float32, copy=False),
        "right_foot_pos_m": right_foot.astype(np.float32, copy=False),
        "left_foot_velocity_mps": left_foot_velocity.astype(np.float32, copy=False),
        "right_foot_velocity_mps": right_foot_velocity.astype(np.float32, copy=False),
        "left_foot_speed_mps": left_foot_speed.astype(np.float32, copy=False),
        "right_foot_speed_mps": right_foot_speed.astype(np.float32, copy=False),
        "left_foot_height_m": left_foot[:, 2].astype(np.float32, copy=False),
        "right_foot_height_m": right_foot[:, 2].astype(np.float32, copy=False),
        "left_foot_contact_prob": left_prob,
        "right_foot_contact_prob": right_prob,
        "left_foot_contact": left_contact.astype(bool, copy=False),
        "right_foot_contact": right_contact.astype(bool, copy=False),
        "contact_confidence": contact_confidence,
        "left_heel_strike": left_hs.astype(bool, copy=False),
        "right_heel_strike": right_hs.astype(bool, copy=False),
        "left_toe_off": left_to.astype(bool, copy=False),
        "right_toe_off": right_to.astype(bool, copy=False),
        "left_gait_phase": left_gait_phase,
        "right_gait_phase": right_gait_phase,
        "gait_phase_global": gait_phase_global,
        "trunk_forward_flexion_deg": trunk_forward_flexion.astype(np.float32, copy=False),
        "trunk_lateral_lean_deg": trunk_lateral_lean.astype(np.float32, copy=False),
        "trunk_lean_angle_deg": trunk_lean.astype(np.float32, copy=False),
        "pelvis_pitch_deg": pelvis_pitch.astype(np.float32, copy=False),
        "pelvis_roll_deg": pelvis_roll.astype(np.float32, copy=False),
        "pelvis_yaw_deg": pelvis_yaw.astype(np.float32, copy=False),
        "trunk_pitch_deg": trunk_pitch.astype(np.float32, copy=False),
        "trunk_roll_deg": trunk_roll.astype(np.float32, copy=False),
        "trunk_yaw_deg": trunk_yaw.astype(np.float32, copy=False),
        "joint_nan_mask": joint_nan_mask.astype(bool, copy=False),
        "velocity_outlier_mask": velocity_outlier_mask.astype(bool, copy=False),
        "representation_quality_score": representation_quality_score,
    }

    quality_flags = {
        "finite_feature_arrays": bool(all(np.all(np.isfinite(value)) for value in arrays.values() if np.issubdtype(value.dtype, np.floating))),
        "heading_available": bool(np.any(heading_source_valid)),
        "trunk_axis_available": bool(np.all(np.isfinite(trunk_lean))),
        "pelvis_orientation_available": bool(np.all(np.isfinite(pelvis_pitch)) and np.all(np.isfinite(pelvis_roll)) and np.all(np.isfinite(pelvis_yaw))),
        "contact_probabilities_finite": bool(np.all(contact_quality_mask)),
        "valid_frame_mask_nonempty": bool(np.any(valid_frame_mask)),
    }
    quality_metrics = {
        "frame_count": frame_count,
        "valid_frame_ratio": float(np.mean(valid_frame_mask.astype(np.float32))),
        "num_invalid_frames": int(np.count_nonzero(~valid_frame_mask)),
        "finite_frame_ratio": float(np.mean(finite_frame_mask.astype(np.float32))),
        "speed_quality_ratio": float(np.mean(speed_quality_mask.astype(np.float32))),
        "velocity_outlier_count": int(np.count_nonzero(velocity_outlier_mask)),
        "joint_nan_frame_count": int(np.count_nonzero(joint_nan_mask)),
        "heading_source_valid_ratio": float(np.mean(heading_source_valid.astype(np.float32))),
        "contact_quality_ratio": float(np.mean(contact_quality_mask.astype(np.float32))),
        "contact_confidence_mean": float(np.mean(contact_confidence)),
        "representation_quality_score_mean": float(np.mean(representation_quality_score)),
        "root_speed_xy_mean_mps": float(np.mean(root_speed_xy)),
        "root_speed_xy_max_mps": float(np.max(root_speed_xy)),
        "left_foot_speed_mean_mps": float(np.mean(left_foot_speed)),
        "right_foot_speed_mean_mps": float(np.mean(right_foot_speed)),
        "floor_z_estimate_m": floor_z,
        "left_contact_ratio": left_contact_ratio,
        "right_contact_ratio": right_contact_ratio,
        "trunk_forward_flexion_mean_deg": float(np.mean(trunk_forward_flexion)),
        "trunk_lateral_lean_mean_deg": float(np.mean(trunk_lateral_lean)),
        "trunk_lean_mean_deg": float(np.mean(trunk_lean)),
    }
    gait_summary = {
        "left_gait_phase_mapping": {
            "0": "swing",
            "1": "stance",
        },
        "right_gait_phase_mapping": {
            "0": "swing",
            "1": "stance",
        },
        "gait_phase_global_mapping": {
            "0": "no_contact",
            "1": "left_stance",
            "2": "right_stance",
            "3": "double_support",
        },
        "left_heel_strike_count": left_hs_count,
        "right_heel_strike_count": right_hs_count,
        "left_toe_off_count": int(np.count_nonzero(left_to)),
        "right_toe_off_count": int(np.count_nonzero(right_to)),
        "step_count": step_count,
        "cadence_steps_per_min": cadence,
        "left_contact_ratio": left_contact_ratio,
        "right_contact_ratio": right_contact_ratio,
        "double_support_ratio": float(np.mean(gait_phase_global == 3)),
        "no_contact_ratio": float(np.mean(gait_phase_global == 0)),
        "contact_stability_score": contact_stability,
    }
    return FeatureBuildResult(
        arrays=arrays,
        quality_flags=quality_flags,
        quality_metrics=quality_metrics,
        gait_summary=gait_summary,
        warnings=tuple(dict.fromkeys(warnings)),
    )
