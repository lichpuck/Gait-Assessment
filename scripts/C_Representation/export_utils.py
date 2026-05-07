"""Output helpers for C_Representation artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from .config import CANONICAL_AXES, MODULE_VERSION, REQUIRED_B_NPZ_KEYS, STRICT_NPZ_FIELDS
from .io_utils import ensure_subset_output_dir, sequence_output_stem
from .joints_schema import joint_schema_payload

if TYPE_CHECKING:
    from .pipeline import SequenceRepresentationResult


NPZ_FIELD_SPECS: dict[str, dict[str, str]] = {
    "fps": {"unit": "Hz", "description": "Original sequence frame rate."},
    "time_s": {"unit": "second", "description": "Frame timestamp derived from frame_index / fps."},
    "frame_index": {"unit": "frame", "description": "Original zero-based frame index."},
    "valid_frame_mask": {"unit": "bool", "description": "Overall valid-frame mask for downstream C/D/E use."},
    "joints_can": {"unit": "meter", "description": "Canonical SMPL 24-joint coordinates in +X forward, +Y left, +Z up axes."},
    "trans_can": {"unit": "meter", "description": "Canonical SMPL global translation from B_Canonicalization."},
    "root_pos_m": {"unit": "meter", "description": "Canonical pelvis/root joint position used for trajectory derivatives."},
    "root_velocity_mps": {"unit": "meter/second", "description": "Central-difference root_pos_m velocity."},
    "root_speed_xy_mps": {"unit": "meter/second", "description": "Horizontal root speed magnitude."},
    "root_acceleration_mps2": {"unit": "meter/second^2", "description": "Central-difference derivative of root_velocity_mps."},
    "pelvis_height_m": {"unit": "meter", "description": "Canonical pelvis Z coordinate."},
    "pelvis_vertical_velocity_mps": {"unit": "meter/second", "description": "Central-difference derivative of pelvis_height_m."},
    "heading_deg": {"unit": "degree", "description": "Wrapped body-facing yaw angle in the canonical XY plane."},
    "heading_unwrapped_deg": {"unit": "degree", "description": "Continuous unwrapped heading angle."},
    "yaw_rate_deg_s": {"unit": "degree/second", "description": "Central-difference derivative of heading_unwrapped_deg."},
    "yaw_acceleration_deg_s2": {"unit": "degree/second^2", "description": "Central-difference derivative of yaw_rate_deg_s."},
    "left_foot_pos_m": {"unit": "meter", "description": "Canonical left SMPL foot joint position."},
    "right_foot_pos_m": {"unit": "meter", "description": "Canonical right SMPL foot joint position."},
    "left_foot_velocity_mps": {"unit": "meter/second", "description": "Central-difference left foot velocity."},
    "right_foot_velocity_mps": {"unit": "meter/second", "description": "Central-difference right foot velocity."},
    "left_foot_speed_mps": {"unit": "meter/second", "description": "Left foot speed magnitude."},
    "right_foot_speed_mps": {"unit": "meter/second", "description": "Right foot speed magnitude."},
    "left_foot_height_m": {"unit": "meter", "description": "Canonical left foot Z coordinate."},
    "right_foot_height_m": {"unit": "meter", "description": "Canonical right foot Z coordinate."},
    "left_foot_contact_prob": {"unit": "0-1", "description": "Geometry-derived left foot contact probability."},
    "right_foot_contact_prob": {"unit": "0-1", "description": "Geometry-derived right foot contact probability."},
    "left_foot_contact": {"unit": "bool", "description": "Binary left foot contact mask."},
    "right_foot_contact": {"unit": "bool", "description": "Binary right foot contact mask."},
    "contact_confidence": {"unit": "0-1", "description": "Binary contact decision confidence, columns [left, right]."},
    "left_heel_strike": {"unit": "bool", "description": "Left contact onset event mask."},
    "right_heel_strike": {"unit": "bool", "description": "Right contact onset event mask."},
    "left_toe_off": {"unit": "bool", "description": "Left contact offset event mask."},
    "right_toe_off": {"unit": "bool", "description": "Right contact offset event mask."},
    "left_gait_phase": {"unit": "phase", "description": "Left gait phase: 0=swing, 1=stance."},
    "right_gait_phase": {"unit": "phase", "description": "Right gait phase: 0=swing, 1=stance."},
    "gait_phase_global": {"unit": "phase", "description": "Global support phase: 0=no_contact, 1=left_stance, 2=right_stance, 3=double_support."},
    "trunk_forward_flexion_deg": {"unit": "degree", "description": "Signed sagittal trunk flexion from canonical joints; forward is positive."},
    "trunk_lateral_lean_deg": {"unit": "degree", "description": "Signed trunk lateral lean from canonical joints; left is positive."},
    "trunk_lean_angle_deg": {"unit": "degree", "description": "Unsigned trunk tilt angle away from vertical."},
    "pelvis_pitch_deg": {"unit": "degree", "description": "Canonical pelvis/root pitch from R_global @ R_total @ pose_raw root rotation."},
    "pelvis_roll_deg": {"unit": "degree", "description": "Canonical pelvis/root roll from R_global @ R_total @ pose_raw root rotation."},
    "pelvis_yaw_deg": {"unit": "degree", "description": "Canonical pelvis/root yaw from R_global @ R_total @ pose_raw root rotation."},
    "trunk_pitch_deg": {"unit": "degree", "description": "Trunk pitch from canonical pelvis-neck/shoulder geometry."},
    "trunk_roll_deg": {"unit": "degree", "description": "Trunk roll from canonical pelvis-neck/shoulder geometry."},
    "trunk_yaw_deg": {"unit": "degree", "description": "Trunk yaw from canonical pelvis-neck/shoulder geometry."},
    "joint_nan_mask": {"unit": "bool", "description": "Frames where joints_can contains at least one NaN/non-finite value."},
    "velocity_outlier_mask": {"unit": "bool", "description": "Frames exceeding root or foot speed sanity thresholds."},
    "representation_quality_score": {"unit": "score", "description": "Per-frame 0-1 representation quality score."},
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _field_payload(result: "SequenceRepresentationResult") -> dict[str, object]:
    fields: dict[str, object] = {}
    for key in STRICT_NPZ_FIELDS:
        value = result.arrays[key]
        spec = NPZ_FIELD_SPECS[key]
        fields[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "unit": spec["unit"],
            "description": spec["description"],
        }
    return fields


def metadata_payload(result: "SequenceRepresentationResult") -> dict[str, object]:
    sequence = result.sequence
    valid_frame_ratio = float(result.quality_metrics.get("valid_frame_ratio", 0.0))
    num_invalid = int(result.quality_metrics.get("num_invalid_frames", 0))
    return {
        "module": "C_Representation",
        "version": MODULE_VERSION,
        "basic_info": {
            "subset": sequence.subset_name,
            "subject_id": sequence.subject_id,
            "trial_id": sequence.trial_id,
            "fps": sequence.fps,
            "num_frames": sequence.num_frames,
            "duration_sec": sequence.duration_sec,
        },
        "metadata": {
            "subset": sequence.subset_name,
            "subject_id": sequence.subject_id,
            "trial_id": sequence.trial_id,
            "fps": sequence.fps,
            "num_frames": sequence.num_frames,
            "duration_sec": sequence.duration_sec,
        },
        "source_files": {
            "upstream_b_npz": str(sequence.npz_path),
            "upstream_b_json": str(sequence.json_path),
        },
        "input_contract": {
            "source_root": "outputs/B_Canonicalization",
            "required_npz_fields": list(REQUIRED_B_NPZ_KEYS),
            "legacy_b_fields_supported": False,
            "canonical_pose_generated": False,
        },
        "coordinate_system": {
            "x": "forward",
            "y": "left",
            "z": "up",
            "canonical_axes": CANONICAL_AXES,
            "unit": "meter",
            "angle_unit": "degree",
            "time_unit": "second",
            "velocity_unit": "meter/second",
            "angular_velocity_unit": "degree/second",
        },
        "smpl_joint_schema": joint_schema_payload(),
        "fields": _field_payload(result),
        "npz_fields": _field_payload(result),
        "derivation_config": {
            "smoothing_method": "none",
            "velocity_calculation": "central_difference",
            "acceleration_calculation": "central_difference_of_velocity",
            "heading_logic": "body forward axis from hips/shoulders with root-velocity fallback",
            "pelvis_orientation_logic": "R_global @ R_total @ pose_raw root axis-angle, then ZYX Euler extraction",
            "trunk_orientation_logic": "canonical pelvis-neck and shoulder geometry",
            "foot_contact_method": "height/speed weighted probability with duration cleanup",
            "gait_event_method": "contact onset/offset proxies",
            "gait_phase_encoding": {
                "left_right": {"0": "swing", "1": "stance"},
                "global": {"0": "no_contact", "1": "left_stance", "2": "right_stance", "3": "double_support"},
            },
            "parameters": result.config.to_parameters(),
        },
        "calculation_parameters": result.config.to_parameters(),
        "quality_info": {
            "valid_frame_ratio": valid_frame_ratio,
            "num_invalid_frames": num_invalid,
            "contact_quality": {
                "contact_quality_ratio": result.quality_metrics.get("contact_quality_ratio"),
                "contact_confidence_mean": result.quality_metrics.get("contact_confidence_mean"),
                "contact_stability_score": result.gait_summary.get("contact_stability_score"),
                "left_contact_ratio": result.gait_summary.get("left_contact_ratio"),
                "right_contact_ratio": result.gait_summary.get("right_contact_ratio"),
            },
            "warnings": list(result.warnings),
        },
        "quality_diagnostics": {
            "representation_success": result.representation_success,
            "quality_flags": result.quality_flags,
            "quality_metrics": result.quality_metrics,
            "warnings": list(result.warnings),
        },
        "gait_summary": result.gait_summary,
        "downstream_hints": [
            "Use valid_frame_mask before estimating clinical parameters.",
            "Use left/right_foot_contact for gait-event proxies; probabilities and confidence are retained for threshold tuning.",
            "heading_deg is body-facing yaw, not root-trajectory tangent.",
            "This is the strict C_Representation v2 contract and does not include legacy D_Segmentation aliases.",
        ],
        "upstream_b_metadata": sequence.metadata,
        "outputs": {name: str(path) for name, path in result.output_paths.items()},
    }


def write_sequence_outputs(result: "SequenceRepresentationResult", output_dir: str | Path) -> dict[str, Path]:
    subset_dir = ensure_subset_output_dir(output_dir, result.sequence.subset_name)
    stem = sequence_output_stem(result.sequence.subset_name, result.sequence.subject_id, result.sequence.trial_id)
    npz_path = subset_dir / f"{stem}.npz"
    json_path = subset_dir / f"{stem}.json"

    missing = [key for key in STRICT_NPZ_FIELDS if key not in result.arrays]
    extra = sorted(set(result.arrays) - set(STRICT_NPZ_FIELDS))
    if missing:
        raise ValueError(f"C_Representation strict output is missing NPZ fields: {missing}")
    if extra:
        raise ValueError(f"C_Representation strict output has unexpected NPZ fields: {extra}")

    arrays_to_save = {key: np.asarray(result.arrays[key]) for key in STRICT_NPZ_FIELDS}
    np.savez_compressed(npz_path, **arrays_to_save)
    result.output_paths.update({"npz": npz_path, "json": json_path})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metadata_payload(result)), handle, indent=2, allow_nan=False)
    return dict(result.output_paths)
