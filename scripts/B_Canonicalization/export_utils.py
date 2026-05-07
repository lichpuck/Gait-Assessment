"""Output helpers for B canonicalization artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from .config import CANONICAL_AXES
from .io_utils import safe_output_name, sequence_output_stem

if TYPE_CHECKING:
    from .pipeline import SequenceCanonicalizationResult


def ensure_subset_output_dir(output_dir: str | Path, subset_name: str) -> Path:
    path = Path(output_dir) / safe_output_name(subset_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def metadata_payload(result: "SequenceCanonicalizationResult") -> dict[str, object]:
    seq = result.sequence
    transform = result.transform
    return {
        "subset": seq.subset_name,
        "subject_id": seq.subject_id,
        "trial_id": seq.trial_id,
        "metadata": {
            "fps": seq.fps,
            "num_frames": seq.num_frames,
            "duration_sec": seq.duration_sec,
            "source_npz": str(seq.npz_path),
            "source_json": str(seq.json_path),
            "audition_metadata": seq.metadata,
        },
        "canonical_axes": CANONICAL_AXES,
        "transform_definition": "point_B = s_global * (R_global @ point_A) + t_global maps A_Audition semantic coordinates to B_Canonicalization coordinates.",
        "R_global": transform.R_global,
        "s_global": transform.s_global,
        "t_global": transform.t_global,
        "rotation_composition": "R_global = R_yaw @ R_floor @ R_body",
        "input_shapes": {
            "joints_3d": list(seq.joints_3d.shape),
            "trans_canonical": list(seq.trans_canonical.shape),
            "pose_raw": list(seq.pose_raw.shape),
            "trans_raw": list(seq.trans_raw.shape),
            "fps": [],
        },
        "output_shapes": {
            "pose_raw": list(seq.pose_raw.shape),
            "trans_raw": list(seq.trans_raw.shape),
            "joints_can": list(result.joints_can.shape),
            "trans_can": list(result.trans_can.shape),
        },
        "body_frame_alignment": {
            "R_body": transform.R_body,
            "forward_axis": transform.body.forward_axis,
            "left_axis": transform.body.left_axis,
            "up_axis": transform.body.up_axis,
            "frames_used": transform.body.frames_used,
            "aggregation": transform.body.aggregation,
            "determinant": transform.body.determinant,
            "metrics": transform.body.metrics,
        },
        "scale_alignment": {
            "enabled": transform.scale.enabled,
            "method": transform.scale.method,
            "skipped_reason": transform.scale.skipped_reason,
            "s_global": transform.scale.s_global,
            "s_global_raw": transform.scale.s_global_raw,
            "clip_applied": transform.scale.clip_applied,
            "clip_range": list(transform.scale.clip_range),
            "aggregation": transform.scale.aggregation,
            "bones_used": list(transform.scale.bones_used),
            "bone_count_valid": transform.scale.bone_count_valid,
            "quality_flag": transform.scale.quality_flag,
            "target_bone_lengths_m": transform.scale.target_bone_lengths_m,
            "observed_bone_lengths_m": transform.scale.observed_bone_lengths_m,
            "per_bone_ratios": transform.scale.per_bone_ratios,
            "metrics": transform.scale.metrics,
        },
        "floor_alignment": {
            "R_floor": transform.R_floor,
            "enabled": transform.floor.enabled,
            "skipped_reason": transform.floor.skipped_reason,
            "support_joint_order": list(transform.floor.support_joint_names),
            "floor_normal_before": transform.floor.floor_normal_before,
            "floor_normal_after": transform.floor.floor_normal_after,
            "plane_coefficients": transform.floor.plane_coefficients,
            "tilt_before_deg": transform.floor.tilt_before_deg,
            "tilt_after_deg": transform.floor.tilt_after_deg,
            "residual_median_abs_m": transform.floor.residual_median_abs_m,
            "robust_support_cloud_count": transform.floor.robust_support_cloud_count,
            "metrics": transform.floor.metrics,
        },
        "ground_alignment": {
            "method": transform.ground.metrics.get("method"),
            "support_joint_order": list(transform.ground.support_joint_names),
            "ground_z_percentile": transform.ground.percentile,
            "ground_z": transform.ground.ground_z,
            "sample_count": transform.ground.sample_count,
            "metrics": transform.ground.metrics,
        },
        "yaw_alignment": {
            "R_yaw": transform.R_yaw,
            "enabled": transform.yaw.enabled,
            "skipped_reason": transform.yaw.skipped_reason,
            "yaw_angle_rad": transform.yaw.yaw_angle_rad,
            "yaw_angle_deg": transform.yaw.yaw_angle_deg,
            "forward_axis_body": transform.yaw.forward_axis_body,
            "pca_variance_ratio": transform.yaw.pca_variance_ratio,
            "root_x_robust_range_m": transform.yaw.robust_range_m,
            "projection_p5_m": transform.yaw.projection_p5_m,
            "projection_p95_m": transform.yaw.projection_p95_m,
            "net_displacement_m": transform.yaw.net_displacement_m,
            "path_length_m": transform.yaw.path_length_m,
            "net_path_ratio": transform.yaw.net_path_ratio,
            "metrics": transform.yaw.metrics,
        },
        "support_points": {
            "support_joint_order": list(result.support_joint_names),
            "per_frame_lowest_support_shape": list(result.support_points_can.shape),
            "ground_joint_points_shape": list(result.ground_joint_points_can.shape),
        },
        "checks": result.checks,
        "warnings": list(result.warnings),
        "outputs": {key: str(path) for key, path in result.output_paths.items()},
    }


def write_sequence_outputs(result: "SequenceCanonicalizationResult", output_dir: str | Path) -> dict[str, Path]:
    subset_dir = ensure_subset_output_dir(output_dir, result.sequence.subset_name)
    stem = sequence_output_stem(result.sequence.subset_name, result.sequence.subject_id, result.sequence.trial_id)
    npz_path = subset_dir / f"{stem}.npz"
    json_path = subset_dir / f"{stem}.json"
    png_path = subset_dir / f"{stem}.png"

    np.savez_compressed(
        npz_path,
        pose_raw=np.asarray(result.sequence.pose_raw, dtype=np.float32),
        trans_raw=np.asarray(result.sequence.trans_raw, dtype=np.float32),
        joints_can=np.asarray(result.joints_can, dtype=np.float32),
        trans_can=np.asarray(result.trans_can, dtype=np.float32),
    )
    result.output_paths.update({"npz": npz_path, "json": json_path, "png": png_path})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metadata_payload(result)), handle, indent=2, allow_nan=False)
    return dict(result.output_paths)
