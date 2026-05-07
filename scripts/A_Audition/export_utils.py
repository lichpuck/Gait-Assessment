"""Export helpers for A_Audition artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import CANONICAL_AXES

if TYPE_CHECKING:
    from .pipeline import AuditionFailure, AuditionSequenceResult, FailedSequenceResult, SkippedSequenceResult


NA_VALUE = "NA"


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


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
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _clean_metadata(raw: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in raw.items():
        if value is None:
            cleaned[str(key)] = NA_VALUE
            continue
        if isinstance(value, str):
            stripped = value.strip()
            cleaned[str(key)] = stripped if stripped else NA_VALUE
            continue
        cleaned[str(key)] = to_jsonable(value)
    return cleaned


def metadata_payload(result: "AuditionSequenceResult") -> dict[str, object]:
    record = result.record
    return {
        "subset": record.subset_name,
        "subject_id": record.subject_id,
        "trial_id": record.trial_id,
        "source_path": str(record.source_path),
        "duration_sec": record.duration_sec,
        "input_shapes": {
            "pose": list(record.pose.shape),
            "trans": list(record.trans.shape),
            "beta": list(record.beta.shape),
            "fps": [],
        },
        "output_shapes": {
            "joints_3d": list(result.joints_3d.shape),
            "trans_canonical": list(result.trans_canonical.shape),
            "pose_raw": list(record.pose.shape),
            "trans_raw": list(record.trans.shape),
            "fps": [],
            "beta": list(record.beta.shape),
            "R_total": list(result.axis.R_total.shape),
            "support_points": list(result.support.support_points.shape),
        },
        "raw_axes": {
            "array_order": ["X", "Y", "Z"],
            "semantics": result.axis.raw_axis_semantics,
            "raw_ranges": result.axis.raw_ranges,
            "robust_ranges": result.axis.robust_ranges,
            "robust_range_percentiles": [
                result.config.robust_range_low_percentile,
                result.config.robust_range_high_percentile,
            ],
            "range_order": list(result.axis.range_order),
        },
        "canonical_axes": CANONICAL_AXES,
        "axis_mapping": {
            "forward_raw_axis": result.axis.forward_raw_axis,
            "lateral_raw_axis": result.axis.lateral_raw_axis,
            "vertical_raw_axis": result.axis.vertical_raw_axis,
            "forward_sign": result.axis.forward_sign,
            "lateral_sign": result.axis.lateral_sign,
            "vertical_sign": result.axis.vertical_sign,
            "forward_axis_margin": result.axis.forward_axis_margin,
            "vertical_alignment": result.axis.vertical_alignment,
            "lateral_alignment": result.axis.lateral_alignment,
            "determinant": result.axis.determinant,
        },
        "forward_segment": result.axis.forward_segment,
        "R_total": result.axis.R_total,
        "support_points": {
            "shape": list(result.support.support_points.shape),
            "selection_order": ["left_ankle", "right_ankle", "left_foot", "right_foot"],
            "selection_rule": "lowest_canonical_z_per_frame",
            "coordinate_frame": "canonical_semantic",
            "metrics": result.support.metrics,
        },
        "quality_flags": result.axis.quality_flags,
        "warnings": list(result.warnings),
        "metadata": {
            "fps": record.fps,
            "num_frames": record.num_frames,
            "raw_metadata": _clean_metadata(record.metadata),
        },
        "outputs": {key: str(path) for key, path in result.output_paths.items()},
    }


def write_metadata_json(result: "AuditionSequenceResult", json_path: str | Path) -> Path:
    path = Path(json_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metadata_payload(result)), handle, indent=2, allow_nan=False)
    return path


def write_sequence_outputs(result: "AuditionSequenceResult", output_dir: str | Path) -> dict[str, Path]:
    subset_dir = ensure_subset_output_dir(output_dir, result.record.subset_name)
    stem = sequence_output_stem(result.record.subset_name, result.record.subject_id, result.record.trial_id)
    npz_path = subset_dir / f"{stem}.npz"
    json_path = subset_dir / f"{stem}.json"
    png_path = subset_dir / f"{stem}.png"

    np.savez_compressed(
        npz_path,
        joints_3d=np.asarray(result.joints_3d, dtype=np.float32),
        trans_canonical=np.asarray(result.trans_canonical, dtype=np.float32),
        pose_raw=np.asarray(result.record.pose, dtype=np.float32),
        trans_raw=np.asarray(result.record.trans, dtype=np.float32),
        beta=np.asarray(result.record.beta, dtype=np.float32),
        fps=np.asarray(result.record.fps, dtype=np.float32),
        R_total=np.asarray(result.axis.R_total, dtype=np.float32),
        support_points=np.asarray(result.support.support_points, dtype=np.float32),
    )

    result.output_paths.update({"npz": npz_path, "json": json_path})
    write_metadata_json(result, json_path)
    return {"npz": npz_path, "json": json_path, "png": png_path}


def _base_row(subset: str, subject_id: str, trial_id: str) -> dict[str, object]:
    return {
        "subset": subset,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "num_frames": NA_VALUE,
        "fps": NA_VALUE,
        "duration_sec": NA_VALUE,
        "status": NA_VALUE,
        "skip_reason": NA_VALUE,
        "warning_count": 0,
        "error_message": NA_VALUE,
        "output_npz": NA_VALUE,
        "output_json": NA_VALUE,
        "output_png": NA_VALUE,
    }


def summary_row(result: "AuditionSequenceResult") -> dict[str, object]:
    row = _base_row(result.record.subset_name, result.record.subject_id, result.record.trial_id)
    row.update(
        {
            "num_frames": result.record.num_frames,
            "fps": result.record.fps,
            "duration_sec": result.record.duration_sec,
            "status": "success",
            "warning_count": len(result.warnings),
            "axis_determinant": result.axis.determinant,
            "forward_raw_axis": result.axis.forward_raw_axis,
            "vertical_raw_axis": result.axis.vertical_raw_axis,
            "support_finite_ratio": result.support.metrics.get("finite_ratio", NA_VALUE),
            "output_npz": str(result.output_paths.get("npz", NA_VALUE)),
            "output_json": str(result.output_paths.get("json", NA_VALUE)),
            "output_png": str(result.output_paths.get("png", NA_VALUE)),
        }
    )
    return row


def skipped_summary_row(result: "SkippedSequenceResult") -> dict[str, object]:
    row = _base_row(result.record.subset_name, result.record.subject_id, result.record.trial_id)
    row.update(
        {
            "num_frames": result.record.num_frames,
            "fps": result.record.fps,
            "duration_sec": result.record.duration_sec,
            "status": "skipped",
            "skip_reason": result.skip_reason,
        }
    )
    return row


def failed_summary_row(failure: "AuditionFailure") -> dict[str, object]:
    row = _base_row(failure.subset_name, failure.subject_id, failure.trial_id)
    row.update(
        {
            "status": "failed",
            "error_message": failure.message,
        }
    )
    return row


def failed_sequence_summary_row(result: "FailedSequenceResult") -> dict[str, object]:
    row = _base_row(result.record.subset_name, result.record.subject_id, result.record.trial_id)
    row.update(
        {
            "num_frames": result.record.num_frames,
            "fps": result.record.fps,
            "duration_sec": result.record.duration_sec,
            "status": "failed",
            "skip_reason": result.failure_reason,
            "error_message": result.message,
        }
    )
    return row


def update_summary_csv(row: dict[str, object], summary_csv: str | Path) -> Path:
    path = Path(summary_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                key = (str(existing["subset"]), str(existing["subject_id"]), str(existing["trial_id"]))
                merged[key] = existing
    key = (str(row["subset"]), str(row["subject_id"]), str(row["trial_id"]))
    merged[key] = row
    fieldnames = list(dict.fromkeys(field for item in merged.values() for field in item.keys()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sorted_key in sorted(merged):
            item = merged[sorted_key]
            writer.writerow({field: item.get(field, "") for field in fieldnames})
    return path
