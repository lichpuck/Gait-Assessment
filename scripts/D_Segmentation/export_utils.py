"""Export helpers for D_Segmentation outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from .config import AUXILIARY_LABELS, PRIMARY_LABELS
from .io_utils import REQUIRED_NPZ_KEYS, ensure_subset_output_dir

if TYPE_CHECKING:
    from .pipeline import SequenceSegmentationResult


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


def _segment_quality(result: "SequenceSegmentationResult", start_frame: int, end_frame: int, hesitation_ratio: float) -> dict[str, object]:
    valid = np.asarray(result.sequence.arrays["valid_frame_mask"], dtype=bool)[start_frame : end_frame + 1]
    scores = np.asarray(result.sequence.arrays["representation_quality_score"], dtype=np.float32)[start_frame : end_frame + 1]
    valid_frame_ratio = float(np.mean(valid.astype(np.float32))) if valid.size else 0.0
    quality_score_mean = float(np.mean(scores)) if scores.size else 0.0

    reasons: list[str] = []
    if valid_frame_ratio < result.config.segment_valid_frame_ratio_min:
        reasons.append(
            f"valid_frame_ratio_below_{result.config.segment_valid_frame_ratio_min:.2f}"
        )
    if quality_score_mean < result.config.segment_quality_score_min:
        reasons.append(
            f"representation_quality_score_below_{result.config.segment_quality_score_min:.2f}"
        )
    if hesitation_ratio > 0.0:
        reasons.append("hesitation_overlap")
    if result.warnings:
        reasons.extend(str(warning) for warning in result.warnings)

    if any(reason.startswith(("valid_frame_ratio_below", "representation_quality_score_below")) for reason in reasons):
        status = "bad"
    elif reasons:
        status = "warning"
    else:
        status = "good"

    return {
        "status": status,
        "valid_frame_ratio": valid_frame_ratio,
        "representation_quality_score_mean": quality_score_mean,
        "hesitation_overlap_ratio": float(hesitation_ratio),
        "reasons": reasons,
    }


def segment_payloads(result: "SequenceSegmentationResult") -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for index, segment in enumerate(result.segments, start=1):
        quality = _segment_quality(
            result,
            segment.start_frame,
            segment.end_frame,
            segment.hesitation_overlap_ratio,
        )
        used_for_extraction = bool(
            quality["status"] != "bad"
            and segment.hesitation_overlap_ratio < result.config.segment_hesitation_exclusion_ratio
        )
        payloads.append(
            {
                "segment_id": f"seg_{index:04d}",
                "label": segment.label,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "used_for_extraction": used_for_extraction,
                "quality": quality,
                "start_time_sec": segment.start_time_sec,
                "end_time_sec": segment.end_time_sec,
                "duration_sec": segment.duration_sec,
                "source_rule": segment.source_rule,
                "confidence": segment.confidence,
                "hesitation_overlap_frames": segment.hesitation_overlap_frames,
                "auxiliary_overlap_labels": segment.auxiliary_overlap_labels,
            }
        )
    return payloads


def segment_rows(result: "SequenceSegmentationResult") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for segment in segment_payloads(result):
        quality = segment["quality"]
        rows.append(
            {
                "subset": result.sequence.subset_name,
                "subject_id": result.sequence.subject_id,
                "trial_id": result.sequence.trial_id,
                "segment_id": segment["segment_id"],
                "label": segment["label"],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "used_for_extraction": segment["used_for_extraction"],
                "quality_status": quality["status"],
                "quality_valid_frame_ratio": quality["valid_frame_ratio"],
                "quality_representation_score_mean": quality["representation_quality_score_mean"],
                "quality_hesitation_overlap_ratio": quality["hesitation_overlap_ratio"],
                "quality_reasons": "; ".join(str(reason) for reason in quality["reasons"]),
                "start_time_sec": segment["start_time_sec"],
                "end_time_sec": segment["end_time_sec"],
                "duration_sec": segment["duration_sec"],
                "source_rule": segment["source_rule"],
                "confidence": segment["confidence"],
                "hesitation_overlap_frames": segment["hesitation_overlap_frames"],
                "auxiliary_overlap_labels": segment["auxiliary_overlap_labels"],
            }
        )
    return rows


def motion_segments_payload(result: "SequenceSegmentationResult") -> list[dict[str, object]]:
    return segment_payloads(result)


def gait_events_payload(result: "SequenceSegmentationResult") -> dict[str, object]:
    gait = result.gait_events
    contact_order = sorted(
        [
            {"frame": int(frame), "side": "left", "event": "heel_strike"}
            for frame in gait.left_heel_strikes
        ]
        + [
            {"frame": int(frame), "side": "right", "event": "heel_strike"}
            for frame in gait.right_heel_strikes
        ]
        + [
            {"frame": int(frame), "side": "left", "event": "toe_off"}
            for frame in gait.left_toe_offs
        ]
        + [
            {"frame": int(frame), "side": "right", "event": "toe_off"}
            for frame in gait.right_toe_offs
        ],
        key=lambda item: (int(item["frame"]), str(item["side"]), str(item["event"])),
    )
    return {
        "left_heel_strikes": np.asarray(gait.left_heel_strikes, dtype=int).tolist(),
        "right_heel_strikes": np.asarray(gait.right_heel_strikes, dtype=int).tolist(),
        "left_toe_offs": np.asarray(gait.left_toe_offs, dtype=int).tolist(),
        "right_toe_offs": np.asarray(gait.right_toe_offs, dtype=int).tolist(),
        "contact_mask_column_order": ["left", "right"],
        "left_contact_frame_count": int(np.count_nonzero(gait.left_contact_mask)),
        "right_contact_frame_count": int(np.count_nonzero(gait.right_contact_mask)),
        "contact_stability_score": float(gait.contact_stability_score),
        "contact_order": contact_order,
        "warnings": list(gait.warnings),
    }


def metadata_payload(result: "SequenceSegmentationResult") -> dict[str, object]:
    source_outputs = result.sequence.metadata.get("outputs", {})
    if not isinstance(source_outputs, dict):
        source_outputs = {}
    source_files = result.sequence.metadata.get("source_files", {})
    if not isinstance(source_files, dict):
        source_files = {}
    return {
        "module": "D_Segmentation",
        "version": "2.0.0",
        "sequence": {
            "subset": result.sequence.subset_name,
            "subject_id": result.sequence.subject_id,
            "trial_id": result.sequence.trial_id,
            "stem": result.sequence.stem,
            "fps": result.sequence.fps,
            "num_frames": result.sequence.num_frames,
            "duration_sec": float(result.sequence.num_frames / result.sequence.fps),
            "source_c_npz": source_outputs.get("npz", str(result.sequence.npz_path)),
            "source_c_json": source_outputs.get("json", str(result.sequence.json_path)),
            "upstream_b_npz": source_files.get("upstream_b_npz", ""),
            "upstream_b_json": source_files.get("upstream_b_json", ""),
        },
        "schema": {
            "input_root": "outputs/C_Representation",
            "output_root": "outputs/D_Segmentation",
            "input_contract": "strict_C_Representation_v2",
            "primary_labels": list(PRIMARY_LABELS),
            "auxiliary_labels": list(AUXILIARY_LABELS),
            "required_npz_fields": list(REQUIRED_NPZ_KEYS),
            "segment_fields": ["segment_id", "label", "start_frame", "end_frame", "used_for_extraction", "quality"],
            "quality_thresholds": {
                "valid_frame_ratio_min": result.config.segment_valid_frame_ratio_min,
                "representation_quality_score_min": result.config.segment_quality_score_min,
                "hesitation_exclusion_ratio": result.config.segment_hesitation_exclusion_ratio,
            },
        },
        "quality_summary": {
            "segmentation_success": result.segmentation_success,
            "quality_flags": result.quality_flags,
            "quality_metrics": result.quality_metrics,
            "gait_events": gait_events_payload(result),
            "warnings": list(result.warnings),
        },
        "segments": motion_segments_payload(result),
        "outputs": {name: str(path) for name, path in result.output_paths.items()},
    }


def _write_segment_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "subset",
        "subject_id",
        "trial_id",
        "segment_id",
        "label",
        "start_frame",
        "end_frame",
        "used_for_extraction",
        "quality_status",
        "quality_valid_frame_ratio",
        "quality_representation_score_mean",
        "quality_hesitation_overlap_ratio",
        "quality_reasons",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "source_rule",
        "confidence",
        "hesitation_overlap_frames",
        "hesitation_overlap_ratio",
        "auxiliary_overlap_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_sequence_outputs(result: "SequenceSegmentationResult", output_dir: str | Path) -> dict[str, Path]:
    subset_dir = ensure_subset_output_dir(output_dir, result.sequence.subset_name)
    stem = result.sequence.stem
    json_path = subset_dir / f"{stem}.json"
    csv_path = subset_dir / f"{stem}.csv"

    rows = segment_rows(result)
    _write_segment_csv(csv_path, rows)
    output_paths = {**result.output_paths, "json": json_path, "segments_csv": csv_path}
    result.output_paths.update(output_paths)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metadata_payload(result)), handle, indent=2, allow_nan=False)
    return output_paths


def summary_row(result: "SequenceSegmentationResult") -> dict[str, object]:
    row = {
        "subset": result.sequence.subset_name,
        "subject_id": result.sequence.subject_id,
        "trial_id": result.sequence.trial_id,
        "num_frames": result.sequence.num_frames,
        "fps": result.sequence.fps,
        "segmentation_success": result.segmentation_success,
        "upstream_representation_success": result.quality_flags.get("upstream_representation_success", False),
        "segment_count": len(result.segments),
        "hesitation_frame_count": int(np.count_nonzero(result.hesitation_mask)),
        "contact_stability_score": result.quality_metrics.get("contact_stability_score", 0.0),
    }
    for label in PRIMARY_LABELS:
        row[f"{label}_frame_count"] = result.quality_metrics.get(f"{label}_frame_count", 0)
    return row


def failure_summary_row(subset_name: str, stem: str, error: Exception) -> dict[str, object]:
    parts = stem.split("__", 2)
    subject_id = parts[1] if len(parts) > 1 else ""
    trial_id = parts[2] if len(parts) > 2 else stem
    return {
        "subset": subset_name,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "segmentation_success": False,
        "error": str(error),
    }


def write_subset_summary(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    subset_name: str,
    summary_name: str,
) -> Path:
    subset_dir = ensure_subset_output_dir(output_dir, subset_name)
    summary_path = subset_dir / summary_name
    fieldnames = list(dict.fromkeys(field for row in rows for field in row.keys()))
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return summary_path
