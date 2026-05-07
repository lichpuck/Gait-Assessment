"""Export helpers for E_Extraction outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .features import COMMON_ROW_COLUMNS, TURN_FEATURE_COLUMNS, WALK_FEATURE_COLUMNS, turn_feature_schema, walk_feature_schema
from .io_utils import ensure_subset_output_dir

if TYPE_CHECKING:
    from .pipeline import SequenceExtractionResult


def to_jsonable(value: Any) -> Any:
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
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_feature_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_subset_feature_csvs(
    subset_name: str,
    output_dir: str | Path,
    walk_rows: list[dict[str, object]],
    turn_rows: list[dict[str, object]],
    walk_csv_name: str,
    turn_csv_name: str,
) -> dict[str, Path]:
    subset_dir = ensure_subset_output_dir(output_dir, subset_name)
    walk_csv_path = subset_dir / walk_csv_name
    turn_csv_path = subset_dir / turn_csv_name
    _write_feature_csv(walk_csv_path, COMMON_ROW_COLUMNS + WALK_FEATURE_COLUMNS, walk_rows)
    _write_feature_csv(turn_csv_path, COMMON_ROW_COLUMNS + TURN_FEATURE_COLUMNS, turn_rows)
    return {
        "walk_features_csv": walk_csv_path,
        "turn_features_csv": turn_csv_path,
    }


def sequence_json_payload(result: "SequenceExtractionResult") -> dict[str, object]:
    sequence = result.sequence
    return {
        "module": "E_Extraction",
        "version": "1.0.0",
        "sequence": {
            "subset": sequence.subset_name,
            "subject_id": sequence.subject_id,
            "trial_id": sequence.trial_id,
            "stem": sequence.stem,
            "fps": sequence.fps,
            "num_frames": sequence.num_frames,
            "duration_sec": float(sequence.num_frames / sequence.fps),
            "source_c_npz": str(sequence.c_npz_path),
            "source_c_json": str(sequence.c_json_path),
            "source_d_json": str(sequence.d_json_path),
            "source_b_npz": str(sequence.source_b_npz_path) if sequence.source_b_npz_path is not None else None,
        },
        "schema": {
            "input_roots": {
                "c_representation": "outputs/C_Representation",
                "d_segmentation": "outputs/D_Segmentation",
            },
            "output_root": "outputs/E_Extraction",
            "subset_csv_files": {
                "walk": result.config.walk_features_csv_name,
                "turn": result.config.turn_features_csv_name,
            },
            "strict_used_for_extraction_policy": True,
            "json_granularity": "segment_summary_only",
            "walk_feature_definitions": walk_feature_schema(),
            "turn_feature_definitions": turn_feature_schema(),
        },
        "quality_summary": {
            "extraction_success": result.extraction_success,
            "walk_segment_count": len(result.walk_rows),
            "turn_segment_count": len(result.turn_rows),
            "warnings": list(result.warnings),
        },
        "segments": result.segment_payloads,
        "outputs": {name: str(path) for name, path in result.output_paths.items()},
    }


def write_sequence_json(result: "SequenceExtractionResult", output_dir: str | Path) -> Path:
    subset_dir = ensure_subset_output_dir(output_dir, result.sequence.subset_name)
    json_path = subset_dir / f"{result.sequence.stem}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(sequence_json_payload(result)), handle, indent=2, allow_nan=False)
    return json_path