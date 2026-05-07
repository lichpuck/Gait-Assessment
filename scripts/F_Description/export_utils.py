"""Export helpers for F_Description outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .io_utils import ensure_subset_output_dir

if TYPE_CHECKING:
    from .pipeline import DescriptionResult


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
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def sequence_output_path(output_dir: str | Path, subset_name: str, stem: str) -> Path:
    subset_dir = ensure_subset_output_dir(output_dir, subset_name)
    return subset_dir / f"{stem}.json"


def sequence_json_payload(result: "DescriptionResult") -> dict[str, object]:
    sequence = result.sequence
    payload = {
        "module": "F_Description",
        "version": "1.0.0",
        "language": result.config.language,
        "sequence": {
            "subset": sequence.subset_name,
            "subject_id": sequence.subject_id,
            "trial_id": sequence.trial_id,
            "stem": sequence.stem,
            "fps": sequence.fps,
            "num_frames": sequence.num_frames,
            "duration_sec": sequence.duration_sec,
            "source_inputs": result.source_inputs,
        },
        "quality_summary": {
            "walk_segment_count": len(sequence.walk_segments),
            "turn_segment_count": len(sequence.turn_segments),
            "warning_count": len(result.warnings),
            "warnings": list(result.warnings),
        },
        "description": {
            "text_summary_zh": result.profile.get("summary_zh", ""),
            "walk": result.profile.get("walk", {}),
            "turn": result.profile.get("turn", {}),
        },
        "outputs": {"json": str(result.output_paths.get("json", ""))},
    }
    return payload


def write_sequence_output(result: "DescriptionResult", output_dir: str | Path) -> Path:
    json_path = sequence_output_path(output_dir, result.sequence.subset_name, result.sequence.stem)
    result.output_paths["json"] = json_path
    payload = sequence_json_payload(result)
    payload["outputs"]["json"] = str(json_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    return json_path
