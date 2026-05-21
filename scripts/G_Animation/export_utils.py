"""Export helpers for CARE-PD G_Animation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import LABEL_COLORS

if TYPE_CHECKING:
    from .pipeline import AnimationRenderResult


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


def _metric_summary(values: np.ndarray) -> dict[str, float | None]:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
    }


def sequence_manifest_payload(result: "AnimationRenderResult") -> dict[str, object]:
    inputs = result.sequence
    metrics = result.metrics
    return {
        "module": "G_Animation",
        "version": result.config.module_version,
        "sequence": {
            "subset": inputs.subset_name,
            "subject_id": inputs.subject_id,
            "trial_id": inputs.trial_id,
            "stem": inputs.stem,
            "fps": inputs.fps,
            "num_frames": inputs.num_frames,
            "duration_sec": inputs.duration_sec,
        },
        "sources": {
            "b_npz": str(inputs.b_sequence.npz_path),
            "b_json": str(inputs.b_sequence.json_path),
            "c_npz": str(inputs.c_sequence.npz_path),
            "c_json": str(inputs.c_sequence.json_path),
            "d_json": str(inputs.d_json_path),
            "f_json": str(inputs.f_json_path),
        },
        "description": {
            "summary_text_zh": inputs.summary_text_zh,
            "summary_present": bool(inputs.summary_text_zh.strip()),
        },
        "segments": {
            "count": len(inputs.segments),
            "label_colors": LABEL_COLORS,
        },
        "metrics": {
            item.key: {
                "title": item.title,
                "unit": item.unit,
                **_metric_summary(item.values),
            }
            for item in metrics.series
        },
        "render": result.render_summary,
        "outputs": {name: str(path) for name, path in result.output_paths.items()},
    }


def write_sequence_manifest(result: "AnimationRenderResult", manifest_path: str | Path) -> Path:
    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sequence_manifest_payload(result)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
    return output_path