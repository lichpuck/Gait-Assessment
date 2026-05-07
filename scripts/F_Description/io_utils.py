"""I/O helpers for F_Description sequence loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .config import DescriptionConfig


TARGET_LABELS = {"walk", "turn"}


@dataclass(frozen=True)
class DescriptionSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    fps: float | None
    num_frames: int | None
    duration_sec: float | None
    source_json_path: Path
    source_inputs: dict[str, str]
    warnings: tuple[str, ...]
    walk_segments: list[dict[str, object]]
    turn_segments: list[dict[str, object]]
    raw_payload: dict[str, object]


def safe_output_name(text: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


def ensure_subset_output_dir(output_dir: str | Path, subset_name: str) -> Path:
    subset_dir = Path(output_dir) / str(subset_name)
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


def _flatten_segment(segment: dict[str, object]) -> dict[str, object]:
    features = dict(segment.get("features", {}))
    quality = dict(segment.get("quality", {}))
    row: dict[str, object] = {
        "segment_id": segment.get("segment_id"),
        "label": segment.get("label"),
        "start_frame": segment.get("start_frame"),
        "end_frame": segment.get("end_frame"),
        "start_time_sec": segment.get("start_time_sec"),
        "end_time_sec": segment.get("end_time_sec"),
        "duration_sec": segment.get("duration_sec"),
        "source_rule": segment.get("source_rule"),
        "confidence": segment.get("confidence"),
        "quality_status": quality.get("status"),
        "quality_valid_frame_ratio": quality.get("valid_frame_ratio"),
        "quality_representation_score_mean": quality.get("representation_quality_score_mean"),
        "quality_hesitation_overlap_ratio": quality.get("hesitation_overlap_ratio"),
        "quality_reasons": ";".join(str(item) for item in quality.get("reasons", []) if item),
        "missing_features": dict(segment.get("missing_features", {})),
    }
    row.update(features)
    return row


def _load_sequence_from_json(path: Path) -> DescriptionSequence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequence_payload = dict(payload.get("sequence", {}))
    quality_summary = dict(payload.get("quality_summary", {}))
    walk_segments: list[dict[str, object]] = []
    turn_segments: list[dict[str, object]] = []
    for segment in payload.get("segments", []):
        if segment.get("label") not in TARGET_LABELS:
            continue
        if segment.get("extraction_status") != "included":
            continue
        flattened = _flatten_segment(dict(segment))
        if segment.get("label") == "walk":
            walk_segments.append(flattened)
        else:
            turn_segments.append(flattened)
    source_inputs = {
        "source_e_json": str(path),
        "source_c_npz": str(sequence_payload.get("source_c_npz") or ""),
        "source_c_json": str(sequence_payload.get("source_c_json") or ""),
        "source_d_json": str(sequence_payload.get("source_d_json") or ""),
        "source_b_npz": str(sequence_payload.get("source_b_npz") or ""),
    }
    return DescriptionSequence(
        subset_name=str(sequence_payload.get("subset") or payload.get("subset") or ""),
        subject_id=str(sequence_payload.get("subject_id") or ""),
        trial_id=str(sequence_payload.get("trial_id") or ""),
        stem=str(sequence_payload.get("stem") or path.stem),
        fps=float(sequence_payload["fps"]) if sequence_payload.get("fps") is not None else None,
        num_frames=int(sequence_payload["num_frames"]) if sequence_payload.get("num_frames") is not None else None,
        duration_sec=float(sequence_payload["duration_sec"]) if sequence_payload.get("duration_sec") is not None else None,
        source_json_path=path,
        source_inputs=source_inputs,
        warnings=tuple(dict.fromkeys(str(item) for item in quality_summary.get("warnings", []) if item)),
        walk_segments=walk_segments,
        turn_segments=turn_segments,
        raw_payload=payload,
    )


def load_sequence_json(path: str | Path) -> DescriptionSequence:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return _load_sequence_from_json(json_path)


def load_subset_reference_tables(
    subset_name: str,
    config: DescriptionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset_dir = Path(config.input_e_dir) / str(subset_name)
    walk_path = subset_dir / config.walk_features_csv_name
    turn_path = subset_dir / config.turn_features_csv_name
    walk_df = pd.read_csv(walk_path) if walk_path.exists() else pd.DataFrame()
    turn_df = pd.read_csv(turn_path) if turn_path.exists() else pd.DataFrame()
    return walk_df, turn_df


def iter_sequence_jsons(
    input_dir: str | Path,
    *,
    subset_name: str | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> list[Path]:
    base = Path(input_dir)
    subset_dirs = [base / subset_name] if subset_name is not None else sorted(path for path in base.iterdir() if path.is_dir())
    results: list[Path] = []
    for subset_dir in subset_dirs:
        if not subset_dir.exists():
            continue
        for path in sorted(subset_dir.glob("*.json")):
            if path.name in {"walk_features.json", "turn_features.json"}:
                continue
            if trial_contains is not None and trial_contains.lower() not in path.stem.lower():
                continue
            results.append(path)
            if max_trials is not None and len(results) >= max_trials:
                return results
    return results


def load_one_sequence(
    subset_name: str,
    subject_id: str,
    trial_id: str,
    *,
    config: DescriptionConfig,
) -> DescriptionSequence:
    path = Path(config.input_e_dir) / str(subset_name) / f"{sequence_output_stem(subset_name, subject_id, trial_id)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing E_Extraction sequence JSON: {path}")
    return _load_sequence_from_json(path)


def load_subset_sequences(
    subset_name: str,
    *,
    config: DescriptionConfig,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> list[DescriptionSequence]:
    return [
        _load_sequence_from_json(path)
        for path in iter_sequence_jsons(
            config.input_e_dir,
            subset_name=subset_name,
            trial_contains=trial_contains,
            max_trials=max_trials,
        )
    ]
