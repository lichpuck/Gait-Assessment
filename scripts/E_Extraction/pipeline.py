"""End-to-end pipeline for the E_Extraction module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import ExtractionConfig
from .export_utils import write_sequence_json, write_subset_feature_csvs
from .features import build_pose_orientation_signals, extract_turn_segment_features, extract_walk_segment_features
from .io_utils import TARGET_LABELS, ExtractionSequence, iter_sequence_jsons, load_extraction_sequence, resolve_sequence_paths


@dataclass
class SequenceExtractionResult:
    sequence: ExtractionSequence
    config: ExtractionConfig
    walk_rows: list[dict[str, object]]
    turn_rows: list[dict[str, object]]
    segment_payloads: list[dict[str, object]]
    extraction_success: bool
    warnings: tuple[str, ...]
    output_paths: dict[str, Path] = field(default_factory=dict)


def extract_sequence(sequence: ExtractionSequence, *, config: ExtractionConfig | None = None) -> SequenceExtractionResult:
    config = config or ExtractionConfig()
    pose_signals = build_pose_orientation_signals(sequence, config)
    warnings = list(pose_signals.warnings)
    walk_rows: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    segment_payloads: list[dict[str, object]] = []

    for segment in sequence.segments:
        payload = {
            "segment_id": segment.segment_id,
            "label": segment.label,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "start_time_sec": segment.start_time_sec,
            "end_time_sec": segment.end_time_sec,
            "duration_sec": segment.duration_sec,
            "used_for_extraction": segment.used_for_extraction,
            "source_rule": segment.source_rule,
            "confidence": segment.confidence,
            "quality": {
                "status": segment.quality_status,
                "valid_frame_ratio": segment.quality_valid_frame_ratio,
                "representation_quality_score_mean": segment.quality_representation_score_mean,
                "hesitation_overlap_ratio": segment.quality_hesitation_overlap_ratio,
                "reasons": list(segment.quality_reasons),
            },
            "auxiliary_overlap_labels": list(segment.auxiliary_overlap_labels),
        }
        if segment.label not in TARGET_LABELS:
            payload["extraction_status"] = "skipped_non_target_label"
            segment_payloads.append(payload)
            continue
        if not segment.used_for_extraction:
            payload["extraction_status"] = "skipped_used_for_extraction_false"
            payload["skip_reason"] = "strict_used_for_extraction_policy"
            segment_payloads.append(payload)
            continue

        if segment.label == "walk":
            feature_result = extract_walk_segment_features(sequence, segment, config)
            walk_rows.append(feature_result.row)
        else:
            feature_result = extract_turn_segment_features(sequence, segment, config, pose_signals)
            turn_rows.append(feature_result.row)
        payload["extraction_status"] = "included"
        payload["features"] = feature_result.features
        payload["missing_features"] = feature_result.missing_features
        segment_payloads.append(payload)

    return SequenceExtractionResult(
        sequence=sequence,
        config=config,
        walk_rows=walk_rows,
        turn_rows=turn_rows,
        segment_payloads=segment_payloads,
        extraction_success=True,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _write_sequence_outputs(
    result: SequenceExtractionResult,
    output_root: Path,
    *,
    write_subset_csvs: bool,
) -> SequenceExtractionResult:
    json_path = write_sequence_json(result, output_root)
    result.output_paths["json"] = json_path
    if write_subset_csvs:
        subset_paths = write_subset_feature_csvs(
            result.sequence.subset_name,
            output_root,
            result.walk_rows,
            result.turn_rows,
            result.config.walk_features_csv_name,
            result.config.turn_features_csv_name,
        )
        result.output_paths.update(subset_paths)
    return result


def process_one_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: ExtractionConfig | None = None,
    input_c_dir: str | Path | None = None,
    input_d_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> SequenceExtractionResult:
    config = config or ExtractionConfig()
    input_c_root = Path(input_c_dir) if input_c_dir is not None else Path(config.input_c_dir)
    input_d_root = Path(input_d_dir) if input_d_dir is not None else Path(config.input_d_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    c_npz_path, c_json_path, d_json_path = resolve_sequence_paths(input_c_root, input_d_root, subset, subject_id, trial_id)
    sequence = load_extraction_sequence(c_npz_path, c_json_path, d_json_path)
    result = extract_sequence(sequence, config=config)
    return _write_sequence_outputs(result, output_root, write_subset_csvs=True)


def process_subset(
    subset: str,
    *,
    config: ExtractionConfig | None = None,
    input_c_dir: str | Path | None = None,
    input_d_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> dict[str, object]:
    config = config or ExtractionConfig()
    input_c_root = Path(input_c_dir) if input_c_dir is not None else Path(config.input_c_dir)
    input_d_root = Path(input_d_dir) if input_d_dir is not None else Path(config.input_d_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    processed = 0
    failed = 0
    walk_rows: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    for d_json_path in iter_sequence_jsons(input_d_root, subset_name=subset, trial_contains=trial_contains, max_trials=max_trials):
        subset_name = d_json_path.parent.name
        stem_parts = d_json_path.stem.split("__", 2)
        if len(stem_parts) != 3:
            failed += 1
            print(f"[error] unexpected sequence stem: {d_json_path.stem}")
            continue
        c_npz_path, c_json_path, d_json_resolved = resolve_sequence_paths(input_c_root, input_d_root, subset_name, stem_parts[1], stem_parts[2])
        try:
            sequence = load_extraction_sequence(c_npz_path, c_json_path, d_json_resolved)
            result = extract_sequence(sequence, config=config)
            _write_sequence_outputs(result, output_root, write_subset_csvs=False)
            walk_rows.extend(result.walk_rows)
            turn_rows.extend(result.turn_rows)
            processed += 1
            print(f"[{processed}] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}")
        except Exception as error:
            failed += 1
            print(f"[error] {d_json_path}: {error}")

    subset_paths = write_subset_feature_csvs(
        subset,
        output_root,
        walk_rows,
        turn_rows,
        config.walk_features_csv_name,
        config.turn_features_csv_name,
    )
    return {
        "subset_name": subset,
        "processed_trials": processed,
        "failed_trials": failed,
        **subset_paths,
    }


def process_all_subsets(
    *,
    config: ExtractionConfig | None = None,
    input_c_dir: str | Path | None = None,
    input_d_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
) -> list[dict[str, object]]:
    config = config or ExtractionConfig()
    input_d_root = Path(input_d_dir) if input_d_dir is not None else Path(config.input_d_dir)
    summaries: list[dict[str, object]] = []
    for subset_dir in sorted(path for path in input_d_root.iterdir() if path.is_dir()):
        if not any(subset_dir.glob("*.json")):
            continue
        summaries.append(
            process_subset(
                subset_dir.name,
                config=config,
                input_c_dir=input_c_dir,
                input_d_dir=input_d_dir,
                output_dir=output_dir,
                max_trials=max_trials_per_subset,
            )
        )
    return summaries