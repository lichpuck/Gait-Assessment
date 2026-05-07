"""End-to-end pipeline for the independent D_Segmentation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PRIMARY_LABELS, SegmentationConfig
from .export_utils import failure_summary_row, summary_row, write_sequence_outputs, write_subset_summary
from .gait_events import GaitEventResult, detect_gait_events
from .io_utils import (
    RepresentationSequence,
    iter_sequence_npzs,
    iter_subset_dirs,
    load_representation_sequence,
    resolve_sequence_paths,
)
from .rule_engine import (
    PrimaryRuleResult,
    SegmentRecord,
    build_segment_records,
    compute_hesitation_mask,
    run_rule_based_primary_segmentation,
)
from .signals import SignalSet, compute_signals


@dataclass
class SequenceSegmentationResult:
    sequence: RepresentationSequence
    config: SegmentationConfig
    signals: SignalSet
    gait_events: GaitEventResult
    contact_mask: np.ndarray
    primary: PrimaryRuleResult
    hesitation_mask: np.ndarray
    segments: list[SegmentRecord]
    quality_flags: dict[str, bool]
    quality_metrics: dict[str, object]
    segmentation_success: bool
    warnings: tuple[str, ...]
    output_paths: dict[str, Path] = field(default_factory=dict)


def _upstream_representation_success(sequence: RepresentationSequence) -> bool:
    diagnostics = sequence.metadata.get("quality_diagnostics", {})
    if isinstance(diagnostics, dict):
        return bool(diagnostics.get("representation_success", False))
    return False


def _finite_signal_arrays(signals: SignalSet) -> bool:
    return bool(
        all(
            np.all(np.isfinite(value))
            for value in signals.__dict__.values()
            if isinstance(value, np.ndarray)
        )
    )


def _gait_event_arrays_valid(gait_events: GaitEventResult, frame_count: int) -> bool:
    arrays = (
        gait_events.left_heel_strikes,
        gait_events.right_heel_strikes,
        gait_events.left_toe_offs,
        gait_events.right_toe_offs,
    )
    for array in arrays:
        values = np.asarray(array, dtype=np.int32)
        if values.size and (np.any(values < 0) or np.any(values >= frame_count)):
            return False
        if values.size and np.any(np.diff(values) < 0):
            return False
    return gait_events.contact_mask.shape == (frame_count, 2)


def _primary_label_exclusive(primary: PrimaryRuleResult) -> bool:
    labels = np.asarray(primary.primary_label_index, dtype=np.int32)
    if labels.ndim != 1 or labels.size == 0:
        return False
    return bool(np.all((0 <= labels) & (labels < len(PRIMARY_LABELS))))


def _hesitation_requires_locomotion(primary: PrimaryRuleResult, hesitation_mask: np.ndarray) -> bool:
    hesitation = np.asarray(hesitation_mask, dtype=bool)
    locomotion = np.asarray(primary.masks["walk"], dtype=bool) | np.asarray(primary.masks["turn"], dtype=bool)
    return not bool(np.any(hesitation & ~locomotion))


def _segments_cover_all_frames(segments: list[SegmentRecord], frame_count: int) -> bool:
    if frame_count == 0:
        return not segments
    if not segments:
        return False
    if segments[0].start_frame != 0 or segments[-1].end_frame != frame_count - 1:
        return False
    cursor = 0
    for segment in segments:
        if segment.start_frame != cursor:
            return False
        if segment.end_frame < segment.start_frame:
            return False
        cursor = segment.end_frame + 1
    return cursor == frame_count


def _quality_metrics(
    sequence: RepresentationSequence,
    gait_events: GaitEventResult,
    primary: PrimaryRuleResult,
    hesitation_mask: np.ndarray,
    segments: list[SegmentRecord],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "frame_count": sequence.num_frames,
        "segment_count": len(segments),
        "hesitation_frame_count": int(np.count_nonzero(hesitation_mask)),
        "contact_stability_score": float(gait_events.contact_stability_score),
        "left_contact_ratio": float(np.mean(gait_events.left_contact_mask.astype(np.float32))),
        "right_contact_ratio": float(np.mean(gait_events.right_contact_mask.astype(np.float32))),
        "valid_frame_ratio": float(np.mean(np.asarray(sequence.arrays["valid_frame_mask"], dtype=np.float32))),
        "representation_quality_score_mean": float(
            np.mean(np.asarray(sequence.arrays["representation_quality_score"], dtype=np.float32))
        ),
    }
    for label in PRIMARY_LABELS:
        frame_count = int(np.count_nonzero(primary.primary_label == label))
        metrics[f"{label}_frame_count"] = frame_count
        metrics[f"{label}_ratio"] = float(frame_count / max(sequence.num_frames, 1))
    return metrics


def segment_sequence(
    sequence: RepresentationSequence,
    *,
    config: SegmentationConfig | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
    generate_diagnostics: bool | None = None,
) -> SequenceSegmentationResult:
    config = config or SegmentationConfig()
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    generate_diagnostics = False

    signals = compute_signals(sequence.arrays, sequence.fps, config)
    gait_events = detect_gait_events(
        sequence.arrays["left_foot_contact"],
        sequence.arrays["right_foot_contact"],
        sequence.fps,
        left_heel_strike=sequence.arrays["left_heel_strike"],
        right_heel_strike=sequence.arrays["right_heel_strike"],
        left_toe_off=sequence.arrays["left_toe_off"],
        right_toe_off=sequence.arrays["right_toe_off"],
    )
    primary = run_rule_based_primary_segmentation(signals, config)
    hesitation_mask = compute_hesitation_mask(signals, primary.primary_label_index, config)
    segments = build_segment_records(primary, hesitation_mask, signals.time_sec, sequence.fps)

    quality_flags = {
        "input_contract_valid": True,
        "upstream_representation_success": _upstream_representation_success(sequence),
        "finite_signal_arrays": _finite_signal_arrays(signals),
        "contact_mask_valid": gait_events.contact_mask.shape == (sequence.num_frames, 2),
        "gait_event_arrays_valid": _gait_event_arrays_valid(gait_events, sequence.num_frames),
        "primary_label_exclusive": _primary_label_exclusive(primary),
        "hesitation_requires_locomotion": _hesitation_requires_locomotion(primary, hesitation_mask),
        "segments_cover_all_frames": _segments_cover_all_frames(segments, sequence.num_frames),
    }
    segmentation_success = bool(
        all(
            quality_flags[key]
            for key in (
                "input_contract_valid",
                "finite_signal_arrays",
                "contact_mask_valid",
                "gait_event_arrays_valid",
                "primary_label_exclusive",
                "hesitation_requires_locomotion",
                "segments_cover_all_frames",
            )
        )
    )
    warnings = tuple(dict.fromkeys(gait_events.warnings))
    result = SequenceSegmentationResult(
        sequence=sequence,
        config=config,
        signals=signals,
        gait_events=gait_events,
        contact_mask=np.asarray(gait_events.contact_mask, dtype=bool),
        primary=primary,
        hesitation_mask=np.asarray(hesitation_mask, dtype=bool),
        segments=segments,
        quality_flags=quality_flags,
        quality_metrics=_quality_metrics(sequence, gait_events, primary, hesitation_mask, segments),
        segmentation_success=segmentation_success,
        warnings=warnings,
    )

    if write_outputs:
        output_paths = write_sequence_outputs(result, output_root)
        result.output_paths.update(output_paths)
    return result


def process_one_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: SegmentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    generate_diagnostics: bool | None = None,
) -> SequenceSegmentationResult:
    config = config or SegmentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    npz_path, json_path = resolve_sequence_paths(input_root, subset, subject_id, trial_id)
    sequence = load_representation_sequence(npz_path, json_path)
    return segment_sequence(
        sequence,
        config=config,
        output_dir=output_dir,
        write_outputs=True,
        generate_diagnostics=generate_diagnostics,
    )


def _has_expected_outputs(
    output_root: Path,
    subset_name: str,
    stem: str,
    *,
    include_diagnostics: bool,
) -> bool:
    subset_dir = output_root / subset_name
    required = [
        subset_dir / f"{stem}.json",
        subset_dir / f"{stem}.csv",
    ]
    return all(path.exists() for path in required)


def process_subset(
    subset: str,
    *,
    config: SegmentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
    generate_diagnostics: bool | None = None,
    skip_existing: bool = False,
) -> dict[str, object]:
    config = config or SegmentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    diagnostics_enabled = config.diagnostic_generate_by_default if generate_diagnostics is None else bool(generate_diagnostics)

    rows: list[dict[str, object]] = []
    processed = 0
    failed = 0
    for npz_path in iter_sequence_npzs(input_root, subset_name=subset, trial_contains=trial_contains, max_trials=max_trials):
        if skip_existing and _has_expected_outputs(
            output_root,
            npz_path.parent.name,
            npz_path.stem,
            include_diagnostics=diagnostics_enabled,
        ):
            print(f"[skip] {npz_path.parent.name}/{npz_path.stem}")
            continue
        try:
            sequence = load_representation_sequence(npz_path)
            print(f"[{processed + 1}] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}")
            result = segment_sequence(
                sequence,
                config=config,
                output_dir=output_root,
                write_outputs=True,
                generate_diagnostics=diagnostics_enabled,
            )
            rows.append(summary_row(result))
            processed += 1
        except Exception as error:
            failed += 1
            print(f"[error] {npz_path}: {error}")
            rows.append(failure_summary_row(npz_path.parent.name, npz_path.stem, error))
    summary_path = write_subset_summary(rows, output_root, subset, config.subset_summary_name)
    return {
        "subset_name": subset,
        "processed_trials": processed,
        "failed_trials": failed,
        "summary_path": summary_path,
    }


def process_all_subsets(
    *,
    config: SegmentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
    generate_diagnostics: bool | None = None,
    skip_existing: bool = False,
) -> list[dict[str, object]]:
    config = config or SegmentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    summaries: list[dict[str, object]] = []
    for subset_dir in iter_subset_dirs(input_root):
        if not iter_sequence_npzs(input_root, subset_name=subset_dir.name, max_trials=1):
            continue
        summaries.append(
            process_subset(
                subset_dir.name,
                config=config,
                input_dir=input_root,
                output_dir=output_dir,
                max_trials=max_trials_per_subset,
                generate_diagnostics=generate_diagnostics,
                skip_existing=skip_existing,
            )
        )
    return summaries
