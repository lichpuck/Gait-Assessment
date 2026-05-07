"""End-to-end pipeline for converting raw SMPL sequences into Joint3D outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .axis_semantics import AxisSemanticsError, AxisSemanticsResult, apply_rotation, estimate_axis_semantics
from .config import AuditionConfig
from .export_utils import (
    failed_summary_row,
    failed_sequence_summary_row,
    safe_output_name,
    sequence_output_stem,
    skipped_summary_row,
    summary_row,
    update_summary_csv,
    write_metadata_json,
    write_sequence_outputs,
)
from .io_utils import (
    SequenceRecord,
    list_subset_paths,
    load_one_sequence,
    load_pkl_dataset,
    record_from_payload,
    resolve_subset_path,
)
from .smpl_forward import smpl_to_joints
from .support_points import SupportPointResult, compute_support_points
from .visualize import create_semantic_plot


@dataclass(frozen=True)
class AuditionFailure:
    subset_name: str
    subject_id: str
    trial_id: str
    message: str


@dataclass
class AuditionSequenceResult:
    record: SequenceRecord
    config: AuditionConfig
    raw_joints: np.ndarray
    joints_3d: np.ndarray
    trans_canonical: np.ndarray
    support: SupportPointResult
    axis: AxisSemanticsResult
    warnings: tuple[str, ...]
    output_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class SkippedSequenceResult:
    record: SequenceRecord
    config: AuditionConfig
    skip_reason: str
    output_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def warnings(self) -> tuple[str, ...]:
        return ()


@dataclass
class FailedSequenceResult:
    record: SequenceRecord
    config: AuditionConfig
    failure_reason: str
    message: str
    output_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def warnings(self) -> tuple[str, ...]:
        return ()


AuditionResult = AuditionSequenceResult | SkippedSequenceResult | FailedSequenceResult


def _short_sequence_reason(record: SequenceRecord, config: AuditionConfig) -> str | None:
    if record.duration_sec < float(config.min_duration_sec):
        return "duration_lt_3s"
    return None


def _result_paths(output_dir: str | Path, record: SequenceRecord) -> dict[str, Path]:
    subset_dir = Path(output_dir) / safe_output_name(record.subset_name)
    stem = sequence_output_stem(record.subset_name, record.subject_id, record.trial_id)
    return {
        "npz": subset_dir / f"{stem}.npz",
        "json": subset_dir / f"{stem}.json",
        "png": subset_dir / f"{stem}.png",
    }


def canonicalize_record(
    record: SequenceRecord,
    *,
    config: AuditionConfig | None = None,
    write_outputs: bool = True,
    generate_diagnostics: bool = True,
    output_dir: str | Path | None = None,
    update_summary: bool = True,
) -> AuditionResult:
    config = config or AuditionConfig()
    output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    skip_reason = _short_sequence_reason(record, config)
    if skip_reason is not None:
        result = SkippedSequenceResult(record=record, config=config, skip_reason=skip_reason)
        if write_outputs and update_summary:
            result.output_paths["summary_csv"] = update_summary_csv(skipped_summary_row(result), config.summary_csv)
        return result

    forward_result = smpl_to_joints(
        record.pose,
        record.trans,
        record.beta,
        model_root=config.model_root,
        smpl_model_path=config.smpl_model_path,
        batch_size=config.smpl_batch_size,
    )
    raw_joints = forward_result.joints
    try:
        axis = estimate_axis_semantics(raw_joints, record.trans, config)
    except AxisSemanticsError as error:
        result = FailedSequenceResult(
            record=record,
            config=config,
            failure_reason=error.reason,
            message=str(error),
        )
        if write_outputs and update_summary:
            result.output_paths["summary_csv"] = update_summary_csv(failed_sequence_summary_row(result), config.summary_csv)
        return result

    joints_3d = apply_rotation(raw_joints, axis.R_total).astype(np.float32, copy=False)
    trans_canonical = apply_rotation(record.trans, axis.R_total).astype(np.float32, copy=False)
    support = compute_support_points(joints_3d, record.fps, config)
    warnings = (*forward_result.backend_notes, *axis.warnings, *support.warnings)

    result = AuditionSequenceResult(
        record=record,
        config=config,
        raw_joints=raw_joints.astype(np.float32, copy=False),
        joints_3d=joints_3d,
        trans_canonical=trans_canonical,
        support=support,
        axis=axis,
        warnings=warnings,
    )

    if write_outputs:
        artifact_paths = write_sequence_outputs(result, output_dir)
        result.output_paths["npz"] = artifact_paths["npz"]
        result.output_paths["json"] = artifact_paths["json"]
        if generate_diagnostics:
            title = f"{record.subset_name} | {record.subject_id} | {record.trial_id}"
            create_semantic_plot(
                joints_3d=result.joints_3d,
                trans_canonical=result.trans_canonical,
                support_points=result.support.support_points,
                output_path=artifact_paths["png"],
                title=title,
                config=config,
            )
            result.output_paths["png"] = artifact_paths["png"]
        if update_summary:
            result.output_paths["summary_csv"] = update_summary_csv(summary_row(result), config.summary_csv)
        write_metadata_json(result, artifact_paths["json"])
    return result


def process_one_sequence(
    subset: str | Path,
    subject_id: str,
    trial_id: str,
    *,
    config: AuditionConfig | None = None,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    generate_diagnostics: bool = True,
) -> AuditionResult:
    config = config or AuditionConfig()
    raw_data_dir = raw_data_dir or config.raw_data_dir
    output_dir = output_dir or config.output_dir
    record = load_one_sequence(subset, subject_id, trial_id, raw_data_dir)
    return canonicalize_record(
        record,
        config=config,
        write_outputs=True,
        generate_diagnostics=generate_diagnostics,
        output_dir=output_dir,
        update_summary=True,
    )


def process_subset(
    subset: str | Path,
    *,
    config: AuditionConfig | None = None,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    subject_filter: str | None = None,
    trial_ids: set[str] | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
    generate_diagnostics: bool = True,
    skip_existing: bool = False,
) -> dict[str, object]:
    config = config or AuditionConfig()
    raw_data_dir = raw_data_dir or config.raw_data_dir
    output_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    subset_path = resolve_subset_path(subset, raw_data_dir)
    dataset = load_pkl_dataset(subset_path)

    processed_trials = 0
    skipped_trials = 0
    failed_trials = 0
    inspected_trials = 0

    for subject_id, trials in dataset.items():
        if subject_filter is not None and str(subject_id) != str(subject_filter):
            continue
        for trial_id, payload in trials.items():
            trial_id_str = str(trial_id)
            if trial_ids is not None and trial_id_str not in trial_ids:
                continue
            if trial_contains is not None and trial_contains not in trial_id_str:
                continue
            inspected_trials += 1
            if max_trials is not None and inspected_trials > max_trials:
                break

            try:
                record = record_from_payload(subset_path.stem, str(subject_id), trial_id_str, payload, subset_path)
                if skip_existing:
                    existing_paths = _result_paths(output_dir, record)
                    required_paths = [existing_paths["npz"], existing_paths["json"]]
                    if generate_diagnostics:
                        required_paths.append(existing_paths["png"])
                    if all(path.exists() for path in required_paths):
                        continue
                result = canonicalize_record(
                    record,
                    config=config,
                    write_outputs=True,
                    generate_diagnostics=generate_diagnostics,
                    output_dir=output_dir,
                    update_summary=True,
                )
                if isinstance(result, SkippedSequenceResult):
                    skipped_trials += 1
                elif isinstance(result, FailedSequenceResult):
                    failed_trials += 1
                else:
                    processed_trials += 1
            except Exception as exc:
                failed_trials += 1
                failure = AuditionFailure(
                    subset_name=subset_path.stem,
                    subject_id=str(subject_id),
                    trial_id=trial_id_str,
                    message=str(exc),
                )
                update_summary_csv(failed_summary_row(failure), config.summary_csv)
        if max_trials is not None and inspected_trials >= max_trials:
            break

    return {
        "subset_name": subset_path.stem,
        "processed_trials": processed_trials,
        "skipped_trials": skipped_trials,
        "failed_trials": failed_trials,
        "inspected_trials": min(inspected_trials, max_trials) if max_trials is not None else inspected_trials,
        "summary_path": config.summary_csv,
    }


def process_all_subsets(
    *,
    config: AuditionConfig | None = None,
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
    subset_names: set[str] | None = None,
    generate_diagnostics: bool = True,
    skip_existing: bool = False,
) -> list[dict[str, object]]:
    config = config or AuditionConfig()
    raw_data_dir = raw_data_dir or config.raw_data_dir
    output_dir = output_dir or config.output_dir
    subset_paths = list_subset_paths(raw_data_dir)
    if subset_names is not None:
        subset_paths = [path for path in subset_paths if path.stem in subset_names]

    summaries = []
    for subset_path in subset_paths:
        summaries.append(
            process_subset(
                subset_path,
                config=config,
                raw_data_dir=raw_data_dir,
                output_dir=output_dir,
                max_trials=max_trials_per_subset,
                generate_diagnostics=generate_diagnostics,
                skip_existing=skip_existing,
            )
        )
    return summaries
