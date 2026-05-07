"""End-to-end simplified B canonicalization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CanonicalizationConfig
from .export_utils import ensure_subset_output_dir, sequence_output_stem, write_sequence_outputs
from .io_utils import (
    AuditionSequence,
    iter_sequence_npzs,
    load_audition_sequence,
    resolve_sequence_paths,
)
from .transform_solver import (
    TransformResult,
    ground_joint_points,
    per_frame_lowest_ground_points,
    solve_transform,
    transform_points,
)
from .visualize_diagnostics import create_diagnostic_plot


@dataclass
class SequenceCanonicalizationResult:
    sequence: AuditionSequence
    config: CanonicalizationConfig
    transform: TransformResult
    joints_can: np.ndarray
    trans_can: np.ndarray
    support_points_can: np.ndarray
    ground_joint_points_can: np.ndarray
    checks: dict[str, float | bool]
    warnings: tuple[str, ...] = ()
    output_paths: dict[str, Path] = field(default_factory=dict)
    support_joint_names: tuple[str, ...] = ()


CanonicalizationPipelineResult = SequenceCanonicalizationResult


def _compute_checks(result: SequenceCanonicalizationResult) -> dict[str, float | bool]:
    ground_z = result.ground_joint_points_can[:, :, 2].reshape(-1)
    finite_ground_z = ground_z[np.isfinite(ground_z)]
    ground_p = float("nan")
    if finite_ground_z.size:
        ground_p = float(np.percentile(finite_ground_z, result.config.ground_percentile))
    det = float(np.linalg.det(result.transform.R_global))
    body_up_after = np.asarray(result.transform.R_global, dtype=np.float64) @ np.asarray(
        result.transform.body.up_axis, dtype=np.float64
    )
    body_up_norm = float(np.linalg.norm(body_up_after))
    body_up_z = float(body_up_after[2] / body_up_norm) if body_up_norm > result.config.normal_eps else float("nan")
    return {
        "npz_contract_field_count": 4,
        "all_outputs_finite": bool(np.all(np.isfinite(result.joints_can)) and np.all(np.isfinite(result.trans_can))),
        "scale_enabled": bool(result.transform.scale.enabled),
        "s_global": float(result.transform.s_global),
        "first_root_xy_norm_m": float(np.linalg.norm(result.trans_can[0, :2])),
        "ground_z_percentile_after_translation_m": ground_p,
        "det_R_global": det,
        "floor_alignment_enabled": bool(result.transform.floor.enabled),
        "floor_tilt_before_deg": float(result.transform.floor.tilt_before_deg),
        "floor_tilt_after_deg": float(result.transform.floor.tilt_after_deg),
        "floor_residual_median_abs_m": float(result.transform.floor.residual_median_abs_m),
        "body_up_after_error": float(np.linalg.norm(body_up_after - np.array([0.0, 0.0, 1.0]))),
        "body_up_after_angle_deg": float(np.degrees(np.arccos(np.clip(body_up_z, -1.0, 1.0)))),
    }


def canonicalize_sequence(
    sequence: AuditionSequence,
    *,
    config: CanonicalizationConfig | None = None,
    write_outputs: bool = True,
    generate_diagnostics: bool = True,
    output_dir: str | Path | None = None,
) -> SequenceCanonicalizationResult:
    config = config or CanonicalizationConfig()
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    transform = solve_transform(sequence.joints_3d, sequence.trans_canonical, sequence.fps, config)
    joints_can = transform_points(sequence.joints_3d, transform.R_global, transform.t_global, scale=transform.s_global)
    trans_can = transform_points(
        sequence.trans_canonical,
        transform.R_global,
        transform.t_global,
        scale=transform.s_global,
    )
    ground_points_can = transform_points(
        ground_joint_points(sequence.joints_3d),
        transform.R_global,
        transform.t_global,
        scale=transform.s_global,
    )
    support_points_can = per_frame_lowest_ground_points(ground_points_can)

    result = SequenceCanonicalizationResult(
        sequence=sequence,
        config=config,
        transform=transform,
        joints_can=joints_can,
        trans_can=trans_can,
        support_points_can=support_points_can,
        ground_joint_points_can=ground_points_can,
        checks={},
        warnings=(),
        support_joint_names=transform.ground.support_joint_names,
    )
    result.checks = _compute_checks(result)

    if write_outputs:
        paths = write_sequence_outputs(result, output_root)
        if generate_diagnostics:
            title = f"{sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}"
            create_diagnostic_plot(
                joints_can,
                trans_can,
                support_points_can,
                sequence.fps,
                title,
                paths["png"],
                config,
            )
    return result


def run_canonicalization(
    npz_path: str | Path | AuditionSequence,
    *,
    json_path: str | Path | None = None,
    config: CanonicalizationConfig | None = None,
    write_outputs: bool = False,
    generate_diagnostics: bool = True,
    output_dir: str | Path | None = None,
) -> SequenceCanonicalizationResult:
    sequence = npz_path if isinstance(npz_path, AuditionSequence) else load_audition_sequence(npz_path, json_path)
    return canonicalize_sequence(
        sequence,
        config=config,
        write_outputs=write_outputs,
        generate_diagnostics=generate_diagnostics,
        output_dir=output_dir,
    )


def process_one_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: CanonicalizationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> SequenceCanonicalizationResult:
    config = config or CanonicalizationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    npz_path, json_path = resolve_sequence_paths(input_root, subset, subject_id, trial_id)
    sequence = load_audition_sequence(npz_path, json_path)
    return canonicalize_sequence(sequence, config=config, write_outputs=True, output_dir=output_root)


def process_subset(
    subset: str,
    *,
    config: CanonicalizationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    subject_filter: str | None = None,
    trial_ids: set[str] | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
    skip_existing: bool = False,
) -> dict[str, object]:
    config = config or CanonicalizationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    subset_output_dir = ensure_subset_output_dir(output_root, subset)

    processed = 0
    failed = 0
    skipped_existing = 0
    inspected = 0
    failures: list[dict[str, str]] = []
    for npz_path in iter_sequence_npzs(
        input_root,
        subset_name=subset,
        subject_filter=subject_filter,
        trial_ids=trial_ids,
        trial_contains=trial_contains,
        max_trials=max_trials,
    ):
        inspected += 1
        sequence = load_audition_sequence(npz_path)
        stem = sequence_output_stem(sequence.subset_name, sequence.subject_id, sequence.trial_id)
        expected = [subset_output_dir / f"{stem}{suffix}" for suffix in (".npz", ".json", ".png")]
        if skip_existing and all(path.exists() for path in expected):
            skipped_existing += 1
            print(f"[skip-existing] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}")
            continue
        try:
            result = canonicalize_sequence(sequence, config=config, write_outputs=True, output_dir=output_root)
        except Exception as error:
            failed += 1
            failures.append(
                {
                    "subset": sequence.subset_name,
                    "subject_id": sequence.subject_id,
                    "trial_id": sequence.trial_id,
                    "error": str(error),
                }
            )
            print(f"[fail] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}: {error}")
            continue
        processed += 1
        print(
            f"[ok] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id} "
            f"frames={sequence.num_frames} det={result.checks['det_R_global']:.6f}"
        )

    return {
        "subset_name": subset,
        "processed_trials": processed,
        "failed_trials": failed,
        "skipped_existing_trials": skipped_existing,
        "inspected_trials": inspected,
        "failures": failures,
    }


def process_all_subsets(
    *,
    config: CanonicalizationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
    subset_names: set[str] | None = None,
    skip_existing: bool = False,
) -> list[dict[str, object]]:
    config = config or CanonicalizationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    summaries: list[dict[str, object]] = []
    for subset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if subset_names is not None and subset_dir.name not in subset_names:
            continue
        summaries.append(
            process_subset(
                subset_dir.name,
                config=config,
                input_dir=input_root,
                output_dir=output_dir,
                max_trials=max_trials_per_subset,
                skip_existing=skip_existing,
            )
        )
    return summaries
