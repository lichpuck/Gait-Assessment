"""End-to-end pipeline for CARE-PD C_Representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RepresentationConfig
from .export_utils import write_sequence_outputs
from .features import build_representation_features
from .io_utils import (
    CanonicalizedSequence,
    iter_sequence_npzs,
    iter_subset_dirs,
    load_canonicalized_sequence,
    resolve_sequence_paths,
)


@dataclass
class SequenceRepresentationResult:
    sequence: CanonicalizedSequence
    config: RepresentationConfig
    arrays: dict[str, np.ndarray]
    quality_flags: dict[str, bool]
    quality_metrics: dict[str, object]
    gait_summary: dict[str, object]
    warnings: tuple[str, ...]
    representation_success: bool
    output_paths: dict[str, Path] = field(default_factory=dict)


def _shape_consistency(arrays: dict[str, np.ndarray], frame_count: int) -> bool:
    for key, value in arrays.items():
        if value.ndim == 0:
            continue
        if key == "fps":
            continue
        if value.shape[0] != frame_count:
            return False
    return True


def represent_sequence(
    sequence: CanonicalizedSequence,
    *,
    config: RepresentationConfig | None = None,
    write_outputs: bool = True,
    output_dir: str | Path | None = None,
) -> SequenceRepresentationResult:
    config = config or RepresentationConfig()
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    feature_result = build_representation_features(
        sequence.joints_can,
        sequence.trans_can,
        sequence.pose_raw,
        sequence.R_global,
        sequence.R_total,
        sequence.fps,
        config,
    )
    arrays = dict(feature_result.arrays)
    shape_consistency = _shape_consistency(arrays, sequence.num_frames)
    finite_inputs = bool(
        np.all(np.isfinite(sequence.pose_raw))
        and np.all(np.isfinite(sequence.trans_raw))
        and np.all(np.isfinite(sequence.joints_can))
        and np.all(np.isfinite(sequence.trans_can))
        and np.all(np.isfinite(sequence.R_global))
        and np.all(np.isfinite(sequence.R_total))
        and np.isfinite(sequence.fps)
    )
    quality_flags = {
        "input_contract_valid": True,
        "finite_input_arrays": finite_inputs,
        "shape_consistency": shape_consistency,
        **feature_result.quality_flags,
    }
    quality_metrics = {
        **feature_result.quality_metrics,
        "input_frame_count": sequence.num_frames,
        "duration_sec": sequence.duration_sec,
    }
    representation_success = bool(
        quality_flags["input_contract_valid"]
        and quality_flags["finite_input_arrays"]
        and quality_flags["shape_consistency"]
        and quality_flags.get("finite_feature_arrays", False)
        and quality_flags.get("heading_available", False)
        and quality_flags.get("pelvis_orientation_available", False)
        and quality_flags.get("contact_probabilities_finite", False)
        and quality_flags.get("valid_frame_mask_nonempty", False)
    )
    result = SequenceRepresentationResult(
        sequence=sequence,
        config=config,
        arrays=arrays,
        quality_flags=quality_flags,
        quality_metrics=quality_metrics,
        gait_summary=feature_result.gait_summary,
        warnings=feature_result.warnings,
        representation_success=representation_success,
    )
    if write_outputs:
        write_sequence_outputs(result, output_root)
    return result


def process_one_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: RepresentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> SequenceRepresentationResult:
    config = config or RepresentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    npz_path, json_path = resolve_sequence_paths(input_root, subset, subject_id, trial_id)
    sequence = load_canonicalized_sequence(npz_path, json_path)
    return represent_sequence(sequence, config=config, output_dir=output_root, write_outputs=True)


def _has_expected_outputs(output_root: Path, subset_name: str, stem: str) -> bool:
    subset_dir = output_root / subset_name
    return (subset_dir / f"{stem}.npz").exists() and (subset_dir / f"{stem}.json").exists()


def process_subset(
    subset: str,
    *,
    config: RepresentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
    skip_existing: bool = False,
) -> dict[str, object]:
    config = config or RepresentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)

    processed = 0
    failed = 0
    skipped_existing = 0
    failures: list[dict[str, str]] = []
    for npz_path in iter_sequence_npzs(input_root, subset_name=subset, trial_contains=trial_contains, max_trials=max_trials):
        if skip_existing and _has_expected_outputs(output_root, npz_path.parent.name, npz_path.stem):
            skipped_existing += 1
            print(f"[skip-existing] {npz_path.parent.name}/{npz_path.stem}")
            continue
        try:
            sequence = load_canonicalized_sequence(npz_path)
            result = represent_sequence(sequence, config=config, output_dir=output_root, write_outputs=True)
        except Exception as error:
            failed += 1
            failures.append({"subset": npz_path.parent.name, "stem": npz_path.stem, "error": str(error)})
            print(f"[fail] {npz_path.parent.name}/{npz_path.stem}: {error}")
            continue
        processed += 1
        print(
            f"[ok] {result.sequence.subset_name}/{result.sequence.subject_id}/{result.sequence.trial_id} "
            f"frames={result.sequence.num_frames} valid={result.quality_metrics.get('valid_frame_ratio', 0.0):.3f}"
        )
    return {
        "subset_name": subset,
        "processed_trials": processed,
        "failed_trials": failed,
        "skipped_existing": skipped_existing,
        "failures": failures,
    }


def process_all_subsets(
    *,
    config: RepresentationConfig | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
    skip_existing: bool = False,
) -> list[dict[str, object]]:
    config = config or RepresentationConfig()
    input_root = Path(input_dir) if input_dir is not None else Path(config.input_dir)
    summaries: list[dict[str, object]] = []
    for subset_dir in iter_subset_dirs(input_root):
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
