"""End-to-end pipeline for F_Description."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import DescriptionConfig
from .descriptor_rules import DescriptorReference, build_reference, derive_sequence_profile
from .export_utils import write_sequence_output
from .io_utils import DescriptionSequence, load_one_sequence, load_subset_reference_tables, load_subset_sequences


@dataclass
class DescriptionResult:
    sequence: DescriptionSequence
    config: DescriptionConfig
    reference: DescriptorReference
    profile: dict[str, object]
    warnings: tuple[str, ...]
    source_inputs: dict[str, str]
    output_paths: dict[str, Path] = field(default_factory=dict)


def describe_one_sequence(
    sequence: DescriptionSequence,
    *,
    reference: DescriptorReference,
    config: DescriptionConfig | None = None,
    output_dir: str | Path | None = None,
    reference_subset_name: str | None = None,
) -> DescriptionResult:
    config = config or DescriptionConfig()
    output_root = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    profile = derive_sequence_profile(sequence, reference)
    reference_subset = str(reference_subset_name or sequence.subset_name)
    subset_walk_csv = Path(config.input_e_dir) / reference_subset / config.walk_features_csv_name
    subset_turn_csv = Path(config.input_e_dir) / reference_subset / config.turn_features_csv_name
    source_inputs = {
        **sequence.source_inputs,
        "reference_subset": reference_subset,
        "subset_walk_csv": str(subset_walk_csv),
        "subset_turn_csv": str(subset_turn_csv),
    }
    result = DescriptionResult(
        sequence=sequence,
        config=config,
        reference=reference,
        profile=profile,
        warnings=sequence.warnings,
        source_inputs=source_inputs,
    )
    write_sequence_output(result, output_root)
    return result


def process_one_sequence(
    subset_name: str,
    subject_id: str,
    trial_id: str,
    *,
    config: DescriptionConfig | None = None,
    input_e_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> DescriptionResult:
    config = config or DescriptionConfig()
    if input_e_dir is not None:
        config = DescriptionConfig(
            input_e_dir=Path(input_e_dir),
            output_dir=Path(output_dir) if output_dir is not None else Path(config.output_dir),
            language=config.language,
            walk_features_csv_name=config.walk_features_csv_name,
            turn_features_csv_name=config.turn_features_csv_name,
        )
    elif output_dir is not None:
        config = DescriptionConfig(
            input_e_dir=Path(config.input_e_dir),
            output_dir=Path(output_dir),
            language=config.language,
            walk_features_csv_name=config.walk_features_csv_name,
            turn_features_csv_name=config.turn_features_csv_name,
        )
    walk_df, turn_df = load_subset_reference_tables(subset_name, config)
    reference = build_reference(walk_df, turn_df)
    sequence = load_one_sequence(subset_name, subject_id, trial_id, config=config)
    return describe_one_sequence(sequence, reference=reference, config=config, output_dir=output_dir)


def process_subset(
    subset_name: str,
    *,
    config: DescriptionConfig | None = None,
    input_e_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> dict[str, object]:
    config = config or DescriptionConfig()
    if input_e_dir is not None or output_dir is not None:
        config = DescriptionConfig(
            input_e_dir=Path(input_e_dir) if input_e_dir is not None else Path(config.input_e_dir),
            output_dir=Path(output_dir) if output_dir is not None else Path(config.output_dir),
            language=config.language,
            walk_features_csv_name=config.walk_features_csv_name,
            turn_features_csv_name=config.turn_features_csv_name,
        )
    walk_df, turn_df = load_subset_reference_tables(subset_name, config)
    reference = build_reference(walk_df, turn_df)
    sequences = load_subset_sequences(subset_name, config=config, trial_contains=trial_contains, max_trials=max_trials)
    processed = 0
    for index, sequence in enumerate(sequences, start=1):
        describe_one_sequence(sequence, reference=reference, config=config, output_dir=output_dir)
        processed += 1
        print(f"[{index}] {sequence.subset_name}/{sequence.subject_id}/{sequence.trial_id}")
    return {
        "subset_name": subset_name,
        "processed_trials": processed,
        "output_dir": str((Path(output_dir) if output_dir is not None else Path(config.output_dir)) / subset_name),
    }


def process_all_subsets(
    *,
    config: DescriptionConfig | None = None,
    input_e_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_trials_per_subset: int | None = None,
) -> list[dict[str, object]]:
    config = config or DescriptionConfig()
    if input_e_dir is not None or output_dir is not None:
        config = DescriptionConfig(
            input_e_dir=Path(input_e_dir) if input_e_dir is not None else Path(config.input_e_dir),
            output_dir=Path(output_dir) if output_dir is not None else Path(config.output_dir),
            language=config.language,
            walk_features_csv_name=config.walk_features_csv_name,
            turn_features_csv_name=config.turn_features_csv_name,
        )
    summaries: list[dict[str, object]] = []
    for subset_dir in sorted(path for path in Path(config.input_e_dir).iterdir() if path.is_dir()):
        if not any(subset_dir.glob("*.json")):
            continue
        summaries.append(
            process_subset(
                subset_dir.name,
                config=config,
                output_dir=output_dir,
                max_trials=max_trials_per_subset,
            )
        )
    return summaries
