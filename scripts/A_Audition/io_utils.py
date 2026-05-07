"""I/O helpers for raw CARE-PD subset pickle files."""

from __future__ import annotations

from pathlib import Path

from care_pd_pipeline.C_Sequence_Animation.io_utils import (
    AnimationSequence as SequenceRecord,
    _normalize_beta as normalize_beta,
    _record_from_payload as record_from_payload,
    iterate_sequences,
    load_one_sequence,
    load_pkl_dataset,
    resolve_subset_path,
    validate_sequence,
)


def list_subset_paths(raw_data_dir: str | Path) -> list[Path]:
    return sorted(Path(raw_data_dir).glob("*.pkl"))


__all__ = [
    "SequenceRecord",
    "iterate_sequences",
    "list_subset_paths",
    "load_one_sequence",
    "load_pkl_dataset",
    "normalize_beta",
    "record_from_payload",
    "resolve_subset_path",
    "validate_sequence",
]
