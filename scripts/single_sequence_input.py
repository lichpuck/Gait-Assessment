"""Normalize external SMPL pickle inputs into one validated sequence record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np

from scripts.A_Audition.io_utils import SequenceRecord, validate_sequence


DEFAULT_SUBSET_NAME = "single_sequence"
DEFAULT_TRIAL_ID = "trial_0001"


def _safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value or "input"


@dataclass(frozen=True)
class NormalizedInput:
    sequence: SequenceRecord
    input_path: Path
    input_kind: str
    schema_version: str


def _load_pickle(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        import joblib

        return joblib.load(file_path)
    except Exception as joblib_error:
        with file_path.open("rb") as handle:
            try:
                return pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                return pickle.load(handle, encoding="latin1")
            except Exception as pickle_error:
                raise ValueError(
                    f"Could not load {file_path} with joblib or pickle: "
                    f"joblib={joblib_error}; pickle={pickle_error}"
                ) from pickle_error


def _extract_optional_str(payload: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _build_flat_sequence(payload: dict[str, object], source_path: Path) -> SequenceRecord:
    validated = validate_sequence(payload)
    if not validated["valid"]:
        raise ValueError(f"Invalid single-sequence input: {validated['reason']}")

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"pose", "trans", "beta", "fps"}
    }
    metadata.setdefault("input_contract", "flat_single_sequence")

    subject_id = _extract_optional_str(payload, ("subject_id", "subject", "sequence_id", "name"))
    trial_id = _extract_optional_str(payload, ("trial_id", "trial", "clip_id", "sequence_name"))
    subset_name = _extract_optional_str(payload, ("subset", "dataset", "source_subset"))

    stem = _safe_name(source_path.stem)
    sequence = SequenceRecord(
        subset_name=_safe_name(subset_name or DEFAULT_SUBSET_NAME),
        subject_id=_safe_name(subject_id or stem),
        trial_id=_safe_name(trial_id or DEFAULT_TRIAL_ID),
        pose=np.asarray(validated["pose"], dtype=np.float32),
        trans=np.asarray(validated["trans"], dtype=np.float32),
        beta=np.asarray(validated["beta"], dtype=np.float32),
        fps=float(validated["fps"]),
        metadata=metadata,
        source_path=source_path,
    )
    return sequence


def _flatten_nested_sequences(payload: dict[str, Any], source_path: Path) -> list[SequenceRecord]:
    sequences: list[SequenceRecord] = []
    subset_name = _safe_name(source_path.stem)
    for subject_id, trials in payload.items():
        if not isinstance(trials, dict):
            continue
        for trial_id, sample in trials.items():
            if not isinstance(sample, dict):
                continue
            validated = validate_sequence(sample)
            if not validated["valid"]:
                raise ValueError(
                    f"Invalid nested sequence {subject_id}/{trial_id} in {source_path.name}: {validated['reason']}"
                )
            metadata = {
                key: value
                for key, value in sample.items()
                if key not in {"pose", "trans", "beta", "fps"}
            }
            metadata.setdefault("input_contract", "nested_dataset_sequence")
            sequences.append(
                SequenceRecord(
                    subset_name=subset_name,
                    subject_id=_safe_name(subject_id),
                    trial_id=_safe_name(trial_id),
                    pose=np.asarray(validated["pose"], dtype=np.float32),
                    trans=np.asarray(validated["trans"], dtype=np.float32),
                    beta=np.asarray(validated["beta"], dtype=np.float32),
                    fps=float(validated["fps"]),
                    metadata=metadata,
                    source_path=source_path,
                )
            )
    return sequences


def normalize_input_pkl(input_pkl: str | Path) -> NormalizedInput:
    input_path = Path(input_pkl).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    payload = _load_pickle(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{input_path.name} must contain a top-level dict payload")

    if {"pose", "trans", "beta", "fps"}.issubset(payload.keys()):
        return NormalizedInput(
            sequence=_build_flat_sequence(payload, input_path),
            input_path=input_path,
            input_kind="flat_single_sequence",
            schema_version="1.0",
        )

    sequences = _flatten_nested_sequences(payload, input_path)
    if not sequences:
        raise ValueError(
            f"{input_path.name} is neither a flat single-sequence payload nor a nested dataset payload with valid sequences"
        )
    if len(sequences) != 1:
        raise ValueError(
            f"{input_path.name} resolves to {len(sequences)} sequences; the one-command pipeline requires exactly 1"
        )
    return NormalizedInput(
        sequence=sequences[0],
        input_path=input_path,
        input_kind="nested_dataset_sequence",
        schema_version="1.0",
    )
