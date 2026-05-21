"""I/O helpers for raw CARE-PD subset pickle files.

This module is intentionally local to ``scripts/A_Audition``.  It replaces the
former dependency on ``care_pd_pipeline`` so the A-stage command-line scripts can
load and validate raw SMPL records on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np


BETA_ATOL = 1e-5
BETA_RTOL = 1e-5
REQUIRED_SEQUENCE_KEYS = ("pose", "trans", "beta", "fps")


@dataclass(frozen=True)
class SequenceRecord:
    subset_name: str
    subject_id: str
    trial_id: str
    pose: np.ndarray
    trans: np.ndarray
    beta: np.ndarray
    fps: float
    metadata: dict[str, object]
    source_path: Path

    @property
    def num_frames(self) -> int:
        return int(self.pose.shape[0])

    @property
    def duration_sec(self) -> float:
        return float(self.num_frames) / float(self.fps)


def _as_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def normalize_beta(beta: object, num_frames: int | None = None) -> np.ndarray:
    """Return a single ``(1, 10)`` SMPL beta row.

    Some external sources store one beta vector per frame.  A_Audition treats
    body shape as sequence-level metadata, so per-frame beta is accepted only
    when all rows are constant within a small floating-point tolerance.
    """

    beta_array = _as_array(beta, dtype=np.float32)
    if beta_array.shape == (10,):
        return beta_array.reshape(1, 10)
    if beta_array.shape == (1, 10):
        return beta_array.astype(np.float32, copy=False)
    if beta_array.ndim == 2 and beta_array.shape[1] == 10:
        if num_frames is not None and beta_array.shape[0] != int(num_frames):
            raise ValueError(f"beta has shape {beta_array.shape}; expected (1, 10), (10,), or ({num_frames}, 10)")
        if beta_array.shape[0] == 0:
            raise ValueError("beta has zero rows")
        reference = beta_array[:1]
        if not np.allclose(beta_array, reference, atol=BETA_ATOL, rtol=BETA_RTOL):
            raise ValueError("per-frame beta must be constant across frames")
        return reference.astype(np.float32, copy=False)
    raise ValueError(f"beta must have shape (1, 10), (10,), or (T, 10), got {beta_array.shape}")


def validate_sequence(payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"valid": False, "reason": "sequence payload must be a dict"}

    missing = [key for key in REQUIRED_SEQUENCE_KEYS if key not in payload]
    if missing:
        return {"valid": False, "reason": f"missing required keys: {', '.join(missing)}"}

    try:
        pose = _as_array(payload["pose"], dtype=np.float32)
        trans = _as_array(payload["trans"], dtype=np.float32)
    except Exception as exc:
        return {"valid": False, "reason": f"pose/trans could not be converted to float arrays: {exc}"}

    if pose.ndim != 2 or pose.shape[1] != 72:
        return {"valid": False, "reason": f"pose must have shape (T, 72), got {pose.shape}"}
    if trans.ndim != 2 or trans.shape[1] != 3:
        return {"valid": False, "reason": f"trans must have shape (T, 3), got {trans.shape}"}
    if pose.shape[0] != trans.shape[0]:
        return {"valid": False, "reason": f"pose/trans frame count mismatch: {pose.shape[0]} vs {trans.shape[0]}"}
    if pose.shape[0] == 0:
        return {"valid": False, "reason": "sequence has zero frames"}
    if not np.all(np.isfinite(pose)):
        return {"valid": False, "reason": "pose contains non-finite values"}
    if not np.all(np.isfinite(trans)):
        return {"valid": False, "reason": "trans contains non-finite values"}

    try:
        beta = normalize_beta(payload["beta"], num_frames=int(pose.shape[0]))
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}
    if not np.all(np.isfinite(beta)):
        return {"valid": False, "reason": "beta contains non-finite values"}

    try:
        fps = float(np.asarray(payload["fps"]).reshape(()))
    except Exception as exc:
        return {"valid": False, "reason": f"fps must be a scalar number: {exc}"}
    if not np.isfinite(fps) or fps <= 0.0:
        return {"valid": False, "reason": f"fps must be a positive finite scalar, got {payload['fps']!r}"}

    return {
        "valid": True,
        "reason": "ok",
        "pose": pose.astype(np.float32, copy=False),
        "trans": trans.astype(np.float32, copy=False),
        "beta": beta.astype(np.float32, copy=False),
        "fps": fps,
    }


def record_from_payload(
    subset_name: str,
    subject_id: str,
    trial_id: str,
    payload: dict[str, object],
    source_path: str | Path,
) -> SequenceRecord:
    validated = validate_sequence(payload)
    if not validated["valid"]:
        raise ValueError(f"Invalid sequence {subset_name}/{subject_id}/{trial_id}: {validated['reason']}")

    metadata = {key: value for key, value in payload.items() if key not in REQUIRED_SEQUENCE_KEYS}
    return SequenceRecord(
        subset_name=str(subset_name),
        subject_id=str(subject_id),
        trial_id=str(trial_id),
        pose=np.asarray(validated["pose"], dtype=np.float32),
        trans=np.asarray(validated["trans"], dtype=np.float32),
        beta=np.asarray(validated["beta"], dtype=np.float32),
        fps=float(validated["fps"]),
        metadata=metadata,
        source_path=Path(source_path),
    )


def _load_with_joblib(path: Path) -> object:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required for compressed raw CARE-PD .pkl files") from exc
    return joblib.load(path)


def _load_with_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def load_pkl_dataset(path: str | Path) -> dict[Any, Any]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        payload = _load_with_joblib(file_path)
    except Exception as joblib_error:
        try:
            payload = _load_with_pickle(file_path)
        except Exception as pickle_error:
            raise ValueError(
                f"Could not load {file_path} with joblib or pickle: "
                f"joblib={joblib_error}; pickle={pickle_error}"
            ) from pickle_error

    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a top-level dict dataset")
    return payload


def resolve_subset_path(subset: str | Path, raw_data_dir: str | Path) -> Path:
    subset_path = Path(subset).expanduser()
    raw_dir = Path(raw_data_dir).expanduser()

    candidates: list[Path] = []
    if subset_path.exists():
        candidates.append(subset_path)
    if subset_path.suffix == ".pkl":
        candidates.append(raw_dir / subset_path.name)
    else:
        candidates.append(raw_dir / f"{subset_path.name}.pkl")
        candidates.append(raw_dir / subset_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve subset {subset!s} under {raw_dir}")


def iterate_sequences(
    dataset: dict[str, Any],
    *,
    subset_name: str,
    source_path: str | Path,
) -> Iterable[SequenceRecord]:
    for subject_id, trials in dataset.items():
        if not isinstance(trials, dict):
            continue
        for trial_id, payload in trials.items():
            if not isinstance(payload, dict):
                continue
            yield record_from_payload(subset_name, str(subject_id), str(trial_id), payload, source_path)


def _lookup_key(mapping: dict[Any, Any], requested: str) -> Any:
    if requested in mapping:
        return requested
    for key in mapping:
        if str(key) == str(requested):
            return key
    raise KeyError(requested)


def load_one_sequence(
    subset: str | Path,
    subject_id: str,
    trial_id: str,
    raw_data_dir: str | Path,
) -> SequenceRecord:
    subset_path = resolve_subset_path(subset, raw_data_dir)
    dataset = load_pkl_dataset(subset_path)
    subject_key = str(subject_id)
    trial_key = str(trial_id)
    try:
        subject_lookup_key = _lookup_key(dataset, subject_key)
        trials = dataset[subject_lookup_key]
        if not isinstance(trials, dict):
            raise KeyError(subject_key)
        trial_lookup_key = _lookup_key(trials, trial_key)
        payload = trials[trial_lookup_key]
    except KeyError as exc:
        raise KeyError(f"Sequence not found: {subset_path.stem}/{subject_key}/{trial_key}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Sequence payload must be a dict: {subset_path.stem}/{subject_key}/{trial_key}")
    return record_from_payload(subset_path.stem, subject_key, trial_key, payload, subset_path)


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
