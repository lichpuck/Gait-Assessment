"""Input loading helpers for A_Audition artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np


REQUIRED_NPZ_KEYS = ("joints_3d", "trans_canonical", "pose_raw", "trans_raw", "fps")


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


@dataclass(frozen=True)
class AuditionSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    npz_path: Path
    json_path: Path
    joints_3d: np.ndarray
    trans_canonical: np.ndarray
    pose_raw: np.ndarray
    trans_raw: np.ndarray
    fps: float
    metadata: dict[str, object]

    @property
    def num_frames(self) -> int:
        return int(self.pose_raw.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def _identifiers_from_metadata(stem: str, subset_name: str, metadata: dict[str, object]) -> tuple[str, str]:
    subject_id = metadata.get("subject_id")
    trial_id = metadata.get("trial_id")
    if subject_id is not None and trial_id is not None:
        return str(subject_id), str(trial_id)

    parts = stem.split("__")
    if len(parts) >= 3:
        return str(parts[1]), "__".join(parts[2:])
    return "unknown_subject", stem.removeprefix(f"{subset_name}__")


def _validate_sequence_arrays(
    npz_path: Path,
    joints: np.ndarray,
    trans: np.ndarray,
    pose_raw: np.ndarray,
    trans_raw: np.ndarray,
    fps: float,
) -> None:
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"{npz_path} joints_3d must have shape (T, 24, 3), got {joints.shape}")
    if pose_raw.ndim != 2 or pose_raw.shape[1] != 72:
        raise ValueError(f"{npz_path} pose_raw must have shape (T, 72), got {pose_raw.shape}")
    expected_trans_shape = (pose_raw.shape[0], 3)
    if trans.shape != expected_trans_shape:
        raise ValueError(f"{npz_path} trans_canonical must have shape {expected_trans_shape}, got {trans.shape}")
    if trans_raw.shape != expected_trans_shape:
        raise ValueError(f"{npz_path} trans_raw must have shape {expected_trans_shape}, got {trans_raw.shape}")
    if joints.shape[0] != pose_raw.shape[0]:
        raise ValueError(f"{npz_path} joints_3d and pose_raw frame counts differ")
    if pose_raw.shape[0] < 2:
        raise ValueError(f"{npz_path} must contain at least 2 frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{npz_path} fps must be positive and finite, got {fps}")
    for name, array in {
        "joints_3d": joints,
        "trans_canonical": trans,
        "pose_raw": pose_raw,
        "trans_raw": trans_raw,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{npz_path} {name} contains non-finite values")


def load_audition_sequence(npz_path: str | Path, json_path: str | Path | None = None) -> AuditionSequence:
    npz_file = Path(npz_path)
    json_file = Path(json_path) if json_path is not None else npz_file.with_suffix(".json")
    if not npz_file.exists():
        raise FileNotFoundError(npz_file)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    metadata = _load_json(json_file)
    with np.load(npz_file, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in data]
        if missing:
            raise ValueError(f"{npz_file} missing required A_Audition keys: {', '.join(missing)}")
        joints = np.asarray(data["joints_3d"], dtype=np.float32)
        trans = np.asarray(data["trans_canonical"], dtype=np.float32)
        pose_raw = np.asarray(data["pose_raw"], dtype=np.float32)
        trans_raw = np.asarray(data["trans_raw"], dtype=np.float32)
        fps_array = np.asarray(data["fps"], dtype=np.float32).reshape(-1)
        fps = float(fps_array[0]) if fps_array.size else float("nan")

    _validate_sequence_arrays(npz_file, joints, trans, pose_raw, trans_raw, fps)
    subset_name = str(metadata.get("subset") or npz_file.parent.name)
    subject_id, trial_id = _identifiers_from_metadata(npz_file.stem, subset_name, metadata)
    return AuditionSequence(
        subset_name=subset_name,
        subject_id=subject_id,
        trial_id=trial_id,
        stem=npz_file.stem,
        npz_path=npz_file.resolve(),
        json_path=json_file.resolve(),
        joints_3d=joints,
        trans_canonical=trans,
        pose_raw=pose_raw,
        trans_raw=trans_raw,
        fps=fps,
        metadata=metadata,
    )


def resolve_sequence_paths(input_dir: str | Path, subset: str, subject_id: str, trial_id: str) -> tuple[Path, Path]:
    subset_dir = Path(input_dir) / safe_output_name(subset)
    stem = sequence_output_stem(subset, subject_id, trial_id)
    npz_path = subset_dir / f"{stem}.npz"
    json_path = subset_dir / f"{stem}.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return npz_path.resolve(), json_path.resolve()


def iter_subset_dirs(input_dir: str | Path) -> Iterable[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def iter_sequence_npzs(
    input_dir: str | Path,
    *,
    subset_name: str | None = None,
    subject_filter: str | None = None,
    trial_ids: set[str] | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> Iterable[Path]:
    root = Path(input_dir)
    subset_dirs = [root / safe_output_name(subset_name)] if subset_name is not None else list(iter_subset_dirs(root))
    emitted = 0
    for subset_dir in subset_dirs:
        if not subset_dir.exists():
            continue
        for npz_path in sorted(subset_dir.glob("*.npz")):
            json_path = npz_path.with_suffix(".json")
            if not json_path.exists():
                continue
            sequence = load_audition_sequence(npz_path, json_path)
            if subject_filter is not None and sequence.subject_id != str(subject_filter):
                continue
            if trial_ids is not None and sequence.trial_id not in trial_ids:
                continue
            if trial_contains is not None and trial_contains.lower() not in sequence.trial_id.lower():
                continue
            yield npz_path.resolve()
            emitted += 1
            if max_trials is not None and emitted >= max_trials:
                return

