"""Input loading helpers for C_Representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np

from .config import REQUIRED_B_NPZ_KEYS
from .joints_schema import validate_joints


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


@dataclass(frozen=True)
class CanonicalizedSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    npz_path: Path
    json_path: Path
    pose_raw: np.ndarray
    trans_raw: np.ndarray
    joints_can: np.ndarray
    trans_can: np.ndarray
    R_global: np.ndarray
    R_total: np.ndarray
    fps: float
    metadata: dict[str, object]

    @property
    def num_frames(self) -> int:
        return int(self.joints_can.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps if self.fps > 0.0 else 0.0


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def _metadata_block(metadata: dict[str, object]) -> dict[str, object]:
    block = metadata.get("metadata", {})
    return block if isinstance(block, dict) else {}


def _extract_fps(metadata: dict[str, object]) -> float:
    block = _metadata_block(metadata)
    fps = block.get("fps", metadata.get("fps"))
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("B JSON must provide a finite positive fps in metadata.fps")
    return value


def _infer_identifiers(npz_path: Path, metadata: dict[str, object]) -> tuple[str, str, str]:
    block = _metadata_block(metadata)
    subset_name = str(metadata.get("subset") or block.get("subset") or npz_path.parent.name)
    subject_id = metadata.get("subject_id") or block.get("subject_id")
    trial_id = metadata.get("trial_id") or block.get("trial_id")
    if subject_id is not None and trial_id is not None:
        return subset_name, str(subject_id), str(trial_id)

    parts = npz_path.stem.split("__", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return subset_name, "unknown_subject", npz_path.stem


def _as_rotation_matrix(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"B JSON must provide {name} with shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"B JSON {name} contains non-finite values")
    det = float(np.linalg.det(matrix.astype(np.float64)))
    if not np.isfinite(det) or abs(det) < 1e-6:
        raise ValueError(f"B JSON {name} is not a valid rotation-like matrix; det={det!r}")
    return matrix.astype(np.float32, copy=False)


def _extract_rotation_metadata(metadata: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if "R_global" not in metadata:
        raise ValueError("B JSON must provide top-level R_global for canonical pelvis/root orientation")
    r_global = _as_rotation_matrix(metadata["R_global"], name="R_global")

    block = _metadata_block(metadata)
    audition_metadata = block.get("audition_metadata")
    if not isinstance(audition_metadata, dict):
        raise ValueError("B JSON must provide metadata.audition_metadata.R_total for canonical pelvis/root orientation")
    if "R_total" not in audition_metadata:
        raise ValueError("B JSON must provide metadata.audition_metadata.R_total for canonical pelvis/root orientation")
    r_total = _as_rotation_matrix(audition_metadata["R_total"], name="metadata.audition_metadata.R_total")
    return r_global, r_total


def _validate_npz_contract(npz_path: Path, files: list[str]) -> None:
    expected = set(REQUIRED_B_NPZ_KEYS)
    actual = set(files)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        suffix = f": {'; '.join(details)}" if details else ""
        raise ValueError(f"{npz_path} must use the new B NPZ contract {list(REQUIRED_B_NPZ_KEYS)}{suffix}")


def load_canonicalized_sequence(npz_path: str | Path, json_path: str | Path | None = None) -> CanonicalizedSequence:
    npz_file = Path(npz_path)
    json_file = Path(json_path) if json_path is not None else npz_file.with_suffix(".json")
    if not npz_file.exists():
        raise FileNotFoundError(npz_file)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    metadata = load_json(json_file)
    fps = _extract_fps(metadata)
    r_global, r_total = _extract_rotation_metadata(metadata)
    with np.load(npz_file, allow_pickle=False) as data:
        _validate_npz_contract(npz_file, list(data.files))
        pose_raw = np.asarray(data["pose_raw"], dtype=np.float32)
        trans_raw = np.asarray(data["trans_raw"], dtype=np.float32)
        joints_can = validate_joints(data["joints_can"], name="joints_can")
        trans_can = np.asarray(data["trans_can"], dtype=np.float32)

    frame_count = int(joints_can.shape[0])
    if pose_raw.shape != (frame_count, 72):
        raise ValueError(f"{npz_file} pose_raw must have shape {(frame_count, 72)}, got {pose_raw.shape}")
    if trans_raw.shape != (frame_count, 3):
        raise ValueError(f"{npz_file} trans_raw must have shape {(frame_count, 3)}, got {trans_raw.shape}")
    if trans_can.shape != (frame_count, 3):
        raise ValueError(f"{npz_file} trans_can must have shape {(frame_count, 3)}, got {trans_can.shape}")
    for name, array in {"pose_raw": pose_raw, "trans_raw": trans_raw, "trans_can": trans_can}.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{npz_file} {name} contains non-finite values")

    subset_name, subject_id, trial_id = _infer_identifiers(npz_file, metadata)
    return CanonicalizedSequence(
        subset_name=subset_name,
        subject_id=subject_id,
        trial_id=trial_id,
        stem=npz_file.stem,
        npz_path=npz_file.resolve(),
        json_path=json_file.resolve(),
        pose_raw=pose_raw.astype(np.float32, copy=False),
        trans_raw=trans_raw.astype(np.float32, copy=False),
        joints_can=joints_can.astype(np.float32, copy=False),
        trans_can=trans_can.astype(np.float32, copy=False),
        R_global=r_global,
        R_total=r_total,
        fps=float(fps),
        metadata=metadata,
    )


def ensure_subset_output_dir(output_dir: str | Path, subset_name: str) -> Path:
    subset_dir = Path(output_dir) / safe_output_name(subset_name)
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


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
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> Iterable[Path]:
    root = Path(input_dir)
    subset_dirs = [root / safe_output_name(subset_name)] if subset_name is not None else list(iter_subset_dirs(root))
    trial_filter = str(trial_contains).lower() if trial_contains else None
    emitted = 0
    for subset_dir in subset_dirs:
        if not subset_dir.exists():
            continue
        for npz_path in sorted(subset_dir.glob("*.npz")):
            if trial_filter is not None and trial_filter not in npz_path.stem.lower():
                continue
            if not npz_path.with_suffix(".json").exists():
                continue
            yield npz_path.resolve()
            emitted += 1
            if max_trials is not None and emitted >= max_trials:
                return
