"""Input loading helpers for D_Segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


REQUIRED_NPZ_KEYS = (
    "joints_can",
    "root_pos_m",
    "root_speed_xy_mps",
    "heading_deg",
    "heading_unwrapped_deg",
    "yaw_rate_deg_s",
    "pelvis_height_m",
    "valid_frame_mask",
    "representation_quality_score",
    "left_foot_pos_m",
    "right_foot_pos_m",
    "left_foot_speed_mps",
    "right_foot_speed_mps",
    "left_foot_contact",
    "right_foot_contact",
    "left_heel_strike",
    "right_heel_strike",
    "left_toe_off",
    "right_toe_off",
    "fps",
)


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


@dataclass(frozen=True)
class RepresentationSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    npz_path: Path
    json_path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    fps: float
    num_frames: int


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def _as_fps(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.shape not in ((), (1,)):
        raise ValueError(f"fps must be scalar-like, got shape {array.shape}")
    fps = float(array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    return fps


def _infer_identity(npz_path: Path, metadata: dict[str, object]) -> tuple[str, str, str]:
    basic_info = metadata.get("basic_info", {})
    if isinstance(basic_info, dict):
        subset_name = str(basic_info.get("subset") or npz_path.parent.name)
        subject_id = str(basic_info.get("subject_id") or "")
        trial_id = str(basic_info.get("trial_id") or "")
        if subject_id and trial_id:
            return subset_name, subject_id, trial_id

    meta = metadata.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    subset_name = str(meta.get("subset") or npz_path.parent.name)
    subject_id = str(meta.get("subject_id") or "")
    trial_id = str(meta.get("trial_id") or "")
    if subject_id and trial_id:
        return subset_name, subject_id, trial_id

    parts = npz_path.stem.split("__", 2)
    if len(parts) != 3:
        raise ValueError(f"could not infer subset/subject/trial from {npz_path.name}")
    return parts[0], parts[1], parts[2]


def validate_npz_payload(payload: np.lib.npyio.NpzFile) -> None:
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in payload.files]
    if missing:
        raise ValueError(f"unsupported C_Representation contract; missing strict v2 NPZ keys: {missing}")

    joints = np.asarray(payload["joints_can"])
    root = np.asarray(payload["root_pos_m"])
    left_foot = np.asarray(payload["left_foot_pos_m"])
    right_foot = np.asarray(payload["right_foot_pos_m"])
    heading = np.asarray(payload["heading_deg"])
    heading_unwrapped = np.asarray(payload["heading_unwrapped_deg"])
    yaw_rate = np.asarray(payload["yaw_rate_deg_s"])
    pelvis_height = np.asarray(payload["pelvis_height_m"])
    root_speed = np.asarray(payload["root_speed_xy_mps"])
    valid_frame_mask = np.asarray(payload["valid_frame_mask"])
    representation_quality_score = np.asarray(payload["representation_quality_score"])
    left_foot_speed = np.asarray(payload["left_foot_speed_mps"])
    right_foot_speed = np.asarray(payload["right_foot_speed_mps"])
    left_contact = np.asarray(payload["left_foot_contact"])
    right_contact = np.asarray(payload["right_foot_contact"])
    left_heel_strike = np.asarray(payload["left_heel_strike"])
    right_heel_strike = np.asarray(payload["right_heel_strike"])
    left_toe_off = np.asarray(payload["left_toe_off"])
    right_toe_off = np.asarray(payload["right_toe_off"])

    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"joints_can must have shape (T, 24, 3), got {joints.shape}")
    frame_count = int(joints.shape[0])
    if frame_count < 2:
        raise ValueError("sequence must contain at least 2 frames")
    for key, value in {
        "root_pos_m": root,
        "left_foot_pos_m": left_foot,
        "right_foot_pos_m": right_foot,
    }.items():
        if value.shape != (frame_count, 3):
            raise ValueError(f"{key} must have shape {(frame_count, 3)}, got {value.shape}")
    for key, value in {
        "heading_deg": heading,
        "heading_unwrapped_deg": heading_unwrapped,
        "yaw_rate_deg_s": yaw_rate,
        "pelvis_height_m": pelvis_height,
        "root_speed_xy_mps": root_speed,
        "valid_frame_mask": valid_frame_mask,
        "representation_quality_score": representation_quality_score,
        "left_foot_speed_mps": left_foot_speed,
        "right_foot_speed_mps": right_foot_speed,
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
        "left_heel_strike": left_heel_strike,
        "right_heel_strike": right_heel_strike,
        "left_toe_off": left_toe_off,
        "right_toe_off": right_toe_off,
    }.items():
        if value.shape != (frame_count,):
            raise ValueError(f"{key} must have shape {(frame_count,)}, got {value.shape}")
    _as_fps(payload["fps"])


def load_representation_sequence(npz_path: str | Path, json_path: str | Path | None = None) -> RepresentationSequence:
    npz = Path(npz_path)
    json_file = Path(json_path) if json_path is not None else npz.with_suffix(".json")
    if not npz.exists():
        raise FileNotFoundError(npz)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    with np.load(npz, allow_pickle=False) as payload:
        validate_npz_payload(payload)
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
        fps = _as_fps(payload["fps"])
        num_frames = int(np.asarray(payload["joints_can"]).shape[0])

    metadata = load_json(json_file)
    if metadata.get("module") != "C_Representation":
        raise ValueError(f"{json_file} is not a C_Representation JSON artifact")
    if str(metadata.get("version", "")).split(".")[0] != "2":
        raise ValueError(f"{json_file} is not a strict C_Representation v2 artifact")
    subset_name, subject_id, trial_id = _infer_identity(npz, metadata)
    return RepresentationSequence(
        subset_name=subset_name,
        subject_id=subject_id,
        trial_id=trial_id,
        stem=npz.stem,
        npz_path=npz.resolve(),
        json_path=json_file.resolve(),
        arrays=arrays,
        metadata=metadata,
        fps=fps,
        num_frames=num_frames,
    )


def ensure_subset_output_dir(output_dir: str | Path, subset_name: str) -> Path:
    subset_dir = Path(output_dir) / safe_output_name(subset_name)
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


def resolve_sequence_paths(
    input_dir: str | Path,
    subset_name: str,
    subject_id: str,
    trial_id: str,
) -> tuple[Path, Path]:
    subset_dir = Path(input_dir) / safe_output_name(subset_name)
    stem = sequence_output_stem(subset_name, subject_id, trial_id)
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
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_sequence_npzs(
    input_dir: str | Path,
    *,
    subset_name: str | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> list[Path]:
    root = Path(input_dir)
    subset_dirs = [root / safe_output_name(subset_name)] if subset_name is not None else list(iter_subset_dirs(root))
    trial_filter = str(trial_contains).lower() if trial_contains else None
    paths: list[Path] = []
    for subset_dir in subset_dirs:
        if not subset_dir.exists():
            continue
        for npz_path in sorted(subset_dir.glob("*.npz")):
            if trial_filter is not None and trial_filter not in npz_path.stem.lower():
                continue
            paths.append(npz_path.resolve())
            if max_trials is not None and len(paths) >= max_trials:
                return paths
    return paths
