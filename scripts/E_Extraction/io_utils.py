"""Input loading helpers for E_Extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


TARGET_LABELS = ("walk", "turn")

REQUIRED_C_NPZ_KEYS = (
    "fps",
    "time_s",
    "valid_frame_mask",
    "representation_quality_score",
    "joints_can",
    "root_pos_m",
    "root_speed_xy_mps",
    "heading_unwrapped_deg",
    "yaw_rate_deg_s",
    "left_foot_pos_m",
    "right_foot_pos_m",
    "left_foot_contact",
    "right_foot_contact",
    "left_heel_strike",
    "right_heel_strike",
    "left_toe_off",
    "right_toe_off",
    "gait_phase_global",
    "trunk_forward_flexion_deg",
    "trunk_lateral_lean_deg",
    "trunk_yaw_deg",
    "pelvis_yaw_deg",
)


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


@dataclass(frozen=True)
class MotionSegment:
    segment_id: str
    label: str
    start_frame: int
    end_frame: int
    used_for_extraction: bool
    quality_status: str
    quality_valid_frame_ratio: float
    quality_representation_score_mean: float
    quality_hesitation_overlap_ratio: float
    quality_reasons: tuple[str, ...]
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    source_rule: str
    confidence: float
    hesitation_overlap_frames: int
    auxiliary_overlap_labels: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    c_npz_path: Path
    c_json_path: Path
    d_json_path: Path
    source_b_npz_path: Path | None
    arrays: dict[str, np.ndarray]
    c_metadata: dict[str, Any]
    d_metadata: dict[str, Any]
    segments: tuple[MotionSegment, ...]
    fps: float
    num_frames: int
    pose_raw: np.ndarray | None
    r_global: np.ndarray | None
    r_total: np.ndarray | None


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def _as_scalar_float(value: np.ndarray | float | int) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.shape not in ((), (1,)):
        raise ValueError(f"scalar-like value expected, got shape {array.shape}")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"expected finite scalar, got {scalar!r}")
    return scalar


def _recursive_find_key(payload: Any, key: str) -> Any | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _recursive_find_key(value, key)
            if found is not None:
                return found
    return None


def _matrix3_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, 3):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.astype(np.float32, copy=False)


def _infer_identity(c_metadata: dict[str, Any], d_metadata: dict[str, Any], stem: str, subset_name: str) -> tuple[str, str, str]:
    for container_key in ("sequence", "basic_info", "metadata"):
        container = d_metadata.get(container_key, {})
        if isinstance(container, dict):
            subset = str(container.get("subset") or subset_name)
            subject_id = str(container.get("subject_id") or "")
            trial_id = str(container.get("trial_id") or "")
            if subject_id and trial_id:
                return subset, subject_id, trial_id
    for container_key in ("basic_info", "metadata"):
        container = c_metadata.get(container_key, {})
        if isinstance(container, dict):
            subset = str(container.get("subset") or subset_name)
            subject_id = str(container.get("subject_id") or "")
            trial_id = str(container.get("trial_id") or "")
            if subject_id and trial_id:
                return subset, subject_id, trial_id
    parts = stem.split("__", 2)
    if len(parts) != 3:
        raise ValueError(f"could not infer subset/subject/trial from {stem!r}")
    return parts[0], parts[1], parts[2]


def _validate_c_npz_payload(payload: np.lib.npyio.NpzFile) -> tuple[float, int]:
    missing = [key for key in REQUIRED_C_NPZ_KEYS if key not in payload.files]
    if missing:
        raise ValueError(f"unsupported C_Representation contract; missing keys: {missing}")

    joints = np.asarray(payload["joints_can"], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"joints_can must have shape (T, 24, 3), got {joints.shape}")
    frame_count = int(joints.shape[0])
    if frame_count < 2:
        raise ValueError("sequence must contain at least 2 frames")

    vector_keys = (
        "time_s",
        "valid_frame_mask",
        "representation_quality_score",
        "root_speed_xy_mps",
        "heading_unwrapped_deg",
        "yaw_rate_deg_s",
        "left_foot_contact",
        "right_foot_contact",
        "left_heel_strike",
        "right_heel_strike",
        "left_toe_off",
        "right_toe_off",
        "gait_phase_global",
        "trunk_forward_flexion_deg",
        "trunk_lateral_lean_deg",
        "trunk_yaw_deg",
        "pelvis_yaw_deg",
    )
    matrix_keys = (
        "root_pos_m",
        "left_foot_pos_m",
        "right_foot_pos_m",
    )
    for key in vector_keys:
        value = np.asarray(payload[key])
        if value.shape != (frame_count,):
            raise ValueError(f"{key} must have shape {(frame_count,)}, got {value.shape}")
    for key in matrix_keys:
        value = np.asarray(payload[key])
        if value.shape != (frame_count, 3):
            raise ValueError(f"{key} must have shape {(frame_count, 3)}, got {value.shape}")
    fps = _as_scalar_float(payload["fps"])
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return fps, frame_count


def _load_motion_segment(payload: dict[str, Any]) -> MotionSegment:
    quality = payload.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
    reasons = quality.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    auxiliary = payload.get("auxiliary_overlap_labels", [])
    if auxiliary is None:
        auxiliary = []
    if not isinstance(auxiliary, (list, tuple)):
        auxiliary = [str(auxiliary)]
    return MotionSegment(
        segment_id=str(payload.get("segment_id") or ""),
        label=str(payload.get("label") or ""),
        start_frame=int(payload.get("start_frame", 0)),
        end_frame=int(payload.get("end_frame", -1)),
        used_for_extraction=bool(payload.get("used_for_extraction", False)),
        quality_status=str(quality.get("status") or ""),
        quality_valid_frame_ratio=float(quality.get("valid_frame_ratio", 0.0) or 0.0),
        quality_representation_score_mean=float(quality.get("representation_quality_score_mean", 0.0) or 0.0),
        quality_hesitation_overlap_ratio=float(quality.get("hesitation_overlap_ratio", 0.0) or 0.0),
        quality_reasons=tuple(str(item) for item in reasons),
        start_time_sec=float(payload.get("start_time_sec", 0.0) or 0.0),
        end_time_sec=float(payload.get("end_time_sec", 0.0) or 0.0),
        duration_sec=float(payload.get("duration_sec", 0.0) or 0.0),
        source_rule=str(payload.get("source_rule") or ""),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        hesitation_overlap_frames=int(payload.get("hesitation_overlap_frames", 0) or 0),
        auxiliary_overlap_labels=tuple(str(item) for item in auxiliary),
    )


def _load_optional_pose_raw(path: Path | None, frame_count: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "pose_raw" not in payload.files:
            return None
        pose_raw = np.asarray(payload["pose_raw"], dtype=np.float32)
    if pose_raw.shape != (frame_count, 72):
        return None
    if not np.all(np.isfinite(pose_raw)):
        return None
    return pose_raw.astype(np.float32, copy=False)


def load_extraction_sequence(
    c_npz_path: str | Path,
    c_json_path: str | Path | None = None,
    d_json_path: str | Path | None = None,
) -> ExtractionSequence:
    c_npz = Path(c_npz_path)
    c_json = Path(c_json_path) if c_json_path is not None else c_npz.with_suffix(".json")
    d_json = Path(d_json_path) if d_json_path is not None else c_npz.with_suffix(".json")

    if not c_npz.exists():
        raise FileNotFoundError(c_npz)
    if not c_json.exists():
        raise FileNotFoundError(c_json)
    if not d_json.exists():
        raise FileNotFoundError(d_json)

    with np.load(c_npz, allow_pickle=False) as payload:
        fps, frame_count = _validate_c_npz_payload(payload)
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    c_metadata = load_json(c_json)
    d_metadata = load_json(d_json)
    if c_metadata.get("module") != "C_Representation":
        raise ValueError(f"{c_json} is not a C_Representation JSON artifact")
    if d_metadata.get("module") != "D_Segmentation":
        raise ValueError(f"{d_json} is not a D_Segmentation JSON artifact")

    source_files = c_metadata.get("source_files", {})
    if not isinstance(source_files, dict):
        source_files = {}
    upstream_b_npz_raw = source_files.get("upstream_b_npz")
    source_b_npz_path = Path(str(upstream_b_npz_raw)).resolve() if upstream_b_npz_raw else None

    r_global = _matrix3_or_none(_recursive_find_key(c_metadata, "R_global"))
    r_total = _matrix3_or_none(_recursive_find_key(c_metadata, "R_total"))
    pose_raw = _load_optional_pose_raw(source_b_npz_path, frame_count)

    segments_payload = d_metadata.get("segments", [])
    if not isinstance(segments_payload, list):
        raise ValueError(f"{d_json} has invalid segments payload")
    segments = tuple(_load_motion_segment(item) for item in segments_payload if isinstance(item, dict))

    subset_name, subject_id, trial_id = _infer_identity(c_metadata, d_metadata, c_npz.stem, c_npz.parent.name)
    return ExtractionSequence(
        subset_name=subset_name,
        subject_id=subject_id,
        trial_id=trial_id,
        stem=c_npz.stem,
        c_npz_path=c_npz.resolve(),
        c_json_path=c_json.resolve(),
        d_json_path=d_json.resolve(),
        source_b_npz_path=source_b_npz_path,
        arrays=arrays,
        c_metadata=c_metadata,
        d_metadata=d_metadata,
        segments=segments,
        fps=float(fps),
        num_frames=frame_count,
        pose_raw=pose_raw,
        r_global=r_global,
        r_total=r_total,
    )


def ensure_subset_output_dir(output_dir: str | Path, subset_name: str) -> Path:
    subset_dir = Path(output_dir) / safe_output_name(subset_name)
    subset_dir.mkdir(parents=True, exist_ok=True)
    return subset_dir


def resolve_sequence_paths(
    input_c_dir: str | Path,
    input_d_dir: str | Path,
    subset_name: str,
    subject_id: str,
    trial_id: str,
) -> tuple[Path, Path, Path]:
    stem = sequence_output_stem(subset_name, subject_id, trial_id)
    c_subset_dir = Path(input_c_dir) / safe_output_name(subset_name)
    d_subset_dir = Path(input_d_dir) / safe_output_name(subset_name)
    c_npz_path = c_subset_dir / f"{stem}.npz"
    c_json_path = c_subset_dir / f"{stem}.json"
    d_json_path = d_subset_dir / f"{stem}.json"
    for path in (c_npz_path, c_json_path, d_json_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return c_npz_path.resolve(), c_json_path.resolve(), d_json_path.resolve()


def iter_subset_dirs(input_dir: str | Path) -> Iterable[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_sequence_jsons(
    input_d_dir: str | Path,
    *,
    subset_name: str | None = None,
    trial_contains: str | None = None,
    max_trials: int | None = None,
) -> list[Path]:
    root = Path(input_d_dir)
    subset_dirs = [root / safe_output_name(subset_name)] if subset_name is not None else list(iter_subset_dirs(root))
    trial_filter = str(trial_contains).lower() if trial_contains else None
    paths: list[Path] = []
    for subset_dir in subset_dirs:
        if not subset_dir.exists():
            continue
        for json_path in sorted(subset_dir.glob("*.json")):
            if trial_filter is not None and trial_filter not in json_path.stem.lower():
                continue
            paths.append(json_path.resolve())
            if max_trials is not None and len(paths) >= max_trials:
                return paths
    return paths