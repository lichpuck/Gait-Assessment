"""Input loading helpers for CARE-PD G_Animation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from scripts.C_Representation.config import STRICT_NPZ_FIELDS
from scripts.C_Representation.io_utils import CanonicalizedSequence, load_canonicalized_sequence
from scripts.D_Segmentation.config import PRIMARY_LABELS

from .config import AnimationConfig


REQUIRED_C_KEYS = (
    "fps",
    "time_s",
    "frame_index",
    "joints_can",
    "root_pos_m",
    "root_speed_xy_mps",
    "root_acceleration_mps2",
    "left_foot_speed_mps",
    "right_foot_speed_mps",
    "left_gait_phase",
    "right_gait_phase",
    "trunk_lean_angle_deg",
    "pelvis_roll_deg",
    "valid_frame_mask",
    "representation_quality_score",
)


def safe_output_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def sequence_output_stem(subset_name: str, subject_id: str, trial_id: str) -> str:
    return "__".join([safe_output_name(subset_name), safe_output_name(subject_id), safe_output_name(trial_id)])


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


@dataclass(frozen=True)
class AnimationSequenceInputs:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    b_sequence: CanonicalizedSequence
    c_sequence: "AnimationRepresentationSequence"
    d_json_path: Path
    f_json_path: Path
    d_payload: dict[str, object]
    f_payload: dict[str, object]
    segments: tuple[dict[str, object], ...]
    primary_label_by_frame: np.ndarray
    summary_text_zh: str

    @property
    def fps(self) -> float:
        return float(self.c_sequence.fps)

    @property
    def num_frames(self) -> int:
        return int(self.c_sequence.num_frames)

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0.0:
            return 0.0
        return float(self.num_frames / self.fps)


@dataclass(frozen=True)
class AnimationRepresentationSequence:
    subset_name: str
    subject_id: str
    trial_id: str
    stem: str
    npz_path: Path
    json_path: Path
    arrays: dict[str, np.ndarray]
    fps: float
    num_frames: int
    metadata: dict[str, object]


def resolve_sequence_paths(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: AnimationConfig | None = None,
) -> dict[str, Path]:
    config = config or AnimationConfig()
    subset_dir_name = safe_output_name(subset)
    stem = sequence_output_stem(subset, subject_id, trial_id)
    paths = {
        "b_npz": Path(config.input_b_dir) / subset_dir_name / f"{stem}.npz",
        "b_json": Path(config.input_b_dir) / subset_dir_name / f"{stem}.json",
        "c_npz": Path(config.input_c_dir) / subset_dir_name / f"{stem}.npz",
        "c_json": Path(config.input_c_dir) / subset_dir_name / f"{stem}.json",
        "d_json": Path(config.input_d_dir) / subset_dir_name / f"{stem}.json",
        "f_json": Path(config.input_f_dir) / subset_dir_name / f"{stem}.json",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing G_Animation input {name}: {path}")
    return {name: path.resolve() for name, path in paths.items()}


def _load_c_sequence(npz_path: Path, json_path: Path) -> AnimationRepresentationSequence:
    metadata = load_json(json_path)
    if metadata.get("module") != "C_Representation":
        raise ValueError(f"{json_path} is not a C_Representation JSON artifact")
    if str(metadata.get("version", "")).split(".")[0] != "2":
        raise ValueError(f"{json_path} is not a strict C_Representation v2 artifact")

    with np.load(npz_path, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_C_KEYS if key not in payload.files]
        if missing:
            raise ValueError(f"{npz_path} missing required C_Representation keys for G_Animation: {missing}")
        unexpected = [key for key in payload.files if key not in STRICT_NPZ_FIELDS]
        if unexpected:
            raise ValueError(f"{npz_path} contains unexpected non-strict C_Representation keys: {unexpected}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    frame_count = int(np.asarray(arrays["joints_can"]).shape[0])
    if frame_count < 2:
        raise ValueError(f"{npz_path} must contain at least 2 frames")

    expected_vector = (frame_count,)
    expected_xyz = (frame_count, 3)
    if np.asarray(arrays["joints_can"]).shape != (frame_count, 24, 3):
        raise ValueError(f"{npz_path} joints_can must have shape {(frame_count, 24, 3)}")
    for key in (
        "time_s",
        "frame_index",
        "root_speed_xy_mps",
        "left_foot_speed_mps",
        "right_foot_speed_mps",
        "left_gait_phase",
        "right_gait_phase",
        "trunk_lean_angle_deg",
        "pelvis_roll_deg",
        "valid_frame_mask",
        "representation_quality_score",
    ):
        if np.asarray(arrays[key]).shape != expected_vector:
            raise ValueError(f"{npz_path} {key} must have shape {expected_vector}, got {np.asarray(arrays[key]).shape}")
    for key in ("root_pos_m", "root_acceleration_mps2"):
        if np.asarray(arrays[key]).shape != expected_xyz:
            raise ValueError(f"{npz_path} {key} must have shape {expected_xyz}, got {np.asarray(arrays[key]).shape}")

    fps_array = np.asarray(arrays["fps"], dtype=np.float32).reshape(-1)
    fps = float(fps_array[0]) if fps_array.size else float("nan")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{npz_path} fps must be finite and positive, got {fps!r}")

    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "b":
            continue
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{npz_path} {key} contains non-finite values")

    sequence = metadata.get("sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError(f"{json_path} must contain a sequence object")
    subset_name = str(sequence.get("subset") or npz_path.parent.name)
    subject_id = str(sequence.get("subject_id") or "")
    trial_id = str(sequence.get("trial_id") or "")
    if not subject_id or not trial_id:
        parts = npz_path.stem.split("__", 2)
        if len(parts) == 3:
            subset_name, subject_id, trial_id = parts[0], parts[1], parts[2]
        else:
            raise ValueError(f"Could not infer subset/subject/trial from {npz_path.name}")

    return AnimationRepresentationSequence(
        subset_name=subset_name,
        subject_id=subject_id,
        trial_id=trial_id,
        stem=npz_path.stem,
        npz_path=npz_path.resolve(),
        json_path=json_path.resolve(),
        arrays=arrays,
        fps=fps,
        num_frames=frame_count,
        metadata=metadata,
    )


def _assert_identity(
    *,
    stage_name: str,
    subset_name: str,
    subject_id: str,
    trial_id: str,
    payload: dict[str, object],
) -> None:
    sequence = payload.get("sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError(f"{stage_name} JSON must contain a sequence object")

    actual_subset = str(sequence.get("subset") or subset_name)
    actual_subject = str(sequence.get("subject_id") or subject_id)
    actual_trial = str(sequence.get("trial_id") or trial_id)
    if actual_subset != subset_name or actual_subject != subject_id or actual_trial != trial_id:
        raise ValueError(
            f"{stage_name} JSON identity mismatch: expected {subset_name}/{subject_id}/{trial_id}, "
            f"got {actual_subset}/{actual_subject}/{actual_trial}"
        )


def _as_frame_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer frame index")
    try:
        frame_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer frame index") from exc
    return frame_index


def _normalize_segments(d_payload: dict[str, object], frame_count: int) -> tuple[tuple[dict[str, object], ...], np.ndarray]:
    raw_segments = d_payload.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("D_Segmentation JSON must contain a non-empty segments list")

    label_by_frame = np.empty(frame_count, dtype=object)
    label_by_frame[:] = ""
    cursor = 0
    normalized_segments: list[dict[str, object]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"D_Segmentation segment #{index + 1} must be an object")
        label = str(raw_segment.get("label") or "")
        if label not in PRIMARY_LABELS:
            raise ValueError(f"Unsupported D_Segmentation label for G_Animation: {label!r}")
        start_frame = _as_frame_index(raw_segment.get("start_frame"), name="start_frame")
        end_frame = _as_frame_index(raw_segment.get("end_frame"), name="end_frame")
        if start_frame != cursor:
            raise ValueError(
                f"D_Segmentation segments must be contiguous from frame 0; expected start {cursor}, got {start_frame}"
            )
        if start_frame < 0 or end_frame < start_frame or end_frame >= frame_count:
            raise ValueError(
                f"D_Segmentation segment #{index + 1} has invalid frame span {start_frame}..{end_frame} "
                f"for frame_count={frame_count}"
            )
        label_by_frame[start_frame : end_frame + 1] = label
        cursor = end_frame + 1
        normalized_segments.append(dict(raw_segment))

    if cursor != frame_count:
        raise ValueError(
            f"D_Segmentation segments do not cover the full sequence; covered 0..{cursor - 1}, frame_count={frame_count}"
        )
    if np.any(label_by_frame == ""):
        raise ValueError("D_Segmentation segments left unlabeled frames")
    return tuple(normalized_segments), label_by_frame


def _validate_b_c_alignment(b_sequence: CanonicalizedSequence, c_sequence: AnimationRepresentationSequence) -> None:
    expected_identity = (b_sequence.subset_name, b_sequence.subject_id, b_sequence.trial_id)
    actual_identity = (c_sequence.subset_name, c_sequence.subject_id, c_sequence.trial_id)
    if actual_identity != expected_identity:
        raise ValueError(f"B/C identity mismatch: {expected_identity} vs {actual_identity}")
    if c_sequence.num_frames != b_sequence.num_frames:
        raise ValueError(
            f"B/C frame-count mismatch for {b_sequence.stem}: {b_sequence.num_frames} vs {c_sequence.num_frames}"
        )
    if not np.isclose(float(c_sequence.fps), float(b_sequence.fps), atol=1e-6):
        raise ValueError(f"B/C fps mismatch for {b_sequence.stem}: {b_sequence.fps} vs {c_sequence.fps}")


def _extract_summary_text(f_payload: dict[str, object]) -> str:
    description = f_payload.get("description", {})
    if not isinstance(description, dict):
        return ""
    value = description.get("text_summary_zh", "")
    if value is None:
        return ""
    return str(value).strip()


def load_animation_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: AnimationConfig | None = None,
) -> AnimationSequenceInputs:
    config = config or AnimationConfig()
    paths = resolve_sequence_paths(subset, subject_id, trial_id, config=config)

    b_sequence = load_canonicalized_sequence(paths["b_npz"], paths["b_json"])
    c_sequence = _load_c_sequence(paths["c_npz"], paths["c_json"])
    _validate_b_c_alignment(b_sequence, c_sequence)

    d_payload = load_json(paths["d_json"])
    f_payload = load_json(paths["f_json"])
    _assert_identity(
        stage_name="D_Segmentation",
        subset_name=b_sequence.subset_name,
        subject_id=b_sequence.subject_id,
        trial_id=b_sequence.trial_id,
        payload=d_payload,
    )
    _assert_identity(
        stage_name="F_Description",
        subset_name=b_sequence.subset_name,
        subject_id=b_sequence.subject_id,
        trial_id=b_sequence.trial_id,
        payload=f_payload,
    )

    d_sequence = d_payload.get("sequence", {})
    if not isinstance(d_sequence, dict):
        raise ValueError("D_Segmentation JSON must contain a sequence object")
    d_num_frames = _as_frame_index(d_sequence.get("num_frames"), name="sequence.num_frames")
    if d_num_frames != c_sequence.num_frames:
        raise ValueError(f"D/C frame-count mismatch for {b_sequence.stem}: {d_num_frames} vs {c_sequence.num_frames}")

    segments, primary_label_by_frame = _normalize_segments(d_payload, c_sequence.num_frames)
    summary_text_zh = _extract_summary_text(f_payload)
    return AnimationSequenceInputs(
        subset_name=b_sequence.subset_name,
        subject_id=b_sequence.subject_id,
        trial_id=b_sequence.trial_id,
        stem=b_sequence.stem,
        b_sequence=b_sequence,
        c_sequence=c_sequence,
        d_json_path=paths["d_json"],
        f_json_path=paths["f_json"],
        d_payload=d_payload,
        f_payload=f_payload,
        segments=segments,
        primary_label_by_frame=primary_label_by_frame,
        summary_text_zh=summary_text_zh,
    )