#!/usr/bin/env python3
"""Render first-frame canonical skeleton PNGs from B outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_DIR = ROOT / "outputs" / "B_Canonicalization"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Visualization"
FIRST_FRAME_DIRNAME = "first_frame"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.B_Canonicalization.joints_schema import JOINT_INDEX, JOINT_NAMES, validate_joints  # noqa: E402


DIAGNOSTIC_DPI = 180

SKELETON_EDGES = (
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("neck", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("neck", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render first-frame canonical 24-joint skeleton PNGs from outputs/B_Canonicalization."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing B_Canonicalization subset folders with .json/.npz pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory. Files are written under {FIRST_FRAME_DIRNAME}/<subset>/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DIAGNOSTIC_DPI,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        help="Optional subset name filter. Repeatable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing when the target PNG already exists.",
    )
    parser.add_argument(
        "--max-sequences-per-subset",
        type=int,
        help="Optional smoke-test limit for each subset.",
    )
    return parser.parse_args()


def safe_filename_part(value: object) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text or "unknown"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON root object, got {type(payload).__name__}")
    return payload


def nested_get(payload: dict[str, object], *keys: str) -> object | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def resolve_npz_path(json_path: Path, payload: dict[str, object]) -> Path:
    output_npz = nested_get(payload, "outputs", "npz")
    if isinstance(output_npz, str) and output_npz.strip():
        candidate = Path(output_npz)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return candidate
    return json_path.with_suffix(".npz")


def require_array(
    arrays: np.lib.npyio.NpzFile,
    name: str,
    *,
    ndim: int | None = None,
    shape_tail: tuple[int, ...] | None = None,
) -> np.ndarray:
    if name not in arrays.files:
        raise ValueError(f"missing {name}")
    array = np.asarray(arrays[name], dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    if shape_tail is not None and array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must end with shape {shape_tail}, got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} has zero frames")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def first_frame_body_facing(joints_can: np.ndarray) -> tuple[np.ndarray, float]:
    joints = validate_joints(joints_can, name="joints_can")
    pelvis = joints[0, JOINT_INDEX["pelvis"]]
    neck = joints[0, JOINT_INDEX["neck"]]
    left_hip = joints[0, JOINT_INDEX["left_hip"]]
    right_hip = joints[0, JOINT_INDEX["right_hip"]]
    left_shoulder = joints[0, JOINT_INDEX["left_shoulder"]]
    right_shoulder = joints[0, JOINT_INDEX["right_shoulder"]]

    body_left = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    torso_up = neck - pelvis
    facing = np.cross(body_left, torso_up)
    facing[2] = 0.0
    norm = float(np.linalg.norm(facing[:2]))
    if norm <= 1e-8:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64), float("nan")
    facing_unit = facing / norm
    angle_deg = float(np.degrees(np.arccos(np.clip(facing_unit[0], -1.0, 1.0))))
    return facing_unit.astype(np.float64), angle_deg


def _edge_index_pairs() -> tuple[tuple[int, int], ...]:
    return tuple((JOINT_INDEX[start], JOINT_INDEX[end]) for start, end in SKELETON_EDGES)


def _set_equal_limits(axis, horizontal_values: np.ndarray, vertical_values: np.ndarray) -> None:
    xmin = float(np.min(horizontal_values))
    xmax = float(np.max(horizontal_values))
    ymin = float(np.min(vertical_values))
    ymax = float(np.max(vertical_values))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    radius = 0.55 * span
    axis.set_xlim(cx - radius, cx + radius)
    axis.set_ylim(cy - radius, cy + radius)


def _plot_projection(axis, joints_first: np.ndarray, edge_pairs: tuple[tuple[int, int], ...], ix: int, iy: int, xlabel: str, ylabel: str) -> None:
    xvals = joints_first[:, ix]
    yvals = joints_first[:, iy]
    for start, end in edge_pairs:
        axis.plot(
            [xvals[start], xvals[end]],
            [yvals[start], yvals[end]],
            color="#1f2937",
            linewidth=1.15,
            alpha=0.9,
        )
    axis.scatter(xvals, yvals, color="#2563eb", s=14, zorder=3)
    pelvis = JOINT_INDEX["pelvis"]
    axis.scatter([xvals[pelvis]], [yvals[pelvis]], color="#dc2626", s=28, zorder=4)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="box")
    _set_equal_limits(axis, xvals, yvals)


def _render_first_frame_png(
    *,
    joints_first: np.ndarray,
    root_translation_first: np.ndarray,
    facing_vector: np.ndarray,
    facing_angle_deg: float,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_pairs = _edge_index_pairs()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    fig.suptitle(title)

    _plot_projection(axes[0], joints_first, edge_pairs, 0, 1, "X (m)", "Y (m)")
    axes[0].set_title("Top View (XY)")

    pelvis = joints_first[JOINT_INDEX["pelvis"]]
    if np.all(np.isfinite(facing_vector[:2])):
        xy_extent = max(float(np.ptp(joints_first[:, 0])), float(np.ptp(joints_first[:, 1])), 0.5)
        arrow_len = 0.25 * xy_extent
        axes[0].arrow(
            pelvis[0],
            pelvis[1],
            float(facing_vector[0] * arrow_len),
            float(facing_vector[1] * arrow_len),
            width=0.005 * xy_extent,
            color="#059669",
            length_includes_head=True,
            zorder=5,
        )

    _plot_projection(axes[1], joints_first, edge_pairs, 0, 2, "X (m)", "Z (m)")
    axes[1].set_title("Side View (XZ)")

    _plot_projection(axes[2], joints_first, edge_pairs, 1, 2, "Y (m)", "Z (m)")
    axes[2].set_title("Front View (YZ)")

    text_lines = [
        f"pelvis xyz: ({pelvis[0]:.3f}, {pelvis[1]:.3f}, {pelvis[2]:.3f}) m",
        (
            "facing to +X: nan"
            if not np.isfinite(facing_angle_deg)
            else f"facing to +X: {facing_angle_deg:.2f} deg"
        ),
        f"root trans xyz: ({root_translation_first[0]:.3f}, {root_translation_first[1]:.3f}, {root_translation_first[2]:.3f}) m",
    ]
    fig.text(0.01, 0.01, " | ".join(text_lines), fontsize=9)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_first_frame(
    json_payload: dict[str, object],
    npz_path: Path,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    with np.load(npz_path) as arrays:
        joints_can = require_array(arrays, "joints_can", ndim=3, shape_tail=(3,))
        trans_can = require_array(arrays, "trans_can", ndim=2, shape_tail=(3,))

    joints_first = joints_can[0]
    facing_vector, facing_angle_deg = first_frame_body_facing(joints_can)
    root_first = trans_can[0]

    subset_name = str(json_payload.get("subset") or "unknown")
    subject_id = str(json_payload.get("subject_id") or "unknown")
    trial_id = str(json_payload.get("trial_id") or "unknown")
    title = f"{subset_name} | {subject_id} | {trial_id} | first frame"

    _render_first_frame_png(
        joints_first=joints_first,
        root_translation_first=root_first,
        facing_vector=facing_vector,
        facing_angle_deg=facing_angle_deg,
        output_path=output_path,
        title=title,
        dpi=dpi,
    )


def process_sequence(
    json_path: Path,
    output_root: Path,
    *,
    skip_existing: bool,
    dpi: int,
) -> str:
    json_payload = load_json(json_path)
    subset_name = str(json_payload.get("subset") or json_path.parent.name)
    output_path = output_root / FIRST_FRAME_DIRNAME / safe_filename_part(subset_name) / f"{json_path.stem}.png"

    if skip_existing and output_path.exists():
        return "skipped_existing"

    npz_path = resolve_npz_path(json_path, json_payload)
    if not npz_path.exists():
        raise ValueError(f"missing NPZ file: {npz_path}")

    render_first_frame(json_payload, npz_path, output_path, dpi=dpi)
    return "written"


def iter_subset_dirs(input_dir: Path, subset_filter: set[str] | None) -> list[Path]:
    subset_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if subset_filter is None:
        return subset_dirs
    return [path for path in subset_dirs if path.name in subset_filter]


def process_subset(
    subset_dir: Path,
    output_root: Path,
    *,
    skip_existing: bool,
    dpi: int,
    max_sequences: int | None,
) -> dict[str, int]:
    stats = {
        "inspected": 0,
        "written": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
    }

    for json_path in sorted(subset_dir.glob("*.json")):
        if max_sequences is not None and stats["inspected"] >= max_sequences:
            break
        stats["inspected"] += 1
        try:
            outcome = process_sequence(json_path, output_root, skip_existing=skip_existing, dpi=dpi)
        except Exception as exc:
            warnings.warn(f"{subset_dir.name}/{json_path.name}: {exc}; skipped", stacklevel=2)
            stats["skipped_invalid"] += 1
            continue
        stats[outcome] += 1
    return stats


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    subset_filter = set(args.subsets) if args.subsets else None

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    subset_dirs = iter_subset_dirs(input_dir, subset_filter)
    if not subset_dirs:
        raise FileNotFoundError(f"no subset directories found under {input_dir}")

    totals = {
        "inspected": 0,
        "written": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
    }

    for subset_dir in subset_dirs:
        stats = process_subset(
            subset_dir,
            output_dir,
            skip_existing=args.skip_existing,
            dpi=args.dpi,
            max_sequences=args.max_sequences_per_subset,
        )
        for key, value in stats.items():
            totals[key] += value
        print(
            f"{subset_dir.name}: inspected={stats['inspected']} written={stats['written']} "
            f"skipped_existing={stats['skipped_existing']} skipped_invalid={stats['skipped_invalid']}"
        )

    print(
        "TOTAL: "
        f"inspected={totals['inspected']} written={totals['written']} "
        f"skipped_existing={totals['skipped_existing']} skipped_invalid={totals['skipped_invalid']} "
        f"output={output_dir / FIRST_FRAME_DIRNAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())