#!/usr/bin/env python3
"""Plot canonical C_Representation sequence time-series features."""

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

DEFAULT_INPUT_DIR = ROOT / "outputs" / "C_Representation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Visualization"
CAN_SEQUENCE_DIRNAME = "can_sequence"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
DEFAULT_FPS = 30.0
DIAGNOSTIC_DPI = 160

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot C_Representation canonical sequence features as per-sequence PNGs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing C_Representation subset folders with .json/.npz pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory. PNG files are written under {CAN_SEQUENCE_DIRNAME}/<subset>/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DIAGNOSTIC_DPI,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip plotting when the target PNG already exists.",
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


def extract_scalar(value: object, *, name: str) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    result = float(array[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def extract_fps(json_payload: dict[str, object], arrays: np.lib.npyio.NpzFile, sequence_label: str) -> float:
    for source_name, value in (
        ("npz fps", arrays["fps"] if "fps" in arrays.files else None),
        ("basic_info.fps", nested_get(json_payload, "basic_info", "fps")),
        ("metadata.fps", nested_get(json_payload, "metadata", "fps")),
    ):
        if value is None:
            continue
        try:
            fps = extract_scalar(value, name=source_name)
        except Exception as exc:
            warnings.warn(f"{sequence_label}: invalid {source_name}: {exc}", stacklevel=2)
            continue
        if fps > 0:
            return fps
        warnings.warn(f"{sequence_label}: {source_name} must be positive, got {fps!r}", stacklevel=2)

    warnings.warn(f"{sequence_label}: missing valid fps; using default {DEFAULT_FPS:g} FPS", stacklevel=2)
    return DEFAULT_FPS


def extract_time_s(arrays: np.lib.npyio.NpzFile, fps: float, length: int, sequence_label: str) -> np.ndarray:
    if "time_s" in arrays.files:
        time_s = np.asarray(arrays["time_s"], dtype=np.float64).reshape(-1)
        if time_s.shape == (length,) and np.all(np.isfinite(time_s)):
            return time_s
        warnings.warn(
            f"{sequence_label}: invalid time_s shape/value; using frame_index / fps",
            stacklevel=2,
        )
    return np.arange(length, dtype=np.float64) / fps


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


def joint_index(json_payload: dict[str, object], joint_name: str, default: int) -> int:
    value = nested_get(json_payload, "smpl_joint_schema", "joint_index", joint_name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def wrapped_abs_heading_delta_deg(heading_deg: np.ndarray) -> np.ndarray:
    delta = (heading_deg - heading_deg[0] + 180.0) % 360.0 - 180.0
    return np.abs(delta)


def plot_sequence_features(
    *,
    time_s: np.ndarray,
    root_pos_m: np.ndarray,
    left_wrist_z_m: np.ndarray,
    right_wrist_z_m: np.ndarray,
    heading_delta_deg: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.0), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    axes[0].plot(time_s, root_pos_m[:, 0], color="#2563eb", linewidth=1.7, label="root X")
    axes[0].set_ylabel("X position (m)")
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(time_s, root_pos_m[:, 2], color="#111827", linewidth=1.7, label="root Z")
    axes[1].plot(time_s, left_wrist_z_m, color="#16a34a", linewidth=1.4, label="left wrist Z")
    axes[1].plot(time_s, right_wrist_z_m, color="#dc2626", linewidth=1.4, label="right wrist Z")
    axes[1].set_ylabel("Z position (m)")
    axes[1].legend(loc="best", frameon=False, ncols=3)

    axes[2].plot(time_s, heading_delta_deg, color="#7c3aed", linewidth=1.7, label="|heading - first heading|")
    axes[2].set_ylabel("Heading delta (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(bottom=0.0, top=max(180.0, float(np.nanmax(heading_delta_deg)) * 1.05))
    axes[2].legend(loc="best", frameon=False)

    for ax in axes:
        ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.75)
        ax.margins(x=0.01)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def iter_subset_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def process_sequence(json_path: Path, output_root: Path, *, dpi: int, skip_existing: bool) -> str:
    json_payload = load_json(json_path)
    subset_name = str(nested_get(json_payload, "basic_info", "subset") or json_path.parent.name)
    output_path = output_root / CAN_SEQUENCE_DIRNAME / safe_filename_part(subset_name) / f"{json_path.stem}.png"

    if skip_existing and output_path.exists():
        return "skipped_existing"

    sequence_label = f"{subset_name}/{json_path.stem}"
    npz_path = resolve_npz_path(json_path, json_payload)
    if not npz_path.exists():
        raise ValueError(f"missing NPZ file: {npz_path}")

    with np.load(npz_path) as arrays:
        root_pos_m = require_array(arrays, "root_pos_m", ndim=2, shape_tail=(3,))
        joints_can = require_array(arrays, "joints_can", ndim=3, shape_tail=(3,))
        heading_deg = require_array(arrays, "heading_deg", ndim=1)

        length = root_pos_m.shape[0]
        if joints_can.shape[0] != length or heading_deg.shape[0] != length:
            raise ValueError(
                "length mismatch: "
                f"root_pos_m={root_pos_m.shape[0]}, joints_can={joints_can.shape[0]}, heading_deg={heading_deg.shape[0]}"
            )

        left_wrist_index = joint_index(json_payload, "left_wrist", 20)
        right_wrist_index = joint_index(json_payload, "right_wrist", 21)
        joint_count = joints_can.shape[1]
        if not (0 <= left_wrist_index < joint_count and 0 <= right_wrist_index < joint_count):
            raise ValueError(
                f"wrist joint indices out of range for {joint_count} joints: "
                f"left={left_wrist_index}, right={right_wrist_index}"
            )

        fps = extract_fps(json_payload, arrays, sequence_label)
        time_s = extract_time_s(arrays, fps, length, sequence_label)
        heading_delta_deg = wrapped_abs_heading_delta_deg(heading_deg)

        subject_id = nested_get(json_payload, "basic_info", "subject_id")
        trial_id = nested_get(json_payload, "basic_info", "trial_id")
        if subject_id is not None and trial_id is not None:
            title = f"{subset_name} | {subject_id} | {trial_id}"
        else:
            title = sequence_label

        plot_sequence_features(
            time_s=time_s,
            root_pos_m=root_pos_m,
            left_wrist_z_m=joints_can[:, left_wrist_index, 2],
            right_wrist_z_m=joints_can[:, right_wrist_index, 2],
            heading_delta_deg=heading_delta_deg,
            title=title,
            output_path=output_path,
            dpi=dpi,
        )

    return "plotted"


def process_subset(
    subset_dir: Path,
    output_root: Path,
    *,
    dpi: int,
    skip_existing: bool,
    max_sequences: int | None,
) -> dict[str, int]:
    stats = {
        "inspected": 0,
        "plotted": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
    }

    for json_path in sorted(subset_dir.glob("*.json")):
        if max_sequences is not None and stats["inspected"] >= max_sequences:
            break
        stats["inspected"] += 1
        try:
            outcome = process_sequence(json_path, output_root, dpi=dpi, skip_existing=skip_existing)
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
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"C_Representation directory does not exist: {input_dir}")

    subset_dirs = iter_subset_dirs(input_dir)
    if not subset_dirs:
        raise FileNotFoundError(f"No subset directories found under {input_dir}")

    totals = {
        "inspected": 0,
        "plotted": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
    }

    for subset_dir in subset_dirs:
        stats = process_subset(
            subset_dir,
            output_dir,
            dpi=args.dpi,
            skip_existing=args.skip_existing,
            max_sequences=args.max_sequences_per_subset,
        )
        for key, value in stats.items():
            totals[key] += value
        print(
            f"{subset_dir.name}: inspected={stats['inspected']} plotted={stats['plotted']} "
            f"skipped_existing={stats['skipped_existing']} skipped_invalid={stats['skipped_invalid']} "
            f"failed={stats['failed']}"
        )

    print(
        "TOTAL: "
        f"inspected={totals['inspected']} plotted={totals['plotted']} "
        f"skipped_existing={totals['skipped_existing']} skipped_invalid={totals['skipped_invalid']} "
        f"failed={totals['failed']} output={output_dir / CAN_SEQUENCE_DIRNAME}"
    )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
