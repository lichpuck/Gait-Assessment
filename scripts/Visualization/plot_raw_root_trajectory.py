#!/usr/bin/env python3
"""Plot raw SMPL root translation trajectories from CARE-PD pickle files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DATA_DIR = ROOT / "raw_data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Visualization"
RAW_SEQUENCE_DIRNAME = "raw_sequence"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"
MIN_DURATION_SEC = 3.0
DIAGNOSTIC_DPI = 160

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw SMPL root translation trajectories for all CARE-PD raw_data sequences."
    )
    parser.add_argument(
        "--raw-data",
        "--raw-data-dir",
        dest="raw_data_dir",
        default=str(RAW_DATA_DIR),
        help="Directory containing raw subset .pkl files. Searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory. PNG files are written under {RAW_SEQUENCE_DIRNAME}/<subset>/.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=MIN_DURATION_SEC,
        help="Skip sequences shorter than this duration.",
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


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def iter_subset_paths(raw_data_dir: Path) -> list[Path]:
    return sorted(path for path in raw_data_dir.rglob("*.pkl") if path.is_file())


def extract_fps(payload: dict[str, object]) -> float:
    fps_value = payload.get("fps", payload.get("FPS"))
    if fps_value is None:
        raise ValueError("missing fps/FPS")
    fps_array = np.asarray(fps_value, dtype=np.float64).reshape(-1)
    if fps_array.size == 0:
        raise ValueError("fps/FPS is empty")
    fps = float(fps_array[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps/FPS must be positive, got {fps!r}")
    return fps


def extract_trans(payload: dict[str, object]) -> np.ndarray:
    if "trans" not in payload:
        raise ValueError("missing trans")
    trans = np.asarray(payload["trans"], dtype=np.float64)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must have shape (T, 3), got {trans.shape}")
    if trans.shape[0] == 0:
        raise ValueError("trans has zero frames")
    if not np.all(np.isfinite(trans)):
        raise ValueError("trans contains non-finite values")
    return trans


def plot_root_trajectory(trans: np.ndarray, fps: float, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = trans[:, 0]
    y = trans[:, 1]
    z = trans[:, 2]
    t = np.arange(trans.shape[0], dtype=np.float64) / fps

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), constrained_layout=True)
    planes = (
        (axes[0, 0], x, y, "X (unitless)", "Y (unitless)"),
        (axes[1, 0], y, z, "Y (unitless)", "Z (unitless)"),
        (axes[2, 0], x, z, "X (unitless)", "Z (unitless)"),
    )
    time_series = (
        (axes[0, 1], t, x, "t (s)", "X (unitless)"),
        (axes[1, 1], t, y, "t (s)", "Y (unitless)"),
        (axes[2, 1], t, z, "t (s)", "Z (unitless)"),
    )

    for ax, horizontal, vertical, xlabel, ylabel in planes:
        ax.plot(horizontal, vertical, color="#2563eb", linewidth=1.8)
        ax.scatter(horizontal[0], vertical[0], color="#16a34a", s=42, zorder=3, label="Start")
        ax.scatter(horizontal[-1], vertical[-1], color="#dc2626", s=42, zorder=3, label="End")
        ax.annotate("Start", (horizontal[0], vertical[0]), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.annotate("End", (horizontal[-1], vertical[-1]), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", frameon=False)

    for ax, horizontal, vertical, xlabel, ylabel in time_series:
        ax.plot(horizontal, vertical, color="#2563eb", linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def process_subset(
    subset_path: Path,
    output_root: Path,
    *,
    min_duration_sec: float,
    dpi: int,
    skip_existing: bool,
    max_sequences: int | None,
) -> dict[str, int]:
    subset_name = subset_path.stem
    subset_output_dir = output_root / RAW_SEQUENCE_DIRNAME / subset_name
    stats = {
        "inspected": 0,
        "plotted": 0,
        "skipped_short": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
    }

    try:
        dataset = load_pickle(subset_path)
    except Exception as exc:  # pragma: no cover - defensive batch behavior.
        warnings.warn(f"{subset_name}: failed to load {subset_path}: {exc}", stacklevel=2)
        stats["failed"] += 1
        return stats

    if not isinstance(dataset, dict):
        warnings.warn(f"{subset_name}: expected pickle root dict, got {type(dataset).__name__}", stacklevel=2)
        stats["failed"] += 1
        return stats

    for subject_id, trials in dataset.items():
        if not isinstance(trials, dict):
            warnings.warn(f"{subset_name}/{subject_id}: expected trial dict, got {type(trials).__name__}", stacklevel=2)
            stats["skipped_invalid"] += 1
            continue
        for trial_id, payload in trials.items():
            if max_sequences is not None and stats["inspected"] >= max_sequences:
                return stats

            stats["inspected"] += 1
            sequence_label = f"{subset_name}/{subject_id}/{trial_id}"
            output_name = f"{safe_filename_part(subject_id)}__{safe_filename_part(trial_id)}.png"
            output_path = subset_output_dir / output_name

            if skip_existing and output_path.exists():
                stats["skipped_existing"] += 1
                continue

            if not isinstance(payload, dict):
                warnings.warn(
                    f"{sequence_label}: expected sequence dict, got {type(payload).__name__}; skipped",
                    stacklevel=2,
                )
                stats["skipped_invalid"] += 1
                continue

            try:
                trans = extract_trans(payload)
                fps = extract_fps(payload)
            except Exception as exc:
                warnings.warn(f"{sequence_label}: {exc}; skipped", stacklevel=2)
                stats["skipped_invalid"] += 1
                continue

            duration_sec = trans.shape[0] / fps
            if duration_sec < min_duration_sec:
                stats["skipped_short"] += 1
                continue

            try:
                plot_root_trajectory(trans, fps, output_path, dpi=dpi)
            except Exception as exc:  # pragma: no cover - defensive batch behavior.
                warnings.warn(f"{sequence_label}: failed to plot {output_path}: {exc}", stacklevel=2)
                stats["failed"] += 1
                continue

            stats["plotted"] += 1

    return stats


def main() -> int:
    args = parse_args()
    raw_data_dir = Path(args.raw_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"raw_data directory does not exist: {raw_data_dir}")

    subset_paths = iter_subset_paths(raw_data_dir)
    if not subset_paths:
        raise FileNotFoundError(f"No .pkl files found under {raw_data_dir}")

    totals = {
        "inspected": 0,
        "plotted": 0,
        "skipped_short": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
    }

    for subset_path in subset_paths:
        stats = process_subset(
            subset_path,
            output_dir,
            min_duration_sec=args.min_duration_sec,
            dpi=args.dpi,
            skip_existing=args.skip_existing,
            max_sequences=args.max_sequences_per_subset,
        )
        for key, value in stats.items():
            totals[key] += value
        print(
            f"{subset_path.stem}: inspected={stats['inspected']} plotted={stats['plotted']} "
            f"skipped_short={stats['skipped_short']} skipped_existing={stats['skipped_existing']} "
            f"skipped_invalid={stats['skipped_invalid']} failed={stats['failed']}"
        )

    print(
        "TOTAL: "
        f"inspected={totals['inspected']} plotted={totals['plotted']} "
        f"skipped_short={totals['skipped_short']} skipped_existing={totals['skipped_existing']} "
        f"skipped_invalid={totals['skipped_invalid']} failed={totals['failed']} "
        f"output={output_dir / RAW_SEQUENCE_DIRNAME}"
    )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
