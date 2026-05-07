#!/usr/bin/env python3
"""Render per-sequence D_Segmentation inspection plots."""

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

DEFAULT_SEGMENTATION_DIR = ROOT / "outputs" / "D_Segmentation"
DEFAULT_REPRESENTATION_DIR = ROOT / "outputs" / "C_Representation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "Visualization"
MPLCONFIGDIR = ROOT / "outputs" / ".matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from scripts.D_Segmentation.config import PRIMARY_LABELS, SegmentationConfig  # noqa: E402
from scripts.D_Segmentation.io_utils import load_representation_sequence  # noqa: E402
from scripts.D_Segmentation.rule_engine import compute_hesitation_mask, true_runs, run_rule_based_primary_segmentation  # noqa: E402
from scripts.D_Segmentation.signals import compute_signals  # noqa: E402


LABEL_COLORS = {
    "stand_to_sit": "#7b61ff",
    "sit": "#377eb8",
    "sit_to_stand": "#ff7f00",
    "turn": "#e7298a",
    "walk": "#1b9e77",
    "adjust": "#d4a017",
}
HESITATION_COLOR = "#f4d35e"
ROW_LABELS = {
    0: ("walk", "adjust"),
    1: ("turn",),
    2: ("stand_to_sit", "sit", "sit_to_stand"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot D_Segmentation inspection figures with row-specific segment overlays."
    )
    parser.add_argument(
        "--segmentation-dir",
        default=str(DEFAULT_SEGMENTATION_DIR),
        help="Directory containing D_Segmentation subset folders with per-sequence JSON files.",
    )
    parser.add_argument(
        "--representation-dir",
        default=str(DEFAULT_REPRESENTATION_DIR),
        help="Directory containing C_Representation subset folders with .json/.npz pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory. PNG files are written under <output-dir>/<subset>/<sequence_stem>.png.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        help="Optional subset filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sequence-stem",
        action="append",
        help="Optional sequence stem filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=SegmentationConfig().diagnostic_dpi,
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


def resolve_existing_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    return candidate if candidate.exists() else None


def resolve_representation_paths(
    segmentation_payload: dict[str, object],
    segmentation_json_path: Path,
    representation_dir: Path,
) -> tuple[Path, Path]:
    sequence = nested_get(segmentation_payload, "sequence")
    if not isinstance(sequence, dict):
        sequence = {}

    source_npz = resolve_existing_path(str(sequence.get("source_c_npz", "")))
    source_json = resolve_existing_path(str(sequence.get("source_c_json", "")))
    if source_npz is not None and source_json is not None:
        return source_npz, source_json

    subset_name = str(sequence.get("subset") or segmentation_json_path.parent.name)
    stem = str(sequence.get("stem") or segmentation_json_path.stem)
    subset_dir = representation_dir / safe_filename_part(subset_name)
    npz_path = subset_dir / f"{stem}.npz"
    json_path = subset_dir / f"{stem}.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return npz_path.resolve(), json_path.resolve()


def iter_segmentation_jsons(segmentation_dir: Path, subset_filters: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for subset_dir in sorted(path for path in segmentation_dir.iterdir() if path.is_dir()):
        subset_name = subset_dir.name
        if subset_filters is not None and subset_name not in subset_filters:
            continue
        for json_path in sorted(subset_dir.glob("*.json")):
            if json_path.name == "segmentation_summary.json":
                continue
            if json_path.stem == "segmentation_summary":
                continue
            paths.append(json_path)
    return paths


def build_primary_masks(frame_count: int, segments: list[dict[str, object]]) -> dict[str, np.ndarray]:
    masks = {label: np.zeros(frame_count, dtype=bool) for label in PRIMARY_LABELS}
    for segment in segments:
        label = str(segment.get("label", ""))
        if label not in masks:
            continue
        start = max(int(segment.get("start_frame", 0)), 0)
        end = min(int(segment.get("end_frame", -1)), frame_count - 1)
        if start <= end:
            masks[label][start : end + 1] = True
    return masks


def add_interval_spans(ax: plt.Axes, time_sec: np.ndarray, mask: np.ndarray, *, color: str, alpha: float) -> None:
    for start, end in true_runs(mask):
        start_time = float(time_sec[start])
        end_index = min(end + 1, len(time_sec) - 1)
        end_time = float(time_sec[end_index])
        ax.axvspan(start_time, end_time, color=color, alpha=alpha, linewidth=0.0, zorder=0)


def create_plot(
    *,
    output_path: Path,
    title: str,
    time_sec: np.ndarray,
    root_x_displacement_m: np.ndarray,
    turn_angle_from_start_deg: np.ndarray,
    pelvis_height_norm: np.ndarray,
    primary_masks: dict[str, np.ndarray],
    hesitation_mask: np.ndarray,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 8.8), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    row_series = (
        (0, root_x_displacement_m, "#111827", "Root X disp. (m)"),
        (1, turn_angle_from_start_deg, "#374151", "|Heading - first| (deg)"),
        (2, pelvis_height_norm, "#2c7f33", "Pelvis height norm"),
    )

    for row_index, values, line_color, ylabel in row_series:
        ax = axes[row_index]
        for label in ROW_LABELS[row_index]:
            add_interval_spans(ax, time_sec, primary_masks[label], color=LABEL_COLORS[label], alpha=0.14)
        if row_index == 1:
            add_interval_spans(ax, time_sec, hesitation_mask, color=HESITATION_COLOR, alpha=0.28)
        ax.plot(time_sec, values, color=line_color, linewidth=1.5, zorder=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", linewidth=0.8, alpha=0.75)
        ax.margins(x=0.01)

    axes[2].set_xlabel("Time (s)")
    axes[1].set_ylim(bottom=0.0, top=max(180.0, float(np.nanmax(turn_angle_from_start_deg)) * 1.05))
    axes[2].set_ylim(bottom=-0.02, top=max(1.02, float(np.nanmax(pelvis_height_norm)) * 1.05))

    axes[0].legend(
        handles=[Patch(facecolor=LABEL_COLORS[label], alpha=0.14, label=label) for label in ROW_LABELS[0]],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    axes[1].legend(
        handles=[
            Patch(facecolor=LABEL_COLORS["turn"], alpha=0.14, label="turn"),
            Patch(facecolor=HESITATION_COLOR, alpha=0.28, label="hesitation"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    axes[2].legend(
        handles=[Patch(facecolor=LABEL_COLORS[label], alpha=0.14, label=label) for label in ROW_LABELS[2]],
        loc="upper right",
        frameon=False,
        fontsize=8,
        ncol=3,
    )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def process_sequence(
    segmentation_json_path: Path,
    *,
    representation_dir: Path,
    output_dir: Path,
    config: SegmentationConfig,
    dpi: int,
    skip_existing: bool,
) -> str:
    segmentation_payload = load_json(segmentation_json_path)
    if segmentation_payload.get("module") != "D_Segmentation":
        raise ValueError(f"{segmentation_json_path} is not a D_Segmentation JSON artifact")

    sequence = nested_get(segmentation_payload, "sequence")
    if not isinstance(sequence, dict):
        raise ValueError(f"{segmentation_json_path} is missing sequence metadata")

    subset_name = str(sequence.get("subset") or segmentation_json_path.parent.name)
    stem = str(sequence.get("stem") or segmentation_json_path.stem)
    output_path = output_dir / safe_filename_part(subset_name) / f"{stem}.png"
    if skip_existing and output_path.exists():
        return "skipped_existing"

    representation_npz_path, representation_json_path = resolve_representation_paths(
        segmentation_payload,
        segmentation_json_path,
        representation_dir,
    )
    representation = load_representation_sequence(representation_npz_path, representation_json_path)
    signals = compute_signals(representation.arrays, representation.fps, config)
    primary = run_rule_based_primary_segmentation(signals, config)
    hesitation_mask = compute_hesitation_mask(signals, primary.primary_label_index, config)

    frame_count = representation.num_frames
    segments = segmentation_payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{segmentation_json_path} has invalid segments payload")
    primary_masks = build_primary_masks(frame_count, [segment for segment in segments if isinstance(segment, dict)])

    root_x_displacement_m = representation.arrays["root_pos_m"][:, 0].astype(np.float32)
    if root_x_displacement_m.size:
        root_x_displacement_m = root_x_displacement_m - float(root_x_displacement_m[0])

    title = (
        f"{representation.subset_name} / {representation.subject_id} / {representation.trial_id}\n"
        f"frames={representation.num_frames}, fps={representation.fps:.1f}, segments={len(segments)}"
    )
    create_plot(
        output_path=output_path,
        title=title,
        time_sec=signals.time_sec,
        root_x_displacement_m=root_x_displacement_m,
        turn_angle_from_start_deg=signals.turn_angle_from_start_deg,
        pelvis_height_norm=signals.pelvis_height_norm,
        primary_masks=primary_masks,
        hesitation_mask=hesitation_mask,
        dpi=dpi,
    )
    return "plotted"


def main() -> int:
    args = parse_args()
    segmentation_dir = Path(args.segmentation_dir).resolve()
    representation_dir = Path(args.representation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = SegmentationConfig()
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    if not segmentation_dir.exists():
        raise FileNotFoundError(f"segmentation directory does not exist: {segmentation_dir}")
    if not representation_dir.exists():
        raise FileNotFoundError(f"representation directory does not exist: {representation_dir}")

    subset_filters = {name.strip() for name in args.subset or [] if name and name.strip()} or None
    stem_filters = {name.strip() for name in args.sequence_stem or [] if name and name.strip()}
    json_paths = iter_segmentation_jsons(segmentation_dir, subset_filters)
    if stem_filters:
        json_paths = [path for path in json_paths if path.stem in stem_filters]
    if not json_paths:
        raise FileNotFoundError("no D_Segmentation JSON files matched the requested filters")

    totals = {
        "inspected": 0,
        "plotted": 0,
        "skipped_existing": 0,
        "failed": 0,
    }
    per_subset_counts: dict[str, int] = {}

    for json_path in json_paths:
        subset_name = json_path.parent.name
        if args.max_sequences_per_subset is not None:
            current = per_subset_counts.get(subset_name, 0)
            if current >= args.max_sequences_per_subset:
                continue
            per_subset_counts[subset_name] = current + 1

        totals["inspected"] += 1
        sequence_label = f"{subset_name}/{json_path.stem}"
        try:
            outcome = process_sequence(
                json_path,
                representation_dir=representation_dir,
                output_dir=output_dir,
                config=config,
                dpi=args.dpi,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            warnings.warn(f"{sequence_label}: failed to render segment check plot: {exc}", stacklevel=2)
            totals["failed"] += 1
            continue

        totals[outcome] += 1

    print(
        "segment_check: "
        f"inspected={totals['inspected']} plotted={totals['plotted']} "
        f"skipped_existing={totals['skipped_existing']} failed={totals['failed']}"
    )
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())