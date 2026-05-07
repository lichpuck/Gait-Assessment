"""Diagnostic plotting for D_Segmentation."""

from __future__ import annotations

from pathlib import Path
import os

from .config import MPLCONFIGDIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from .config import PRIMARY_LABELS, SegmentationConfig
from .rule_engine import true_runs


LABEL_COLORS = {
    "stand_to_sit": "#7b61ff",
    "sit": "#377eb8",
    "sit_to_stand": "#ff7f00",
    "turn": "#e7298a",
    "walk": "#1b9e77",
    "adjust": "#d4a017",
}
HESITATION_COLOR = "#f4d35e"


def _add_interval_spans(ax: plt.Axes, time_sec: np.ndarray, mask: np.ndarray, *, color: str, alpha: float) -> None:
    for start, end in true_runs(mask):
        start_time = float(time_sec[start])
        end_index = min(end + 1, len(time_sec) - 1)
        end_time = float(time_sec[end_index])
        ax.axvspan(start_time, end_time, color=color, alpha=alpha, linewidth=0.0, zorder=0)


def _overlay_primary_spans(ax: plt.Axes, time_sec: np.ndarray, labels: np.ndarray) -> None:
    for label in PRIMARY_LABELS:
        mask = np.asarray(labels == label, dtype=bool)
        if np.any(mask):
            _add_interval_spans(ax, time_sec, mask, color=LABEL_COLORS[label], alpha=0.10)


def create_diagnostic_plot(result, output_path: str | Path, config: SegmentationConfig) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    time_sec = result.signals.time_sec
    labels = np.asarray(result.primary.primary_label, dtype=object)
    fig, axes = plt.subplots(3, 1, figsize=(14.0, 8.5), sharex=True, constrained_layout=True)
    ax_distance, ax_turn, ax_height = axes

    fig.suptitle(
        f"{result.sequence.subset_name} / {result.sequence.subject_id} / {result.sequence.trial_id}\n"
        f"segments={len(result.segments)}, hesitation={int(np.count_nonzero(result.hesitation_mask))} frames"
    )

    for ax in axes:
        _overlay_primary_spans(ax, time_sec, labels)

    ax_distance.plot(time_sec, result.signals.distance_from_start_m, color="#3a3a3a", linewidth=1.5)
    ax_distance.set_ylabel("distance (m)")
    ax_distance.legend(
        handles=[Patch(facecolor=LABEL_COLORS[label], alpha=0.10, label=label) for label in PRIMARY_LABELS],
        loc="upper right",
        fontsize=8,
        ncol=3,
    )

    _add_interval_spans(ax_turn, time_sec, result.hesitation_mask, color=HESITATION_COLOR, alpha=0.28)
    ax_turn.plot(time_sec, result.signals.turn_angle_from_start_deg, color="#444444", linewidth=1.4)
    ax_turn.axhline(15.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_turn.axhline(165.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_turn.set_ylabel("turn (deg)")
    ax_turn.legend(
        handles=[Patch(facecolor=HESITATION_COLOR, alpha=0.28, label="hesitation")],
        loc="upper right",
        fontsize=8,
    )

    ax_height.plot(time_sec, result.signals.pelvis_height_norm, color="#2c7f33", linewidth=1.4)
    ax_height.axhline(0.2, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7)
    ax_height.axhline(0.8, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7)
    ax_height.set_ylabel("height norm")
    ax_height.set_xlabel("time (s)")

    fig.savefig(output, dpi=config.diagnostic_dpi)
    plt.close(fig)
    return output
