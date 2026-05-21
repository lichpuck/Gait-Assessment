"""Composite MP4 rendering for CARE-PD G_Animation."""

from __future__ import annotations

import os
from pathlib import Path
import textwrap

from .config import LABEL_COLORS, MPLCONFIGDIR, AnimationConfig

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from scripts.B_Canonicalization.joints_schema import JOINT_INDEX

from .io_utils import AnimationSequenceInputs
from .metrics import AnimationMetrics


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

METRIC_LINE_COLORS = {
    "stability": "#1f77b4",
    "balance": "#ff7f0e",
    "symmetry": "#2ca02c",
    "coordination": "#d62728",
    "mobility": "#9467bd",
    "control": "#8c564b",
}

CJK_FONT_CANDIDATES = (
    "Hiragino Sans GB",
    "STHeiti",
    "Songti SC",
    "Arial Unicode MS",
)

ABNORMAL_SUMMARY_PHRASES = (
    "重度异常",
    "中度异常",
    "轻度异常",
    "明显不对称性",
    "节律不规则",
    "不对称",
    "不规则",
    "减慢",
    "迟疑",
    "受限",
    "不稳",
    "下降",
)


def _preferred_cjk_font() -> str | None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            return font_name
    return None


def _split_summary_clauses(text: str) -> tuple[str, tuple[str, ...]]:
    normalized = str(text or "").strip()
    if not normalized:
        return "", ()

    header = ""
    body = normalized
    for delimiter in ("：", ":"):
        if delimiter in normalized:
            prefix, suffix = normalized.split(delimiter, 1)
            header = f"{prefix.strip()}{delimiter}"
            body = suffix.strip()
            break

    cleaned_body = body.replace("；", "、").rstrip("。")
    clauses = tuple(item.strip() for item in cleaned_body.split("、") if item.strip())
    return header, clauses


def _is_abnormal_clause(text: str) -> bool:
    clause = str(text or "").strip()
    if not clause:
        return False
    return any(phrase in clause for phrase in ABNORMAL_SUMMARY_PHRASES)


def _edge_index_pairs() -> tuple[tuple[int, int], ...]:
    return tuple((JOINT_INDEX[start], JOINT_INDEX[end]) for start, end in SKELETON_EDGES)


def _segment_intervals(inputs: AnimationSequenceInputs) -> list[tuple[float, float, str]]:
    frame_period = 1.0 / float(inputs.fps)
    intervals: list[tuple[float, float, str]] = []
    time_s = np.asarray(inputs.c_sequence.arrays["time_s"], dtype=np.float32)
    for segment in inputs.segments:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        intervals.append((float(time_s[start_frame]), float(time_s[end_frame] + frame_period), str(segment["label"])))
    return intervals


def _apply_segment_spans(ax: plt.Axes, inputs: AnimationSequenceInputs) -> None:
    for start_time, end_time, label in _segment_intervals(inputs):
        ax.axvspan(start_time, end_time, color=LABEL_COLORS[label], alpha=0.08, linewidth=0.0, zorder=0)


def _set_metric_ylim(ax: plt.Axes, values: np.ndarray, key: str) -> None:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        ax.set_ylim(-1.0, 1.0)
        return
    if key == "coordination":
        ax.set_ylim(-1.05, 1.05)
        return
    value_min = float(np.min(finite))
    value_max = float(np.max(finite))
    if np.isclose(value_min, value_max):
        padding = max(abs(value_min) * 0.1, 0.1)
    else:
        padding = (value_max - value_min) * 0.12
    ax.set_ylim(value_min - padding, value_max + padding)


def _compute_scene_bounds(joints_can: np.ndarray, config: AnimationConfig) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = np.asarray(joints_can, dtype=np.float32).reshape(-1, 3)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-3)
    radius = 0.5 * np.max(span) * (1.0 + float(config.skeleton_padding_ratio))
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(max(0.0, center[2] - radius))),
        (float(center[2] + radius)),
    )


def _add_coordinate_axes(ax, config: AnimationConfig) -> None:
    axis_length = float(config.axis_length_m)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axes = (
        (np.array([axis_length, 0.0, 0.0], dtype=np.float32), "#d62728", "X"),
        (np.array([0.0, axis_length, 0.0], dtype=np.float32), "#1f77b4", "Y"),
        (np.array([0.0, 0.0, axis_length], dtype=np.float32), "#2ca02c", "Z"),
    )
    for vector, color, label in axes:
        endpoint = origin + vector
        ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linewidth=2.0,
            alpha=0.9,
        )
        ax.text(float(endpoint[0]), float(endpoint[1]), float(endpoint[2]), label, color=color, fontsize=9)


def _wrap_summary(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    return textwrap.fill(normalized, width=28)


def _render_summary_panel(ax: plt.Axes, summary_text: str, *, fontfamily: str | None = None) -> None:
    ax.axis("off")
    ax.set_title("Overall assessment", loc="left", fontsize=11, fontweight="bold")

    text_kwargs = {"fontfamily": fontfamily} if fontfamily is not None else {}
    header, clauses = _split_summary_clauses(summary_text)
    y_position = 0.88
    if header:
        ax.text(0.0, y_position, header, va="top", ha="left", fontsize=9.5, color="#4b5563", **text_kwargs)
        y_position -= 0.14

    if not clauses:
        ax.text(0.0, y_position, "无总体描述", va="top", ha="left", fontsize=9.5, color="#9ca3af", **text_kwargs)
        return

    for clause in clauses:
        wrapped = textwrap.fill(clause, width=18)
        line_count = max(len(wrapped.splitlines()), 1)
        color = "#c62828" if _is_abnormal_clause(clause) else "#1f2937"
        ax.text(0.0, y_position, wrapped, va="top", ha="left", fontsize=9.5, color=color, **text_kwargs)
        y_position -= 0.11 * line_count + 0.05
        if y_position < 0.08:
            break


def _draw_label_legend(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_title("Segments", loc="left", fontsize=10, fontweight="bold")

    labels = list(LABEL_COLORS.items())
    column_x = (0.02, 0.53)
    row_y = (0.76, 0.48, 0.20)
    for index, (label, color) in enumerate(labels):
        column = index % 2
        row = index // 2
        x_pos = column_x[column]
        y_pos = row_y[row]
        ax.add_patch(
            Rectangle(
                (x_pos, y_pos - 0.08),
                0.08,
                0.10,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
        )
        ax.text(x_pos + 0.10, y_pos, label, transform=ax.transAxes, va="center", ha="left", fontsize=7.6)


def _render_meta_panel(ax: plt.Axes, inputs: AnimationSequenceInputs, *, summary_present: bool) -> None:
    ax.axis("off")
    ax.set_title("Status", loc="left", fontsize=10, fontweight="bold")
    lines = (
        f"fps: {inputs.fps:.2f}",
        f"duration: {inputs.duration_sec:.2f}s",
        f"summary: {'present' if summary_present else 'empty'}",
    )
    ax.text(0.0, 0.86, "\n".join(lines), va="top", ha="left", fontsize=9.5, linespacing=1.55)


def render_animation(
    inputs: AnimationSequenceInputs,
    metrics: AnimationMetrics,
    output_path: str | Path,
    *,
    config: AnimationConfig | None = None,
) -> dict[str, object]:
    config = config or AnimationConfig()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    joints_sequence = np.asarray(inputs.b_sequence.joints_can, dtype=np.float32)
    root_sequence = np.asarray(inputs.b_sequence.trans_can, dtype=np.float32)
    edge_pairs = _edge_index_pairs()
    x_limits, y_limits, z_min, z_max = _compute_scene_bounds(joints_sequence, config)
    video_fps = float(config.output_fps or inputs.fps)
    summary_text = str(inputs.summary_text_zh or "").strip()
    summary_font = _preferred_cjk_font()

    figure = plt.figure(
        figsize=(config.figure_width_in, config.figure_height_in),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(8, 12)
    ax_3d = figure.add_subplot(grid[:, :7], projection="3d")
    top_right = grid[:2, 7:].subgridspec(1, 2, width_ratios=(3.2, 1.8), wspace=0.12)
    ax_summary = figure.add_subplot(top_right[0, 0])
    side_panel = top_right[0, 1].subgridspec(2, 1, height_ratios=(1.0, 1.15), hspace=0.08)
    ax_meta = figure.add_subplot(side_panel[0, 0])
    ax_legend = figure.add_subplot(side_panel[1, 0])
    metric_axes = [figure.add_subplot(grid[row, 7:]) for row in range(2, 8)]

    figure.suptitle(
        f"{inputs.subset_name} / {inputs.subject_id} / {inputs.trial_id}",
        fontsize=13,
        fontweight="bold",
    )

    ax_3d.set_title("Canonical skeleton animation")
    ax_3d.view_init(elev=config.camera_elev_deg, azim=config.camera_azim_deg)
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_xlim(*x_limits)
    ax_3d.set_ylim(*y_limits)
    ax_3d.set_zlim(z_min, z_max)
    ax_3d.set_box_aspect((x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_max - z_min))
    ax_3d.grid(True, alpha=0.25)
    _add_coordinate_axes(ax_3d, config)

    skeleton_artists = []
    for _ in edge_pairs:
        line_artist, = ax_3d.plot([], [], [], linewidth=config.skeleton_line_width, alpha=0.95)
        skeleton_artists.append(line_artist)
    root_trail_artist, = ax_3d.plot([], [], [], color="#7f7f7f", linewidth=config.root_trail_line_width, alpha=0.75)
    joint_artist = ax_3d.scatter([], [], [], s=18, depthshade=False, color=LABEL_COLORS["walk"])
    frame_text = ax_3d.text2D(0.02, 0.96, "", transform=ax_3d.transAxes, fontsize=10, fontweight="bold")

    _render_summary_panel(ax_summary, summary_text, fontfamily=summary_font)
    _render_meta_panel(ax_meta, inputs, summary_present=bool(summary_text))
    _draw_label_legend(ax_legend)

    time_s = np.asarray(metrics.time_s, dtype=np.float32)
    metric_cursors = []
    metric_points = []
    metric_texts = []
    for ax, series in zip(metric_axes, metrics.series):
        _apply_segment_spans(ax, inputs)
        line_color = METRIC_LINE_COLORS.get(series.key, "#333333")
        values = np.asarray(series.values, dtype=np.float32)
        ax.plot(time_s, values, color=line_color, linewidth=config.curve_line_width)
        cursor = ax.axvline(float(time_s[0]), color="#111111", linewidth=1.0, alpha=0.85)
        point_artist, = ax.plot([float(time_s[0])], [float(values[0])], marker="o", color=line_color, markersize=4)
        metric_value_text = ax.text(0.99, 0.80, "", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        metric_title_kwargs = {"fontfamily": summary_font} if summary_font is not None else {}
        ax.text(0.01, 0.80, series.title, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold", **metric_title_kwargs)
        ax.set_ylabel(series.unit, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(float(time_s[0]), float(time_s[-1]))
        _set_metric_ylim(ax, values, series.key)
        metric_cursors.append(cursor)
        metric_points.append(point_artist)
        metric_texts.append(metric_value_text)
    for ax in metric_axes[:-1]:
        ax.tick_params(labelbottom=False)
    metric_axes[-1].set_xlabel("Time (s)")

    def update(frame_index: int):
        joints = joints_sequence[frame_index]
        current_label = str(inputs.primary_label_by_frame[frame_index])
        current_color = LABEL_COLORS[current_label]
        for line_artist, (start_index, end_index) in zip(skeleton_artists, edge_pairs):
            segment = joints[[start_index, end_index]]
            line_artist.set_data(segment[:, 0], segment[:, 1])
            line_artist.set_3d_properties(segment[:, 2])
            line_artist.set_color(current_color)
        joint_artist._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        joint_artist.set_color(current_color)
        trail = root_sequence[: frame_index + 1]
        root_trail_artist.set_data(trail[:, 0], trail[:, 1])
        root_trail_artist.set_3d_properties(trail[:, 2])

        current_time = float(time_s[frame_index])
        frame_text.set_text(
            f"Label: {current_label} | frame: {frame_index + 1}/{inputs.num_frames} | t={current_time:.2f}s"
        )
        for cursor, point_artist, value_text, series in zip(metric_cursors, metric_points, metric_texts, metrics.series):
            value = float(series.values[frame_index])
            cursor.set_xdata([current_time, current_time])
            point_artist.set_data([current_time], [value])
            value_text.set_text(f"{value:.2f} {series.unit}")

        return [
            *skeleton_artists,
            root_trail_artist,
            joint_artist,
            frame_text,
            *metric_cursors,
            *metric_points,
            *metric_texts,
        ]

    animation_object = animation.FuncAnimation(
        figure,
        update,
        frames=inputs.num_frames,
        interval=1000.0 / max(video_fps, 1.0),
        blit=False,
        repeat=False,
    )
    writer = animation.FFMpegWriter(fps=video_fps, metadata={"title": inputs.stem})
    animation_object.save(output_file, writer=writer, dpi=config.render_dpi)
    plt.close(figure)

    return {
        "video_fps": video_fps,
        "frame_count": inputs.num_frames,
        "duration_sec": inputs.duration_sec,
        "camera": {
            "elev_deg": float(config.camera_elev_deg),
            "azim_deg": float(config.camera_azim_deg),
        },
        "layout": "3d_skeleton + overall_text + 6_curves",
    }