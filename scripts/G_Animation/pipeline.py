"""End-to-end rendering pipeline for CARE-PD G_Animation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import AnimationConfig
from .export_utils import write_sequence_manifest
from .io_utils import AnimationSequenceInputs, load_animation_sequence
from .metrics import AnimationMetrics, build_animation_metrics
from .render import render_animation


@dataclass
class AnimationRenderResult:
    sequence: AnimationSequenceInputs
    config: AnimationConfig
    metrics: AnimationMetrics
    render_summary: dict[str, object]
    output_paths: dict[str, Path] = field(default_factory=dict)


def render_sequence(
    sequence: AnimationSequenceInputs,
    *,
    config: AnimationConfig | None = None,
    output_dir: str | Path | None = None,
) -> AnimationRenderResult:
    config = config or AnimationConfig()
    subset_output_dir = Path(output_dir) / sequence.subset_name if output_dir is not None else config.subset_output_dir(sequence.subset_name)
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = subset_output_dir / f"{sequence.stem}{config.video_suffix}"
    manifest_path = subset_output_dir / f"{sequence.stem}{config.manifest_suffix}"
    metrics = build_animation_metrics(sequence, config=config)
    render_summary = render_animation(sequence, metrics, video_path, config=config)
    result = AnimationRenderResult(
        sequence=sequence,
        config=config,
        metrics=metrics,
        render_summary=render_summary,
        output_paths={"mp4": video_path},
    )
    result.output_paths["json"] = write_sequence_manifest(result, manifest_path)
    return result


def process_one_sequence(
    subset: str,
    subject_id: str,
    trial_id: str,
    *,
    config: AnimationConfig | None = None,
    output_dir: str | Path | None = None,
) -> AnimationRenderResult:
    config = config or AnimationConfig()
    sequence = load_animation_sequence(subset, subject_id, trial_id, config=config)
    return render_sequence(sequence, config=config, output_dir=output_dir)