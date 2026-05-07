"""Rule-based primary and auxiliary motion segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PRIMARY_LABELS, PRIMARY_LABEL_TO_INDEX, SegmentationConfig, seconds_to_frames
from .signals import SignalSet


@dataclass(frozen=True)
class IntervalEvent:
    start_frame: int
    end_frame: int
    core_start_frame: int
    core_end_frame: int
    kind: str


@dataclass(frozen=True)
class PrimaryRuleResult:
    primary_label_index: np.ndarray
    primary_label: np.ndarray
    primary_confidence: np.ndarray
    source_rule: np.ndarray
    masks: dict[str, np.ndarray]
    adjust_mask: np.ndarray
    stand_to_sit_events: tuple[IntervalEvent, ...]
    sit_to_stand_events: tuple[IntervalEvent, ...]
    turn_events: tuple[IntervalEvent, ...]


@dataclass(frozen=True)
class SegmentRecord:
    label: str
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    source_rule: str
    confidence: float
    hesitation_overlap_frames: int
    hesitation_overlap_ratio: float
    auxiliary_overlap_labels: str


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        if not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def remove_short_true_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if min_length <= 1:
        return result
    for start, end in true_runs(result):
        if end - start + 1 < min_length:
            result[start : end + 1] = False
    return result


def _find_crossing(
    values: np.ndarray,
    *,
    start_index: int,
    threshold: float,
    direction: str,
    allowed_mask: np.ndarray | None = None,
) -> int | None:
    array = np.asarray(values, dtype=np.float32)
    allowed = None if allowed_mask is None else np.asarray(allowed_mask, dtype=bool)
    for index in range(max(int(start_index), 1), len(array)):
        if allowed is not None and not allowed[index]:
            continue
        previous = float(array[index - 1])
        current = float(array[index])
        if direction == "up" and previous < threshold <= current:
            return index
        if direction == "down" and previous > threshold >= current:
            return index
    return None


def _backtrack_to_local_extremum(values: np.ndarray, index: int, *, mode: str) -> int:
    array = np.asarray(values, dtype=np.float32)
    position = int(index)
    while position > 0:
        previous = float(array[position - 1])
        current = float(array[position])
        if mode == "max" and previous > current:
            position -= 1
            continue
        if mode == "min" and previous < current:
            position -= 1
            continue
        break
    return position


def _forward_to_local_extremum(values: np.ndarray, index: int, *, mode: str) -> int:
    array = np.asarray(values, dtype=np.float32)
    position = int(index)
    while position < len(array) - 1:
        current = float(array[position])
        following = float(array[position + 1])
        if mode == "max" and following > current:
            position += 1
            continue
        if mode == "min" and following < current:
            position += 1
            continue
        break
    return position


def _intervals_to_mask(length: int, intervals: list[IntervalEvent]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for interval in intervals:
        start = max(int(interval.start_frame), 0)
        end = min(int(interval.end_frame), length - 1)
        if start <= end:
            mask[start : end + 1] = True
    return mask


def _is_turn_boundary_angle(angle_deg: float) -> bool:
    return 0.0 <= angle_deg <= 5.0 or 175.0 <= angle_deg <= 180.0


def detect_sts_intervals(pelvis_height_norm: np.ndarray) -> dict[str, list[IntervalEvent]]:
    height = np.asarray(pelvis_height_norm, dtype=np.float32)
    cursor = 1
    stand_to_sit: list[IntervalEvent] = []
    sit_to_stand: list[IntervalEvent] = []

    while cursor < len(height):
        next_down_08 = _find_crossing(height, start_index=cursor, threshold=0.8, direction="down")
        next_up_02 = _find_crossing(height, start_index=cursor, threshold=0.2, direction="up")
        candidates = [value for value in (next_down_08, next_up_02) if value is not None]
        if not candidates:
            break

        start_index = min(candidates)
        if next_down_08 is not None and start_index == next_down_08:
            core_end = _find_crossing(height, start_index=start_index + 1, threshold=0.2, direction="down")
            if core_end is None:
                cursor = start_index + 1
                continue
            start_frame = _backtrack_to_local_extremum(height, start_index, mode="max")
            end_frame = _forward_to_local_extremum(height, core_end, mode="min")
            stand_to_sit.append(
                IntervalEvent(start_frame, end_frame, start_index, core_end, "stand_to_sit")
            )
            cursor = end_frame + 1
            continue

        core_end = _find_crossing(height, start_index=start_index + 1, threshold=0.8, direction="up")
        if core_end is None:
            cursor = start_index + 1
            continue
        start_frame = _backtrack_to_local_extremum(height, start_index, mode="min")
        end_frame = _forward_to_local_extremum(height, core_end, mode="max")
        sit_to_stand.append(
            IntervalEvent(start_frame, end_frame, start_index, core_end, "sit_to_stand")
        )
        cursor = end_frame + 1

    return {"stand_to_sit": stand_to_sit, "sit_to_stand": sit_to_stand}


def build_sit_mask_from_sts_masks(length: int, stand_to_sit_mask: np.ndarray, sit_to_stand_mask: np.ndarray) -> np.ndarray:
    sit_mask = np.zeros(length, dtype=bool)
    events = [("stand_to_sit", start, end) for start, end in true_runs(stand_to_sit_mask)] + [
        ("sit_to_stand", start, end) for start, end in true_runs(sit_to_stand_mask)
    ]
    if not events:
        return sit_mask

    events.sort(key=lambda item: (item[1], item[2]))
    first_label, first_start, _ = events[0]
    if first_label == "sit_to_stand" and first_start > 0:
        sit_mask[:first_start] = True

    for index in range(len(events) - 1):
        current_label, _, current_end = events[index]
        next_label, next_start, _ = events[index + 1]
        if current_label == "stand_to_sit" and next_label == "sit_to_stand" and current_end + 1 < next_start:
            sit_mask[current_end + 1 : next_start] = True

    last_label, _, last_end = events[-1]
    if last_label == "stand_to_sit" and last_end < length - 1:
        sit_mask[last_end + 1 :] = True
    return sit_mask


def detect_turn_intervals(
    turn_angle_from_start_deg: np.ndarray,
    walk_speed_mps: np.ndarray,
    turn_speed_deg_s: np.ndarray,
    allowed_mask: np.ndarray,
    config: SegmentationConfig,
) -> list[IntervalEvent]:
    angle = np.asarray(turn_angle_from_start_deg, dtype=np.float32)
    walk = np.asarray(walk_speed_mps, dtype=np.float32)
    turn_speed = np.asarray(turn_speed_deg_s, dtype=np.float32)
    allowed = np.asarray(allowed_mask, dtype=bool)
    intervals: list[IntervalEvent] = []

    for run_start, run_end in true_runs(allowed):
        cursor = run_start
        while cursor <= run_end:
            outward_start = _find_crossing(angle, start_index=cursor, threshold=15.0, direction="up", allowed_mask=allowed)
            outward_end = (
                _find_crossing(angle, start_index=outward_start + 1, threshold=165.0, direction="up", allowed_mask=allowed)
                if outward_start is not None
                else None
            )
            return_start = _find_crossing(angle, start_index=cursor, threshold=165.0, direction="down", allowed_mask=allowed)
            return_end = (
                _find_crossing(angle, start_index=return_start + 1, threshold=15.0, direction="down", allowed_mask=allowed)
                if return_start is not None
                else None
            )

            candidates: list[tuple[str, int, int]] = []
            if outward_start is not None and outward_end is not None and outward_end <= run_end:
                candidates.append(("outward_turn", outward_start, outward_end))
            if return_start is not None:
                if return_end is not None and return_end <= run_end:
                    candidates.append(("return_turn", return_start, return_end))
                elif return_start <= run_end:
                    candidates.append(("return_turn", return_start, run_end))
            if not candidates:
                break

            kind, core_start, core_end = min(candidates, key=lambda item: (item[1], item[2]))
            start_frame = core_start
            for candidate_frame in range(core_start - 1, run_start - 1, -1):
                if float(walk[candidate_frame]) > config.walk_speed_turn_backfill_mps:
                    start_frame = candidate_frame
                    break

            end_frame = core_end
            for candidate_frame in range(core_end + 1, run_end + 1):
                if (
                    float(turn_speed[candidate_frame]) <= config.turn_speed_end_deg_s
                    or _is_turn_boundary_angle(float(angle[candidate_frame]))
                ):
                    end_frame = candidate_frame
                    break

            intervals.append(IntervalEvent(start_frame, end_frame, core_start, core_end, kind))
            cursor = core_end + 1
    return intervals


def labels_to_index_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    array = np.asarray(labels, dtype=np.int32)
    if array.size == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(array[0])
    for index in range(1, len(array)):
        if int(array[index]) != current:
            segments.append((start, index - 1, current))
            start = index
            current = int(array[index])
    segments.append((start, len(array) - 1, current))
    return segments


def apply_adjust_rule(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(labels, dtype=np.int32).copy()
    adjust_mask = np.zeros_like(result, dtype=bool)
    walk_index = PRIMARY_LABEL_TO_INDEX["walk"]
    turn_index = PRIMARY_LABEL_TO_INDEX["turn"]
    stand_to_sit_index = PRIMARY_LABEL_TO_INDEX["stand_to_sit"]
    adjust_index = PRIMARY_LABEL_TO_INDEX["adjust"]

    segments = labels_to_index_segments(result)
    for segment_index, (start, end, label_index) in enumerate(segments):
        if label_index != walk_index or segment_index == 0 or segment_index == len(segments) - 1:
            continue
        previous_label = segments[segment_index - 1][2]
        next_label = segments[segment_index + 1][2]
        if previous_label == turn_index and next_label == stand_to_sit_index:
            result[start : end + 1] = adjust_index
            adjust_mask[start : end + 1] = True
    return result, adjust_mask


def run_rule_based_primary_segmentation(signals: SignalSet, config: SegmentationConfig) -> PrimaryRuleResult:
    frame_count = len(signals.time_sec)
    sts_events = detect_sts_intervals(signals.pelvis_height_norm)
    stand_to_sit_mask = _intervals_to_mask(frame_count, sts_events["stand_to_sit"])
    sit_to_stand_mask = _intervals_to_mask(frame_count, sts_events["sit_to_stand"])
    sit_mask = build_sit_mask_from_sts_masks(frame_count, stand_to_sit_mask, sit_to_stand_mask)
    sit_mask &= ~(stand_to_sit_mask | sit_to_stand_mask)

    locomotion_allowed_mask = ~(stand_to_sit_mask | sit_to_stand_mask | sit_mask)
    turn_events = detect_turn_intervals(
        signals.turn_angle_from_start_deg,
        signals.pelvis_speed_mps,
        signals.turn_speed_deg_s,
        locomotion_allowed_mask,
        config,
    )
    turn_mask = _intervals_to_mask(frame_count, turn_events) & locomotion_allowed_mask

    primary_label_index = np.full(frame_count, PRIMARY_LABEL_TO_INDEX["walk"], dtype=np.int32)
    primary_label_index[turn_mask] = PRIMARY_LABEL_TO_INDEX["turn"]
    primary_label_index[sit_mask] = PRIMARY_LABEL_TO_INDEX["sit"]
    primary_label_index[stand_to_sit_mask] = PRIMARY_LABEL_TO_INDEX["stand_to_sit"]
    primary_label_index[sit_to_stand_mask] = PRIMARY_LABEL_TO_INDEX["sit_to_stand"]
    primary_label_index, adjust_mask = apply_adjust_rule(primary_label_index)

    primary_label = np.array([PRIMARY_LABELS[index] for index in primary_label_index], dtype=object)
    confidence_by_label = {
        "stand_to_sit": 0.95,
        "sit": 0.92,
        "sit_to_stand": 0.95,
        "turn": 0.90,
        "walk": 0.78,
        "adjust": 0.82,
    }
    primary_confidence = np.array([confidence_by_label[label] for label in primary_label], dtype=np.float32)

    source_rule = np.full(frame_count, "walk_residual", dtype=object)
    source_rule[turn_mask] = "turn_rule"
    source_rule[sit_mask] = "sit_fill"
    source_rule[stand_to_sit_mask] = "stand_to_sit_rule"
    source_rule[sit_to_stand_mask] = "sit_to_stand_rule"
    source_rule[adjust_mask] = "adjust_relabel"

    masks = {
        "stand_to_sit": stand_to_sit_mask.astype(bool),
        "sit": sit_mask.astype(bool),
        "sit_to_stand": sit_to_stand_mask.astype(bool),
        "turn": (primary_label_index == PRIMARY_LABEL_TO_INDEX["turn"]),
        "walk": (primary_label_index == PRIMARY_LABEL_TO_INDEX["walk"]),
        "adjust": adjust_mask.astype(bool),
    }

    return PrimaryRuleResult(
        primary_label_index=primary_label_index.astype(np.int32),
        primary_label=primary_label,
        primary_confidence=primary_confidence.astype(np.float32),
        source_rule=source_rule.astype(object),
        masks=masks,
        adjust_mask=adjust_mask.astype(bool),
        stand_to_sit_events=tuple(sts_events["stand_to_sit"]),
        sit_to_stand_events=tuple(sts_events["sit_to_stand"]),
        turn_events=tuple(turn_events),
    )


def compute_hesitation_mask(signals: SignalSet, primary_label_index: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    labels = np.asarray(primary_label_index, dtype=np.int32)
    walk_mask = labels == PRIMARY_LABEL_TO_INDEX["walk"]
    turn_mask = labels == PRIMARY_LABEL_TO_INDEX["turn"]
    hesitation = (walk_mask & (signals.pelvis_speed_mps < config.hesitation_walk_speed_mps)) | (
        turn_mask & (signals.turn_speed_deg_s < config.hesitation_turn_speed_deg_s)
    )
    return remove_short_true_runs(
        hesitation,
        seconds_to_frames(config.hesitation_min_duration_sec, signals.fps),
    ).astype(bool)


def build_segment_records(
    rule_result: PrimaryRuleResult,
    hesitation_mask: np.ndarray,
    time_sec: np.ndarray,
    fps: float,
) -> list[SegmentRecord]:
    hesitation = np.asarray(hesitation_mask, dtype=bool)
    segments: list[SegmentRecord] = []
    for start, end, label_index in labels_to_index_segments(rule_result.primary_label_index):
        source_values = rule_result.source_rule[start : end + 1].astype(str)
        unique, counts = np.unique(source_values, return_counts=True)
        source_rule = str(unique[int(np.argmax(counts))]) if len(unique) else ""
        hesitation_overlap_frames = int(np.count_nonzero(hesitation[start : end + 1]))
        frame_count = end - start + 1
        segments.append(
            SegmentRecord(
                label=PRIMARY_LABELS[label_index],
                start_frame=int(start),
                end_frame=int(end),
                start_time_sec=float(time_sec[start]),
                end_time_sec=float(time_sec[end]),
                duration_sec=float(frame_count / float(fps)),
                source_rule=source_rule,
                confidence=float(np.mean(rule_result.primary_confidence[start : end + 1])),
                hesitation_overlap_frames=hesitation_overlap_frames,
                hesitation_overlap_ratio=float(hesitation_overlap_frames / max(frame_count, 1)),
                auxiliary_overlap_labels="hesitation" if hesitation_overlap_frames else "",
            )
        )
    return segments
