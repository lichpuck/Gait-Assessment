"""Gait-event proxies derived from upstream contact masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaitEventResult:
    fps: float
    left_contact_mask: np.ndarray
    right_contact_mask: np.ndarray
    contact_mask: np.ndarray
    left_heel_strikes: np.ndarray
    right_heel_strikes: np.ndarray
    left_toe_offs: np.ndarray
    right_toe_offs: np.ndarray
    contact_stability_score: float
    warnings: tuple[str, ...]


def _detect_onsets(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return np.zeros((0,), dtype=np.int32)
    previous = np.concatenate([np.zeros(1, dtype=bool), mask_bool[:-1]])
    return np.flatnonzero(mask_bool & ~previous).astype(np.int32)


def _detect_offsets(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return np.zeros((0,), dtype=np.int32)
    previous = np.concatenate([np.zeros(1, dtype=bool), mask_bool[:-1]])
    return np.flatnonzero(~mask_bool & previous).astype(np.int32)


def _duty_cycle_score(contact_mask: np.ndarray) -> float:
    duty = float(np.mean(np.asarray(contact_mask, dtype=np.float32)))
    return max(0.0, 1.0 - min(abs(duty - 0.60) / 0.60, 1.0))


def _alternation_score(left_hs: np.ndarray, right_hs: np.ndarray) -> float:
    merged = sorted([(int(frame), "left") for frame in left_hs] + [(int(frame), "right") for frame in right_hs])
    if len(merged) < 2:
        return 0.0
    alternating = sum(1 for (_, previous), (_, current) in zip(merged[:-1], merged[1:]) if previous != current)
    return alternating / max(len(merged) - 1, 1)


def _event_frames(mask: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.asarray(fallback, dtype=np.int32)
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int32)


def detect_gait_events(
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    fps: float,
    *,
    left_heel_strike: np.ndarray | None = None,
    right_heel_strike: np.ndarray | None = None,
    left_toe_off: np.ndarray | None = None,
    right_toe_off: np.ndarray | None = None,
) -> GaitEventResult:
    left_contact_mask = np.asarray(left_contact, dtype=bool)
    right_contact_mask = np.asarray(right_contact, dtype=bool)
    if left_contact_mask.shape != right_contact_mask.shape:
        raise ValueError(
            f"left/right contact masks must have identical shape, got {left_contact_mask.shape} and {right_contact_mask.shape}"
        )

    left_heel_strikes = _event_frames(left_heel_strike, _detect_onsets(left_contact_mask))
    right_heel_strikes = _event_frames(right_heel_strike, _detect_onsets(right_contact_mask))
    left_toe_offs = _event_frames(left_toe_off, _detect_offsets(left_contact_mask))
    right_toe_offs = _event_frames(right_toe_off, _detect_offsets(right_contact_mask))

    event_balance = 1.0 - abs(len(left_heel_strikes) - len(right_heel_strikes)) / max(
        len(left_heel_strikes) + len(right_heel_strikes),
        1,
    )
    contact_stability_score = float(
        np.mean(
            [
                _alternation_score(left_heel_strikes, right_heel_strikes),
                _duty_cycle_score(left_contact_mask),
                _duty_cycle_score(right_contact_mask),
                event_balance,
            ]
        )
    )
    warnings: list[str] = []
    if len(left_heel_strikes) == 0 or len(right_heel_strikes) == 0:
        warnings.append("At least one foot is missing contact onsets.")
    if len(left_heel_strikes) + len(right_heel_strikes) < 2:
        warnings.append("Too few contact onsets were available for stable gait-event proxies.")
    if contact_stability_score < 0.45:
        warnings.append(f"Low contact stability score ({contact_stability_score:.3f}).")

    return GaitEventResult(
        fps=float(fps),
        left_contact_mask=left_contact_mask.astype(bool),
        right_contact_mask=right_contact_mask.astype(bool),
        contact_mask=np.column_stack([left_contact_mask, right_contact_mask]).astype(bool),
        left_heel_strikes=left_heel_strikes.astype(np.int32),
        right_heel_strikes=right_heel_strikes.astype(np.int32),
        left_toe_offs=left_toe_offs.astype(np.int32),
        right_toe_offs=right_toe_offs.astype(np.int32),
        contact_stability_score=contact_stability_score,
        warnings=tuple(warnings),
    )
