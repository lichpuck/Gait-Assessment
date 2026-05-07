"""Walk and turn descriptor rules for F_Description."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .io_utils import DescriptionSequence


WALK_PERCENTILE_METRICS = (
    "step_time_variability_cv_percent",
    "stride_time_variability_cv_percent",
    "step_length_asymmetry_percent",
    "stride_length_asymmetry_percent",
    "double_support_percent",
    "trunk_ml_sway_m",
    "arm_swing_amplitude_mean_deg",
    "arm_swing_asymmetry_percent",
    "pelvis_rotation_rom_deg",
)

TURN_PERCENTILE_METRICS = (
    "turn_duration_sec",
    "turn_step_count",
    "mean_turn_angular_velocity_deg_s",
    "turn_angular_velocity_variability_cv_percent",
    "turn_path_radius_m",
    "turn_path_compactness_deg_per_m",
    "en_bloc_index",
    "trunk_lateral_lean_during_turn_deg",
    "pelvis_ml_excursion_during_turn_m",
)


@dataclass(frozen=True)
class DescriptorReference:
    walk_metric_values: dict[str, np.ndarray]
    turn_metric_values: dict[str, np.ndarray]


def _finite_array(values: pd.Series | np.ndarray | list[object]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _weighted_mean(values: list[object], weights: list[float]) -> float | None:
    valid_pairs: list[tuple[float, float]] = []
    for raw_value, raw_weight in zip(values, weights):
        try:
            value = float(raw_value)
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or not np.isfinite(weight):
            continue
        valid_pairs.append((value, weight))
    if not valid_pairs:
        return None
    valid_values = np.asarray([value for value, _ in valid_pairs], dtype=float)
    valid_weights = np.asarray([weight for _, weight in valid_pairs], dtype=float)
    if float(np.sum(valid_weights)) <= 0.0:
        return float(np.mean(valid_values))
    return float(np.average(valid_values, weights=valid_weights))


def _percentile_rank(value: float | None, reference_values: np.ndarray, *, invert: bool = False) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    if reference_values.size == 0:
        return None
    rank = float(np.mean(reference_values <= float(value)))
    return 1.0 - rank if invert else rank


def _average_optional(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def _descriptor_entry(
    *,
    label: str,
    zh_phrase: str,
    policy: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "label": label,
        "zh_phrase": zh_phrase,
        "policy": policy,
        "evidence": evidence,
    }


def build_reference(walk_df: pd.DataFrame, turn_df: pd.DataFrame) -> DescriptorReference:
    walk_metric_values: dict[str, np.ndarray] = {}
    turn_metric_values: dict[str, np.ndarray] = {}
    for metric_name in WALK_PERCENTILE_METRICS:
        walk_metric_values[metric_name] = _finite_array(walk_df[metric_name]) if metric_name in walk_df.columns else np.zeros((0,), dtype=float)
    for metric_name in TURN_PERCENTILE_METRICS:
        turn_metric_values[metric_name] = _finite_array(turn_df[metric_name]) if metric_name in turn_df.columns else np.zeros((0,), dtype=float)
    return DescriptorReference(
        walk_metric_values=walk_metric_values,
        turn_metric_values=turn_metric_values,
    )


def _aggregate_segment_metrics(segments: list[dict[str, object]], metric_names: tuple[str, ...]) -> dict[str, float | None]:
    frame = pd.DataFrame(segments)
    weights = [float(value) if pd.notna(value) else 0.0 for value in frame.get("duration_sec", pd.Series(dtype=float)).tolist()]
    metrics: dict[str, float | None] = {}
    for metric_name in metric_names:
        if metric_name not in frame.columns:
            metrics[metric_name] = None
            continue
        metrics[metric_name] = _weighted_mean(frame[metric_name].tolist(), weights)
    return metrics


def _walk_descriptor_block(metrics: dict[str, float | None], reference: DescriptorReference) -> tuple[dict[str, dict[str, object]], list[str]]:
    descriptors: dict[str, dict[str, object]] = {}
    omissions: list[str] = []

    gait_speed = metrics.get("gait_speed_mps")
    if gait_speed is None:
        omissions.append("pace")
    elif gait_speed < 0.4:
        descriptors["pace"] = _descriptor_entry(
            label="very_slow",
            zh_phrase="速度很慢",
            policy="fixed",
            evidence={"metric_name": "gait_speed_mps", "metric_value": gait_speed, "thresholds": [0.4, 0.8, 1.1]},
        )
    elif gait_speed < 0.8:
        descriptors["pace"] = _descriptor_entry(
            label="slow",
            zh_phrase="速度偏慢",
            policy="fixed",
            evidence={"metric_name": "gait_speed_mps", "metric_value": gait_speed, "thresholds": [0.4, 0.8, 1.1]},
        )
    elif gait_speed < 1.1:
        descriptors["pace"] = _descriptor_entry(
            label="moderate",
            zh_phrase="速度中等",
            policy="fixed",
            evidence={"metric_name": "gait_speed_mps", "metric_value": gait_speed, "thresholds": [0.4, 0.8, 1.1]},
        )
    else:
        descriptors["pace"] = _descriptor_entry(
            label="fast",
            zh_phrase="速度较快",
            policy="fixed",
            evidence={"metric_name": "gait_speed_mps", "metric_value": gait_speed, "thresholds": [0.4, 0.8, 1.1]},
        )

    cadence = metrics.get("cadence_steps_per_min")
    if cadence is None:
        omissions.append("cadence")
    elif cadence < 100.0:
        descriptors["cadence"] = _descriptor_entry(
            label="low",
            zh_phrase="步频偏低",
            policy="fixed",
            evidence={"metric_name": "cadence_steps_per_min", "metric_value": cadence, "thresholds": [100.0, 130.0]},
        )
    elif cadence <= 130.0:
        descriptors["cadence"] = _descriptor_entry(
            label="moderate",
            zh_phrase="步频中等",
            policy="fixed",
            evidence={"metric_name": "cadence_steps_per_min", "metric_value": cadence, "thresholds": [100.0, 130.0]},
        )
    else:
        descriptors["cadence"] = _descriptor_entry(
            label="high",
            zh_phrase="步频偏高",
            policy="fixed",
            evidence={"metric_name": "cadence_steps_per_min", "metric_value": cadence, "thresholds": [100.0, 130.0]},
        )

    step_length = metrics.get("step_length_mean_m")
    if step_length is None:
        omissions.append("step_amplitude")
    elif step_length < 0.20:
        descriptors["step_amplitude"] = _descriptor_entry(
            label="very_short",
            zh_phrase="步幅很短",
            policy="fixed",
            evidence={"metric_name": "step_length_mean_m", "metric_value": step_length, "thresholds": [0.2, 0.3, 0.4]},
        )
    elif step_length < 0.30:
        descriptors["step_amplitude"] = _descriptor_entry(
            label="short",
            zh_phrase="步幅偏短",
            policy="fixed",
            evidence={"metric_name": "step_length_mean_m", "metric_value": step_length, "thresholds": [0.2, 0.3, 0.4]},
        )
    elif step_length < 0.40:
        descriptors["step_amplitude"] = _descriptor_entry(
            label="medium",
            zh_phrase="步幅中等",
            policy="fixed",
            evidence={"metric_name": "step_length_mean_m", "metric_value": step_length, "thresholds": [0.2, 0.3, 0.4]},
        )
    else:
        descriptors["step_amplitude"] = _descriptor_entry(
            label="long",
            zh_phrase="步幅较大",
            policy="fixed",
            evidence={"metric_name": "step_length_mean_m", "metric_value": step_length, "thresholds": [0.2, 0.3, 0.4]},
        )

    rhythm_score = _average_optional(
        [
            _percentile_rank(metrics.get("step_time_variability_cv_percent"), reference.walk_metric_values["step_time_variability_cv_percent"]),
            _percentile_rank(metrics.get("stride_time_variability_cv_percent"), reference.walk_metric_values["stride_time_variability_cv_percent"]),
        ]
    )
    if rhythm_score is None:
        omissions.append("rhythm")
    elif rhythm_score < 0.40:
        descriptors["rhythm"] = _descriptor_entry(
            label="regular",
            zh_phrase="节律较稳",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_variability",
                "metric_value": rhythm_score,
                "source_metrics": {
                    "step_time_variability_cv_percent": metrics.get("step_time_variability_cv_percent"),
                    "stride_time_variability_cv_percent": metrics.get("stride_time_variability_cv_percent"),
                },
            },
        )
    elif rhythm_score < 0.75:
        descriptors["rhythm"] = _descriptor_entry(
            label="mildly_variable",
            zh_phrase="节律略有波动",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_variability",
                "metric_value": rhythm_score,
                "source_metrics": {
                    "step_time_variability_cv_percent": metrics.get("step_time_variability_cv_percent"),
                    "stride_time_variability_cv_percent": metrics.get("stride_time_variability_cv_percent"),
                },
            },
        )
    else:
        descriptors["rhythm"] = _descriptor_entry(
            label="irregular",
            zh_phrase="节律不规则",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_variability",
                "metric_value": rhythm_score,
                "source_metrics": {
                    "step_time_variability_cv_percent": metrics.get("step_time_variability_cv_percent"),
                    "stride_time_variability_cv_percent": metrics.get("stride_time_variability_cv_percent"),
                },
            },
        )

    asymmetry_score = _average_optional(
        [
            _percentile_rank(metrics.get("step_length_asymmetry_percent"), reference.walk_metric_values["step_length_asymmetry_percent"]),
            _percentile_rank(metrics.get("stride_length_asymmetry_percent"), reference.walk_metric_values["stride_length_asymmetry_percent"]),
        ]
    )
    if asymmetry_score is None:
        omissions.append("asymmetry")
    elif asymmetry_score < 0.40:
        descriptors["asymmetry"] = _descriptor_entry(
            label="symmetric",
            zh_phrase="左右较对称",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_asymmetry",
                "metric_value": asymmetry_score,
                "source_metrics": {
                    "step_length_asymmetry_percent": metrics.get("step_length_asymmetry_percent"),
                    "stride_length_asymmetry_percent": metrics.get("stride_length_asymmetry_percent"),
                },
            },
        )
    elif asymmetry_score < 0.75:
        descriptors["asymmetry"] = _descriptor_entry(
            label="mild",
            zh_phrase="存在轻度不对称",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_asymmetry",
                "metric_value": asymmetry_score,
                "source_metrics": {
                    "step_length_asymmetry_percent": metrics.get("step_length_asymmetry_percent"),
                    "stride_length_asymmetry_percent": metrics.get("stride_length_asymmetry_percent"),
                },
            },
        )
    else:
        descriptors["asymmetry"] = _descriptor_entry(
            label="marked",
            zh_phrase="存在明显不对称",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_asymmetry",
                "metric_value": asymmetry_score,
                "source_metrics": {
                    "step_length_asymmetry_percent": metrics.get("step_length_asymmetry_percent"),
                    "stride_length_asymmetry_percent": metrics.get("stride_length_asymmetry_percent"),
                },
            },
        )

    trunk_flexion = metrics.get("trunk_flexion_mean_deg")
    if trunk_flexion is None:
        omissions.append("posture")
    elif trunk_flexion < 5.0:
        descriptors["posture"] = _descriptor_entry(
            label="upright",
            zh_phrase="姿势较直立",
            policy="fixed",
            evidence={"metric_name": "trunk_flexion_mean_deg", "metric_value": trunk_flexion, "thresholds": [5.0, 15.0]},
        )
    elif trunk_flexion < 15.0:
        descriptors["posture"] = _descriptor_entry(
            label="mild_forward_lean",
            zh_phrase="轻度前倾",
            policy="fixed",
            evidence={"metric_name": "trunk_flexion_mean_deg", "metric_value": trunk_flexion, "thresholds": [5.0, 15.0]},
        )
    else:
        descriptors["posture"] = _descriptor_entry(
            label="stooped",
            zh_phrase="前倾较明显",
            policy="fixed",
            evidence={"metric_name": "trunk_flexion_mean_deg", "metric_value": trunk_flexion, "thresholds": [5.0, 15.0]},
        )

    stability_score = _average_optional(
        [
            _percentile_rank(metrics.get("double_support_percent"), reference.walk_metric_values["double_support_percent"]),
            _percentile_rank(metrics.get("trunk_ml_sway_m"), reference.walk_metric_values["trunk_ml_sway_m"]),
        ]
    )
    if stability_score is None:
        omissions.append("stability")
    elif stability_score < 0.40:
        descriptors["stability"] = _descriptor_entry(
            label="stable",
            zh_phrase="稳定性较好",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_stability",
                "metric_value": stability_score,
                "source_metrics": {
                    "double_support_percent": metrics.get("double_support_percent"),
                    "trunk_ml_sway_m": metrics.get("trunk_ml_sway_m"),
                },
            },
        )
    elif stability_score < 0.75:
        descriptors["stability"] = _descriptor_entry(
            label="mild_reduction",
            zh_phrase="稳定性略下降",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_stability",
                "metric_value": stability_score,
                "source_metrics": {
                    "double_support_percent": metrics.get("double_support_percent"),
                    "trunk_ml_sway_m": metrics.get("trunk_ml_sway_m"),
                },
            },
        )
    else:
        descriptors["stability"] = _descriptor_entry(
            label="reduced",
            zh_phrase="稳定性下降",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_stability",
                "metric_value": stability_score,
                "source_metrics": {
                    "double_support_percent": metrics.get("double_support_percent"),
                    "trunk_ml_sway_m": metrics.get("trunk_ml_sway_m"),
                },
            },
        )

    coordination_score = _average_optional(
        [
            _percentile_rank(metrics.get("arm_swing_amplitude_mean_deg"), reference.walk_metric_values["arm_swing_amplitude_mean_deg"], invert=True),
            _percentile_rank(metrics.get("arm_swing_asymmetry_percent"), reference.walk_metric_values["arm_swing_asymmetry_percent"]),
            _percentile_rank(metrics.get("pelvis_rotation_rom_deg"), reference.walk_metric_values["pelvis_rotation_rom_deg"], invert=True),
        ]
    )
    if coordination_score is None:
        omissions.append("coordination")
    elif coordination_score < 0.40:
        descriptors["coordination"] = _descriptor_entry(
            label="preserved",
            zh_phrase="协调性尚可",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_coordination",
                "metric_value": coordination_score,
                "source_metrics": {
                    "arm_swing_amplitude_mean_deg": metrics.get("arm_swing_amplitude_mean_deg"),
                    "arm_swing_asymmetry_percent": metrics.get("arm_swing_asymmetry_percent"),
                    "pelvis_rotation_rom_deg": metrics.get("pelvis_rotation_rom_deg"),
                },
            },
        )
    elif coordination_score < 0.75:
        descriptors["coordination"] = _descriptor_entry(
            label="mild_reduction",
            zh_phrase="协调性略下降",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_coordination",
                "metric_value": coordination_score,
                "source_metrics": {
                    "arm_swing_amplitude_mean_deg": metrics.get("arm_swing_amplitude_mean_deg"),
                    "arm_swing_asymmetry_percent": metrics.get("arm_swing_asymmetry_percent"),
                    "pelvis_rotation_rom_deg": metrics.get("pelvis_rotation_rom_deg"),
                },
            },
        )
    else:
        descriptors["coordination"] = _descriptor_entry(
            label="reduced",
            zh_phrase="协调性下降",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_walk_coordination",
                "metric_value": coordination_score,
                "source_metrics": {
                    "arm_swing_amplitude_mean_deg": metrics.get("arm_swing_amplitude_mean_deg"),
                    "arm_swing_asymmetry_percent": metrics.get("arm_swing_asymmetry_percent"),
                    "pelvis_rotation_rom_deg": metrics.get("pelvis_rotation_rom_deg"),
                },
            },
        )
    return descriptors, omissions


def _turn_descriptor_block(metrics: dict[str, float | None], reference: DescriptorReference) -> tuple[dict[str, dict[str, object]], list[str]]:
    descriptors: dict[str, dict[str, object]] = {}
    omissions: list[str] = []

    turn_angle = metrics.get("turn_angle_deg")
    if turn_angle is None:
        omissions.append("extent")
    elif turn_angle < 60.0:
        descriptors["extent"] = _descriptor_entry(
            label="small",
            zh_phrase="转身角度较小",
            policy="fixed",
            evidence={"metric_name": "turn_angle_deg", "metric_value": turn_angle, "thresholds": [60.0, 120.0]},
        )
    elif turn_angle < 120.0:
        descriptors["extent"] = _descriptor_entry(
            label="medium",
            zh_phrase="转身角度中等",
            policy="fixed",
            evidence={"metric_name": "turn_angle_deg", "metric_value": turn_angle, "thresholds": [60.0, 120.0]},
        )
    else:
        descriptors["extent"] = _descriptor_entry(
            label="large",
            zh_phrase="转身角度较大",
            policy="fixed",
            evidence={"metric_name": "turn_angle_deg", "metric_value": turn_angle, "thresholds": [60.0, 120.0]},
        )

    turn_speed = metrics.get("mean_turn_angular_velocity_deg_s")
    if turn_speed is None:
        omissions.append("speed")
    elif turn_speed < 20.0:
        descriptors["speed"] = _descriptor_entry(
            label="slow",
            zh_phrase="转身角速度偏慢",
            policy="fixed",
            evidence={"metric_name": "mean_turn_angular_velocity_deg_s", "metric_value": turn_speed, "thresholds": [20.0, 45.0]},
        )
    elif turn_speed < 45.0:
        descriptors["speed"] = _descriptor_entry(
            label="moderate",
            zh_phrase="转身角速度中等",
            policy="fixed",
            evidence={"metric_name": "mean_turn_angular_velocity_deg_s", "metric_value": turn_speed, "thresholds": [20.0, 45.0]},
        )
    else:
        descriptors["speed"] = _descriptor_entry(
            label="fast",
            zh_phrase="转身角速度较快",
            policy="fixed",
            evidence={"metric_name": "mean_turn_angular_velocity_deg_s", "metric_value": turn_speed, "thresholds": [20.0, 45.0]},
        )

    efficiency_score = _average_optional(
        [
            _percentile_rank(metrics.get("turn_duration_sec"), reference.turn_metric_values["turn_duration_sec"]),
            _percentile_rank(metrics.get("turn_step_count"), reference.turn_metric_values["turn_step_count"]),
        ]
    )
    if efficiency_score is None:
        omissions.append("efficiency")
    elif efficiency_score < 0.40:
        descriptors["efficiency"] = _descriptor_entry(
            label="efficient",
            zh_phrase="耗时和步数控制较好",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_efficiency",
                "metric_value": efficiency_score,
                "source_metrics": {
                    "turn_duration_sec": metrics.get("turn_duration_sec"),
                    "turn_step_count": metrics.get("turn_step_count"),
                },
            },
        )
    elif efficiency_score < 0.75:
        descriptors["efficiency"] = _descriptor_entry(
            label="mild_cost",
            zh_phrase="耗时和步数略增",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_efficiency",
                "metric_value": efficiency_score,
                "source_metrics": {
                    "turn_duration_sec": metrics.get("turn_duration_sec"),
                    "turn_step_count": metrics.get("turn_step_count"),
                },
            },
        )
    else:
        descriptors["efficiency"] = _descriptor_entry(
            label="costly",
            zh_phrase="耗时较长且步数偏多",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_efficiency",
                "metric_value": efficiency_score,
                "source_metrics": {
                    "turn_duration_sec": metrics.get("turn_duration_sec"),
                    "turn_step_count": metrics.get("turn_step_count"),
                },
            },
        )

    hesitation = metrics.get("pre_turn_hesitation_time_sec")
    if hesitation is None:
        omissions.append("hesitation")
    elif hesitation <= 0.0:
        descriptors["hesitation"] = _descriptor_entry(
            label="none",
            zh_phrase="起转前无明显迟疑",
            policy="fixed",
            evidence={"metric_name": "pre_turn_hesitation_time_sec", "metric_value": hesitation, "thresholds": [0.2, 0.5]},
        )
    elif hesitation < 0.5:
        descriptors["hesitation"] = _descriptor_entry(
            label="mild",
            zh_phrase="起转前略有迟疑",
            policy="fixed",
            evidence={"metric_name": "pre_turn_hesitation_time_sec", "metric_value": hesitation, "thresholds": [0.2, 0.5]},
        )
    else:
        descriptors["hesitation"] = _descriptor_entry(
            label="marked",
            zh_phrase="起转前迟疑较明显",
            policy="fixed",
            evidence={"metric_name": "pre_turn_hesitation_time_sec", "metric_value": hesitation, "thresholds": [0.2, 0.5]},
        )

    smoothness_score = _percentile_rank(
        metrics.get("turn_angular_velocity_variability_cv_percent"),
        reference.turn_metric_values["turn_angular_velocity_variability_cv_percent"],
    )
    if smoothness_score is None:
        omissions.append("smoothness")
    elif smoothness_score < 0.40:
        descriptors["smoothness"] = _descriptor_entry(
            label="smooth",
            zh_phrase="转身过程较平稳",
            policy="subset_percentile",
            evidence={
                "metric_name": "turn_angular_velocity_variability_cv_percent",
                "metric_value": metrics.get("turn_angular_velocity_variability_cv_percent"),
                "rank": smoothness_score,
            },
        )
    elif smoothness_score < 0.75:
        descriptors["smoothness"] = _descriptor_entry(
            label="mild_fragmentation",
            zh_phrase="转身略显分段",
            policy="subset_percentile",
            evidence={
                "metric_name": "turn_angular_velocity_variability_cv_percent",
                "metric_value": metrics.get("turn_angular_velocity_variability_cv_percent"),
                "rank": smoothness_score,
            },
        )
    else:
        descriptors["smoothness"] = _descriptor_entry(
            label="fragmented",
            zh_phrase="转身分段化明显",
            policy="subset_percentile",
            evidence={
                "metric_name": "turn_angular_velocity_variability_cv_percent",
                "metric_value": metrics.get("turn_angular_velocity_variability_cv_percent"),
                "rank": smoothness_score,
            },
        )

    compactness_score = _average_optional(
        [
            _percentile_rank(metrics.get("turn_path_compactness_deg_per_m"), reference.turn_metric_values["turn_path_compactness_deg_per_m"], invert=True),
            _percentile_rank(metrics.get("turn_path_radius_m"), reference.turn_metric_values["turn_path_radius_m"]),
        ]
    )
    if compactness_score is None:
        omissions.append("compactness")
    elif compactness_score < 0.40:
        descriptors["compactness"] = _descriptor_entry(
            label="compact",
            zh_phrase="路径较紧凑",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_compactness",
                "metric_value": compactness_score,
                "source_metrics": {
                    "turn_path_compactness_deg_per_m": metrics.get("turn_path_compactness_deg_per_m"),
                    "turn_path_radius_m": metrics.get("turn_path_radius_m"),
                },
            },
        )
    elif compactness_score < 0.75:
        descriptors["compactness"] = _descriptor_entry(
            label="moderate",
            zh_phrase="路径紧凑性一般",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_compactness",
                "metric_value": compactness_score,
                "source_metrics": {
                    "turn_path_compactness_deg_per_m": metrics.get("turn_path_compactness_deg_per_m"),
                    "turn_path_radius_m": metrics.get("turn_path_radius_m"),
                },
            },
        )
    else:
        descriptors["compactness"] = _descriptor_entry(
            label="dispersed",
            zh_phrase="路径较分散",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_compactness",
                "metric_value": compactness_score,
                "source_metrics": {
                    "turn_path_compactness_deg_per_m": metrics.get("turn_path_compactness_deg_per_m"),
                    "turn_path_radius_m": metrics.get("turn_path_radius_m"),
                },
            },
        )

    en_bloc = metrics.get("en_bloc_index")
    if en_bloc is None:
        omissions.append("reorientation")
    elif en_bloc < 0.55:
        descriptors["reorientation"] = _descriptor_entry(
            label="sequential",
            zh_phrase="分节式重定向较明显",
            policy="fixed",
            evidence={
                "metric_name": "en_bloc_index",
                "metric_value": en_bloc,
                "thresholds": [0.55, 0.85],
                "source_metrics": {
                    "head_trunk_pelvis_reorientation_delay_sec": metrics.get("head_trunk_pelvis_reorientation_delay_sec"),
                    "trunk_pelvis_reorientation_delay_sec": metrics.get("trunk_pelvis_reorientation_delay_sec"),
                    "head_trunk_reorientation_delay_sec": metrics.get("head_trunk_reorientation_delay_sec"),
                },
            },
        )
    elif en_bloc < 0.85:
        descriptors["reorientation"] = _descriptor_entry(
            label="partially_en_bloc",
            zh_phrase="重定向部分趋于整体化",
            policy="fixed",
            evidence={
                "metric_name": "en_bloc_index",
                "metric_value": en_bloc,
                "thresholds": [0.55, 0.85],
                "source_metrics": {
                    "head_trunk_pelvis_reorientation_delay_sec": metrics.get("head_trunk_pelvis_reorientation_delay_sec"),
                    "trunk_pelvis_reorientation_delay_sec": metrics.get("trunk_pelvis_reorientation_delay_sec"),
                    "head_trunk_reorientation_delay_sec": metrics.get("head_trunk_reorientation_delay_sec"),
                },
            },
        )
    else:
        descriptors["reorientation"] = _descriptor_entry(
            label="en_bloc",
            zh_phrase="重定向更趋整体化",
            policy="fixed",
            evidence={
                "metric_name": "en_bloc_index",
                "metric_value": en_bloc,
                "thresholds": [0.55, 0.85],
                "source_metrics": {
                    "head_trunk_pelvis_reorientation_delay_sec": metrics.get("head_trunk_pelvis_reorientation_delay_sec"),
                    "trunk_pelvis_reorientation_delay_sec": metrics.get("trunk_pelvis_reorientation_delay_sec"),
                    "head_trunk_reorientation_delay_sec": metrics.get("head_trunk_reorientation_delay_sec"),
                },
            },
        )

    control_score = _average_optional(
        [
            _percentile_rank(metrics.get("trunk_lateral_lean_during_turn_deg"), reference.turn_metric_values["trunk_lateral_lean_during_turn_deg"]),
            _percentile_rank(metrics.get("pelvis_ml_excursion_during_turn_m"), reference.turn_metric_values["pelvis_ml_excursion_during_turn_m"]),
        ]
    )
    if control_score is None:
        omissions.append("postural_control")
    elif control_score < 0.40:
        descriptors["postural_control"] = _descriptor_entry(
            label="stable",
            zh_phrase="姿势控制较稳",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_postural_control",
                "metric_value": control_score,
                "source_metrics": {
                    "trunk_lateral_lean_during_turn_deg": metrics.get("trunk_lateral_lean_during_turn_deg"),
                    "pelvis_ml_excursion_during_turn_m": metrics.get("pelvis_ml_excursion_during_turn_m"),
                },
            },
        )
    elif control_score < 0.75:
        descriptors["postural_control"] = _descriptor_entry(
            label="mild_reduction",
            zh_phrase="姿势控制略不足",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_postural_control",
                "metric_value": control_score,
                "source_metrics": {
                    "trunk_lateral_lean_during_turn_deg": metrics.get("trunk_lateral_lean_during_turn_deg"),
                    "pelvis_ml_excursion_during_turn_m": metrics.get("pelvis_ml_excursion_during_turn_m"),
                },
            },
        )
    else:
        descriptors["postural_control"] = _descriptor_entry(
            label="reduced",
            zh_phrase="姿势控制不足",
            policy="subset_percentile",
            evidence={
                "metric_name": "combined_turn_postural_control",
                "metric_value": control_score,
                "source_metrics": {
                    "trunk_lateral_lean_during_turn_deg": metrics.get("trunk_lateral_lean_during_turn_deg"),
                    "pelvis_ml_excursion_during_turn_m": metrics.get("pelvis_ml_excursion_during_turn_m"),
                },
            },
        )
    return descriptors, omissions


def _summary_sentence(prefix: str, descriptors: dict[str, dict[str, object]], preferred_keys: tuple[str, ...]) -> str:
    phrases = [descriptors[key]["zh_phrase"] for key in preferred_keys if key in descriptors and descriptors[key].get("zh_phrase")]
    if not phrases:
        return ""
    return f"{prefix}{'、'.join(str(item) for item in phrases)}。"


def derive_sequence_profile(sequence: DescriptionSequence, reference: DescriptorReference) -> dict[str, object]:
    walk_metric_names = (
        "gait_speed_mps",
        "cadence_steps_per_min",
        "step_length_mean_m",
        "step_time_variability_cv_percent",
        "stride_time_variability_cv_percent",
        "step_length_asymmetry_percent",
        "stride_length_asymmetry_percent",
        "trunk_flexion_mean_deg",
        "double_support_percent",
        "trunk_ml_sway_m",
        "arm_swing_amplitude_mean_deg",
        "arm_swing_asymmetry_percent",
        "pelvis_rotation_rom_deg",
    )
    turn_metric_names = (
        "turn_angle_deg",
        "turn_duration_sec",
        "turn_step_count",
        "pre_turn_hesitation_time_sec",
        "mean_turn_angular_velocity_deg_s",
        "turn_angular_velocity_variability_cv_percent",
        "turn_path_radius_m",
        "turn_path_compactness_deg_per_m",
        "head_trunk_pelvis_reorientation_delay_sec",
        "trunk_pelvis_reorientation_delay_sec",
        "head_trunk_reorientation_delay_sec",
        "en_bloc_index",
        "trunk_lateral_lean_during_turn_deg",
        "pelvis_ml_excursion_during_turn_m",
    )

    walk_metrics = _aggregate_segment_metrics(sequence.walk_segments, walk_metric_names)
    turn_metrics = _aggregate_segment_metrics(sequence.turn_segments, turn_metric_names)
    walk_descriptors, walk_omissions = _walk_descriptor_block(walk_metrics, reference) if sequence.walk_segments else ({}, [])
    turn_descriptors, turn_omissions = _turn_descriptor_block(turn_metrics, reference) if sequence.turn_segments else ({}, [])

    walk_summary = _summary_sentence(
        "步行表现为",
        walk_descriptors,
        ("pace", "step_amplitude", "rhythm", "asymmetry", "posture", "stability", "coordination"),
    )
    turn_summary = _summary_sentence(
        "转身表现为",
        turn_descriptors,
        ("extent", "speed", "efficiency", "hesitation", "smoothness", "compactness", "reorientation", "postural_control"),
    )
    summary_parts = [part for part in (walk_summary, turn_summary) if part]

    return {
        "summary_zh": " ".join(summary_parts).strip(),
        "walk": {
            "available": bool(sequence.walk_segments),
            "segment_count": len(sequence.walk_segments),
            "descriptors": walk_descriptors,
            "omitted_descriptors": walk_omissions,
        },
        "turn": {
            "available": bool(sequence.turn_segments),
            "segment_count": len(sequence.turn_segments),
            "descriptors": turn_descriptors,
            "omitted_descriptors": turn_omissions,
        },
    }
