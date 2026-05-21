from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.G_Animation.config import AnimationConfig
from scripts.G_Animation.io_utils import load_animation_sequence
from scripts.G_Animation.metrics import build_animation_metrics
from scripts.G_Animation.render import _is_abnormal_clause, _split_summary_clauses


class GAnimationIndependentTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_npz(self, path: Path, **arrays: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def _write_minimal_artifacts(self, root: Path) -> AnimationConfig:
        config = AnimationConfig(
            input_b_dir=root / "B_Canonicalization",
            input_c_dir=root / "C_Representation",
            input_d_dir=root / "D_Segmentation",
            input_f_dir=root / "F_Description",
            output_dir=root / "G_Animation",
        )
        subset = "TestSubset"
        subject_id = "SUB01"
        trial_id = "TRIAL01"
        stem = f"{subset}__{subject_id}__{trial_id}"
        frame_count = 6
        fps = 30.0
        time_s = np.arange(frame_count, dtype=np.float32) / fps
        joints_can = np.zeros((frame_count, 24, 3), dtype=np.float32)
        joints_can[:, :, 0] = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)[:, None]
        joints_can[:, :, 2] = 1.0
        trans_can = np.stack(
            [np.linspace(0.0, 1.0, frame_count, dtype=np.float32), np.zeros(frame_count, dtype=np.float32), np.ones(frame_count, dtype=np.float32)],
            axis=1,
        )

        self._write_npz(
            config.input_b_dir / subset / f"{stem}.npz",
            pose_raw=np.zeros((frame_count, 72), dtype=np.float32),
            trans_raw=np.zeros((frame_count, 3), dtype=np.float32),
            joints_can=joints_can,
            trans_can=trans_can,
        )
        self._write_json(
            config.input_b_dir / subset / f"{stem}.json",
            {
                "subset": subset,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "R_global": np.eye(3, dtype=np.float32).tolist(),
                "metadata": {
                    "fps": fps,
                    "audition_metadata": {"R_total": np.eye(3, dtype=np.float32).tolist()},
                },
            },
        )

        self._write_npz(
            config.input_c_dir / subset / f"{stem}.npz",
            fps=np.array(fps, dtype=np.float32),
            time_s=time_s,
            frame_index=np.arange(frame_count, dtype=np.int32),
            joints_can=joints_can,
            root_pos_m=trans_can,
            root_speed_xy_mps=np.linspace(0.2, 0.8, frame_count, dtype=np.float32),
            root_acceleration_mps2=np.stack(
                [np.linspace(0.0, 0.5, frame_count, dtype=np.float32), np.zeros(frame_count, dtype=np.float32), np.zeros(frame_count, dtype=np.float32)],
                axis=1,
            ),
            left_foot_speed_mps=np.array([0.1, 0.6, 0.2, 0.7, 0.3, 0.8], dtype=np.float32),
            right_foot_speed_mps=np.array([0.8, 0.3, 0.7, 0.2, 0.6, 0.1], dtype=np.float32),
            left_gait_phase=np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8),
            right_gait_phase=np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8),
            trunk_lean_angle_deg=np.array([1.0, 2.0, 3.0, 2.5, 1.5, 0.5], dtype=np.float32),
            pelvis_roll_deg=np.array([-1.0, 1.5, -2.0, 2.5, -3.0, 3.5], dtype=np.float32),
            valid_frame_mask=np.ones(frame_count, dtype=bool),
            representation_quality_score=np.ones(frame_count, dtype=np.float32),
        )
        self._write_json(
            config.input_c_dir / subset / f"{stem}.json",
            {
                "module": "C_Representation",
                "version": "2.0.0",
                "sequence": {
                    "subset": subset,
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                },
            },
        )

        self._write_json(
            config.input_d_dir / subset / f"{stem}.json",
            {
                "sequence": {
                    "subset": subset,
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "num_frames": frame_count,
                },
                "segments": [
                    {"segment_id": "seg_0001", "label": "walk", "start_frame": 0, "end_frame": 2},
                    {"segment_id": "seg_0002", "label": "turn", "start_frame": 3, "end_frame": 5},
                ],
            },
        )

        self._write_json(
            config.input_f_dir / subset / f"{stem}.json",
            {
                "sequence": {
                    "subset": subset,
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                },
                "description": {"text_summary_zh": ""},
            },
        )
        return config

    def test_load_animation_sequence_accepts_empty_summary_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            config = self._write_minimal_artifacts(Path(temp_root_str))

            loaded = load_animation_sequence("TestSubset", "SUB01", "TRIAL01", config=config)

            self.assertEqual(loaded.summary_text_zh, "")
            self.assertEqual(loaded.num_frames, 6)
            self.assertEqual(loaded.primary_label_by_frame.tolist(), ["walk", "walk", "walk", "turn", "turn", "turn"])

    def test_build_animation_metrics_returns_six_finite_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            config = self._write_minimal_artifacts(Path(temp_root_str))
            loaded = load_animation_sequence("TestSubset", "SUB01", "TRIAL01", config=config)

            metrics = build_animation_metrics(loaded, config=config)

            self.assertEqual(
                [item.key for item in metrics.series],
                ["stability", "balance", "symmetry", "coordination", "mobility", "control"],
            )
            self.assertTrue(all(len(item.values) == loaded.num_frames for item in metrics.series))
            self.assertTrue(all(np.all(np.isfinite(item.values)) for item in metrics.series))
            self.assertEqual(
                [item.title for item in metrics.series],
                [
                    "稳定性：躯干倾斜角度",
                    "平衡性：骨盆侧倾角度",
                    "对称性：左右脚相位差",
                    "协调性：左右脚速度一致性",
                    "移动能力：前进速度",
                    "控制能力：运动平滑度",
                ],
            )
            symmetry_series = next(item for item in metrics.series if item.key == "symmetry")
            np.testing.assert_array_equal(symmetry_series.values, np.ones(loaded.num_frames, dtype=np.float32))

    def test_split_summary_clauses_preserves_header_and_dimensions(self) -> None:
        header, clauses = _split_summary_clauses(
            "六维运动功能等级：稳定性轻度异常(1.11)、平衡正常(0.00)、对称性中度异常(2.00)。"
        )

        self.assertEqual(header, "六维运动功能等级：")
        self.assertEqual(
            clauses,
            (
                "稳定性轻度异常(1.11)",
                "平衡正常(0.00)",
                "对称性中度异常(2.00)",
            ),
        )

    def test_abnormal_clause_detection_is_narrow(self) -> None:
        self.assertTrue(_is_abnormal_clause("运动控制重度异常(3.00)"))
        self.assertTrue(_is_abnormal_clause("节律不规则"))
        self.assertFalse(_is_abnormal_clause("平衡正常(0.00)"))


if __name__ == "__main__":
    unittest.main()