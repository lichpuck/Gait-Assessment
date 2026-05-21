from __future__ import annotations

import unittest

import numpy as np

from scripts.B_Canonicalization.config import CanonicalizationConfig, SMPL_MODEL_PATH
from scripts.B_Canonicalization.joints_schema import ROBUST_SCALE_BONE_PAIRS
from scripts.B_Canonicalization.smpl_forward import load_smpl_model, smpl_to_joints
from scripts.B_Canonicalization.transform_solver import _target_scale_bone_lengths


class CanonicalizationSmplForwardTests(unittest.TestCase):
    def test_b_smpl_model_loads_through_local_a_backend(self) -> None:
        model = load_smpl_model(SMPL_MODEL_PATH)

        self.assertEqual(tuple(model.J_regressor.shape), (24, 6890))

    def test_b_smpl_to_joints_exposes_backend_used(self) -> None:
        pose = np.zeros((2, 72), dtype=np.float32)
        trans = np.zeros((2, 3), dtype=np.float32)
        beta = np.zeros((1, 10), dtype=np.float32)

        result = smpl_to_joints(pose, trans, beta, smpl_model_path=SMPL_MODEL_PATH, batch_size=1)

        self.assertEqual(result.joints.shape, (2, 24, 3))
        self.assertTrue(np.all(np.isfinite(result.joints)))
        self.assertEqual(result.backend_used, "smplx")
        self.assertEqual(result.backend_notes, ())

    def test_target_scale_bone_lengths_uses_local_smpl_backend(self) -> None:
        config = CanonicalizationConfig()

        target_lengths, backend_used, backend_notes = _target_scale_bone_lengths(
            str(config.model_root),
            str(config.smpl_model_path),
            ROBUST_SCALE_BONE_PAIRS,
            int(config.smpl_batch_size),
        )

        self.assertEqual(backend_used, "smplx")
        self.assertEqual(backend_notes, ())
        self.assertEqual(set(target_lengths), {"left_hip-left_knee", "left_knee-left_ankle", "right_hip-right_knee", "right_knee-right_ankle"})
        self.assertTrue(all(np.isfinite(value) and value > 0.0 for value in target_lengths.values()))


if __name__ == "__main__":
    unittest.main()
