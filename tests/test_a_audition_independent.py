from __future__ import annotations

from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from scripts.A_Audition.config import RAW_DATA_DIR, SMPL_MODEL_PATH
from scripts.A_Audition.io_utils import (
    SequenceRecord,
    load_pkl_dataset,
    load_one_sequence,
    normalize_beta,
    record_from_payload,
    resolve_subset_path,
    validate_sequence,
)
from scripts.A_Audition.smpl_forward import load_smpl_model, smpl_to_joints


def _payload(frame_count: int = 8, beta: np.ndarray | None = None) -> dict[str, object]:
    return {
        "pose": np.zeros((frame_count, 72), dtype=np.float32),
        "trans": np.zeros((frame_count, 3), dtype=np.float32),
        "beta": np.zeros((1, 10), dtype=np.float32) if beta is None else beta,
        "fps": 30.0,
        "medication": "off",
    }


class AuditionIoUtilsTests(unittest.TestCase):
    def test_nested_pickle_dataset_loads_and_builds_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            subset_path = temp_root / "sample.pkl"
            with subset_path.open("wb") as handle:
                pickle.dump({"SUB01": {"TRIAL01": _payload()}}, handle)

            dataset = load_pkl_dataset(subset_path)
            record = record_from_payload("sample", "SUB01", "TRIAL01", dataset["SUB01"]["TRIAL01"], subset_path)

            self.assertIsInstance(record, SequenceRecord)
            self.assertEqual(record.pose.shape, (8, 72))
            self.assertEqual(record.trans.shape, (8, 3))
            self.assertEqual(record.beta.shape, (1, 10))
            self.assertEqual(record.metadata["medication"], "off")

    def test_beta_shapes_normalize_to_single_row(self) -> None:
        self.assertEqual(normalize_beta(np.zeros(10, dtype=np.float32)).shape, (1, 10))
        self.assertEqual(normalize_beta(np.zeros((1, 10), dtype=np.float32)).shape, (1, 10))
        self.assertEqual(normalize_beta(np.zeros((5, 10), dtype=np.float32), num_frames=5).shape, (1, 10))

    def test_nonconstant_per_frame_beta_is_rejected(self) -> None:
        beta = np.zeros((5, 10), dtype=np.float32)
        beta[-1, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "per-frame beta must be constant"):
            normalize_beta(beta, num_frames=5)

    def test_invalid_pose_shape_reports_reason(self) -> None:
        payload = _payload()
        payload["pose"] = np.zeros((8, 71), dtype=np.float32)

        validated = validate_sequence(payload)

        self.assertFalse(validated["valid"])
        self.assertIn("pose must have shape", str(validated["reason"]))

    def test_resolve_subset_path_finds_raw_test_subset(self) -> None:
        resolved = resolve_subset_path("test", RAW_DATA_DIR)

        self.assertEqual(resolved.name, "test.pkl")
        self.assertTrue(resolved.exists())

    def test_load_one_sequence_accepts_string_for_numeric_subject_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            subset_path = temp_root / "numeric.pkl"
            with subset_path.open("wb") as handle:
                pickle.dump({2: {"TRIAL01": _payload()}}, handle)

            record = load_one_sequence("numeric", "2", "TRIAL01", temp_root)

            self.assertEqual(record.subject_id, "2")
            self.assertEqual(record.trial_id, "TRIAL01")


class AuditionSmplForwardTests(unittest.TestCase):
    def test_smpl_model_loads(self) -> None:
        model = load_smpl_model(SMPL_MODEL_PATH)

        self.assertEqual(tuple(model.J_regressor.shape), (24, 6890))

    def test_smpl_to_joints_returns_24_finite_joints(self) -> None:
        pose = np.zeros((2, 72), dtype=np.float32)
        trans = np.zeros((2, 3), dtype=np.float32)
        beta = np.zeros((1, 10), dtype=np.float32)

        result = smpl_to_joints(pose, trans, beta, smpl_model_path=SMPL_MODEL_PATH, batch_size=1)

        self.assertEqual(result.joints.shape, (2, 24, 3))
        self.assertTrue(np.all(np.isfinite(result.joints)))


if __name__ == "__main__":
    unittest.main()
