from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.single_sequence_input import normalize_input_pkl


def _sample_sequence_payload(frame_count: int = 8) -> dict[str, object]:
    return {
        "pose": np.zeros((frame_count, 72), dtype=np.float32),
        "trans": np.zeros((frame_count, 3), dtype=np.float32),
        "beta": np.zeros((10,), dtype=np.float32),
        "fps": 30.0,
    }


class NormalizeInputPklTests(unittest.TestCase):
    def _write_pickle(self, root: Path, name: str, payload: object) -> Path:
        path = root / name
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    def test_flat_single_sequence_payload_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            input_path = self._write_pickle(temp_root, "flat.pkl", _sample_sequence_payload())

            normalized = normalize_input_pkl(input_path)

            self.assertEqual(normalized.input_kind, "flat_single_sequence")
            self.assertEqual(normalized.sequence.subject_id, "flat")
            self.assertEqual(normalized.sequence.trial_id, "trial_0001")
            self.assertEqual(normalized.sequence.pose.shape, (8, 72))

    def test_nested_single_sequence_payload_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            payload = {
                "SUB01": {
                    "TRIAL01": _sample_sequence_payload(),
                }
            }
            input_path = self._write_pickle(temp_root, "nested.pkl", payload)

            normalized = normalize_input_pkl(input_path)

            self.assertEqual(normalized.input_kind, "nested_dataset_sequence")
            self.assertEqual(normalized.sequence.subset_name, "nested")
            self.assertEqual(normalized.sequence.subject_id, "SUB01")
            self.assertEqual(normalized.sequence.trial_id, "TRIAL01")

    def test_nested_multi_sequence_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            payload = {
                "SUB01": {
                    "TRIAL01": _sample_sequence_payload(),
                    "TRIAL02": _sample_sequence_payload(),
                }
            }
            input_path = self._write_pickle(temp_root, "multi.pkl", payload)

            with self.assertRaisesRegex(ValueError, "requires exactly 1"):
                normalize_input_pkl(input_path)