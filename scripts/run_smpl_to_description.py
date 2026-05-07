#!/usr/bin/env python3
"""Run one external SMPL pickle through A->F and emit one final JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.A_Audition.pipeline import (  # noqa: E402
    AuditionSequenceResult,
    FailedSequenceResult,
    SkippedSequenceResult,
    canonicalize_record,
)
from scripts.B_Canonicalization.io_utils import load_audition_sequence  # noqa: E402
from scripts.B_Canonicalization.pipeline import run_canonicalization  # noqa: E402
from scripts.C_Representation.io_utils import load_canonicalized_sequence  # noqa: E402
from scripts.C_Representation.pipeline import represent_sequence  # noqa: E402
from scripts.D_Segmentation.io_utils import load_representation_sequence  # noqa: E402
from scripts.D_Segmentation.pipeline import segment_sequence  # noqa: E402
from scripts.E_Extraction.export_utils import write_sequence_json  # noqa: E402
from scripts.E_Extraction.io_utils import load_extraction_sequence  # noqa: E402
from scripts.E_Extraction.pipeline import extract_sequence  # noqa: E402
from scripts.F_Description.descriptor_rules import build_reference  # noqa: E402
from scripts.F_Description.io_utils import load_sequence_json, load_subset_reference_tables  # noqa: E402
from scripts.F_Description.pipeline import describe_one_sequence  # noqa: E402
from scripts.single_sequence_input import NormalizedInput, normalize_input_pkl  # noqa: E402


DEFAULT_REFERENCE_SUBSET = "BMCLab"
FINAL_OUTPUT_ROOT = ROOT / "outputs" / "G_Integrated"
HIDDEN_WORK_ROOT = ROOT / "outputs" / ".smpl_pipeline_tmp"


def _safe_name(text: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(text)).strip("_") or "input"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _stage_status(a_result, b_result, c_result, d_result, e_result, f_result) -> dict[str, object]:
    return {
        "A_Audition": {
            "status": "success",
            "warnings": list(a_result.warnings),
        },
        "B_Canonicalization": {
            "status": "success",
            "warnings": list(b_result.warnings),
            "checks": b_result.checks,
        },
        "C_Representation": {
            "status": "success",
            "warnings": list(c_result.warnings),
            "quality_flags": c_result.quality_flags,
            "quality_metrics": c_result.quality_metrics,
        },
        "D_Segmentation": {
            "status": "success",
            "warnings": list(d_result.warnings),
            "quality_flags": d_result.quality_flags,
            "quality_metrics": d_result.quality_metrics,
        },
        "E_Extraction": {
            "status": "success",
            "warnings": list(e_result.warnings),
            "walk_segment_count": len(e_result.walk_rows),
            "turn_segment_count": len(e_result.turn_rows),
        },
        "F_Description": {
            "status": "success",
            "warnings": list(f_result.warnings),
        },
    }


def _final_payload(
    normalized: NormalizedInput,
    final_output_path: Path,
    reference_subset: str,
    a_result: AuditionSequenceResult,
    b_result,
    c_result,
    d_result,
    e_result,
    f_result,
) -> dict[str, object]:
    sequence = normalized.sequence
    return {
        "module": "G_Integrated",
        "version": "1.0.0",
        "input": {
            "input_pkl": str(normalized.input_path),
            "input_kind": normalized.input_kind,
            "schema_version": normalized.schema_version,
        },
        "sequence": {
            "subset": sequence.subset_name,
            "subject_id": sequence.subject_id,
            "trial_id": sequence.trial_id,
            "fps": sequence.fps,
            "num_frames": sequence.num_frames,
            "duration_sec": sequence.duration_sec,
            "source_path": str(sequence.source_path),
        },
        "reference_library": {
            "name": reference_subset,
            "policy": "pluggable_reference_library",
        },
        "quality_summary": {
            "warnings": list(dict.fromkeys([*a_result.warnings, *b_result.warnings, *c_result.warnings, *d_result.warnings, *e_result.warnings, *f_result.warnings])),
            "stage_status": _stage_status(a_result, b_result, c_result, d_result, e_result, f_result),
        },
        "physical_features": {
            "walk_segments": e_result.walk_rows,
            "turn_segments": e_result.turn_rows,
            "segment_payloads": e_result.segment_payloads,
        },
        "semi_structured_description": {
            "language": f_result.config.language,
            "profile": f_result.profile,
        },
        "outputs": {
            "json": str(final_output_path),
        },
    }


def _stage_error(stage_name: str, detail: str) -> RuntimeError:
    return RuntimeError(f"[{stage_name}] {detail}")


def run_one_sequence(input_pkl: str | Path) -> Path:
    normalized = normalize_input_pkl(input_pkl)
    sequence = normalized.sequence

    FINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    HIDDEN_WORK_ROOT.mkdir(parents=True, exist_ok=True)

    run_name = _safe_name(f"{sequence.subject_id}__{sequence.trial_id}")
    final_output_path = FINAL_OUTPUT_ROOT / f"{run_name}.json"

    with tempfile.TemporaryDirectory(prefix="run_", dir=HIDDEN_WORK_ROOT) as temp_root_str:
        temp_root = Path(temp_root_str)
        stage_a_dir = temp_root / "A_Audition"
        stage_b_dir = temp_root / "B_Canonicalization"
        stage_c_dir = temp_root / "C_Representation"
        stage_d_dir = temp_root / "D_Segmentation"
        stage_e_dir = temp_root / "E_Extraction"
        stage_f_dir = temp_root / "F_Description"

        a_result = canonicalize_record(
            sequence,
            write_outputs=True,
            generate_diagnostics=False,
            output_dir=stage_a_dir,
            update_summary=False,
        )
        if isinstance(a_result, SkippedSequenceResult):
            raise _stage_error("A_Audition", f"sequence skipped: {a_result.skip_reason}")
        if isinstance(a_result, FailedSequenceResult):
            raise _stage_error("A_Audition", a_result.message)

        audition_sequence = load_audition_sequence(a_result.output_paths["npz"], a_result.output_paths["json"])
        b_result = run_canonicalization(
            audition_sequence,
            write_outputs=True,
            generate_diagnostics=False,
            output_dir=stage_b_dir,
        )

        canonicalized_sequence = load_canonicalized_sequence(b_result.output_paths["npz"], b_result.output_paths["json"])
        c_result = represent_sequence(canonicalized_sequence, write_outputs=True, output_dir=stage_c_dir)

        representation_sequence = load_representation_sequence(c_result.output_paths["npz"], c_result.output_paths["json"])
        d_result = segment_sequence(
            representation_sequence,
            output_dir=stage_d_dir,
            write_outputs=True,
            generate_diagnostics=False,
        )

        extraction_sequence = load_extraction_sequence(
            c_result.output_paths["npz"],
            c_result.output_paths["json"],
            d_result.output_paths["json"],
        )
        e_result = extract_sequence(extraction_sequence)
        e_json_path = write_sequence_json(e_result, stage_e_dir)
        e_result.output_paths["json"] = e_json_path

        from scripts.F_Description.config import DescriptionConfig, INPUT_E_DIR  # noqa: E402

        description_config = DescriptionConfig(input_e_dir=INPUT_E_DIR, output_dir=stage_f_dir)
        walk_df, turn_df = load_subset_reference_tables(DEFAULT_REFERENCE_SUBSET, description_config)
        reference = build_reference(walk_df, turn_df)
        description_sequence = load_sequence_json(e_json_path)
        f_result = describe_one_sequence(
            description_sequence,
            reference=reference,
            config=description_config,
            output_dir=stage_f_dir,
            reference_subset_name=DEFAULT_REFERENCE_SUBSET,
        )

        payload = _final_payload(
            normalized,
            final_output_path,
            DEFAULT_REFERENCE_SUBSET,
            a_result,
            b_result,
            c_result,
            d_result,
            e_result,
            f_result,
        )
        with final_output_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)

    return final_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SMPL pickle through A->F and emit one final JSON.")
    parser.add_argument("--input-pkl", required=True, help="Path to a single-sequence or single-record dataset pickle.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_one_sequence(args.input_pkl)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
