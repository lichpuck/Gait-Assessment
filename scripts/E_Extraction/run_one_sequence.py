#!/usr/bin/env python3
"""Build one E_Extraction sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.E_Extraction.config import ExtractionConfig, INPUT_C_DIR, INPUT_D_DIR, OUTPUT_DIR  # noqa: E402
from scripts.E_Extraction.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD E_Extraction sequence.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--input-c-dir", default=str(INPUT_C_DIR), help="Input root for C_Representation outputs.")
    parser.add_argument("--input-d-dir", default=str(INPUT_D_DIR), help="Input root for D_Segmentation outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for E_Extraction outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExtractionConfig(
        input_c_dir=Path(args.input_c_dir),
        input_d_dir=Path(args.input_d_dir),
        output_dir=Path(args.output_dir),
    )
    result = process_one_sequence(
        args.subset,
        args.subject_id,
        args.trial_id,
        config=config,
        input_c_dir=args.input_c_dir,
        input_d_dir=args.input_d_dir,
        output_dir=args.output_dir,
    )
    print(f"Extracted {result.sequence.subset_name}/{result.sequence.subject_id}/{result.sequence.trial_id}")
    print(f"- walk segments: {len(result.walk_rows)}")
    print(f"- turn segments: {len(result.turn_rows)}")
    print(f"- extraction success: {result.extraction_success}")
    for name, path in sorted(result.output_paths.items()):
        print(f"- {name}: {path}")
    if result.warnings:
        print("- warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())