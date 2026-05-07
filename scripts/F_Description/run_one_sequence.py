#!/usr/bin/env python3
"""Build one F_Description sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.F_Description.config import DescriptionConfig, INPUT_E_DIR, OUTPUT_DIR  # noqa: E402
from scripts.F_Description.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD F_Description sequence.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--input-e-dir", default=str(INPUT_E_DIR), help="Input root for E_Extraction outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for F_Description outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DescriptionConfig(
        input_e_dir=Path(args.input_e_dir),
        output_dir=Path(args.output_dir),
    )
    result = process_one_sequence(
        args.subset,
        args.subject_id,
        args.trial_id,
        config=config,
        input_e_dir=args.input_e_dir,
        output_dir=args.output_dir,
    )
    print(f"Described {result.sequence.subset_name}/{result.sequence.subject_id}/{result.sequence.trial_id}")
    print(f"- walk segments: {len(result.sequence.walk_segments)}")
    print(f"- turn segments: {len(result.sequence.turn_segments)}")
    print(f"- json: {result.output_paths['json']}")
    print(f"- summary: {result.profile.get('summary_zh', '')}")
    if result.warnings:
        print("- warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())