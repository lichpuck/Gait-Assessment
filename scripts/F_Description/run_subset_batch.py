#!/usr/bin/env python3
"""Build one subset of F_Description outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.F_Description.config import DescriptionConfig, INPUT_E_DIR, OUTPUT_DIR  # noqa: E402
from scripts.F_Description.pipeline import process_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD F_Description subset.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab.")
    parser.add_argument("--input-e-dir", default=str(INPUT_E_DIR), help="Input root for E_Extraction outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for F_Description outputs.")
    parser.add_argument("--trial-contains", default=None, help="Optional case-insensitive filter for trial id.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional max number of trials to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DescriptionConfig(
        input_e_dir=Path(args.input_e_dir),
        output_dir=Path(args.output_dir),
    )
    summary = process_subset(
        args.subset,
        config=config,
        input_e_dir=args.input_e_dir,
        output_dir=args.output_dir,
        trial_contains=args.trial_contains,
        max_trials=args.max_trials,
    )
    print(f"Processed {summary['processed_trials']} trial(s) from {summary['subset_name']}")
    print(f"- output_dir: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
