#!/usr/bin/env python3
"""Build one subset of E_Extraction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.E_Extraction.config import ExtractionConfig, INPUT_C_DIR, INPUT_D_DIR, OUTPUT_DIR  # noqa: E402
from scripts.E_Extraction.pipeline import process_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD E_Extraction subset.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab.")
    parser.add_argument("--input-c-dir", default=str(INPUT_C_DIR), help="Input root for C_Representation outputs.")
    parser.add_argument("--input-d-dir", default=str(INPUT_D_DIR), help="Input root for D_Segmentation outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for E_Extraction outputs.")
    parser.add_argument("--trial-contains", default=None, help="Optional case-insensitive filter for trial id.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional max number of trials to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExtractionConfig(
        input_c_dir=Path(args.input_c_dir),
        input_d_dir=Path(args.input_d_dir),
        output_dir=Path(args.output_dir),
    )
    summary = process_subset(
        args.subset,
        config=config,
        input_c_dir=args.input_c_dir,
        input_d_dir=args.input_d_dir,
        output_dir=args.output_dir,
        trial_contains=args.trial_contains,
        max_trials=args.max_trials,
    )
    print(
        f"Processed {summary['processed_trials']} trial(s) from {summary['subset_name']} "
        f"with {summary['failed_trials']} failure(s)."
    )
    print(f"- walk_features_csv: {summary['walk_features_csv']}")
    print(f"- turn_features_csv: {summary['turn_features_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())