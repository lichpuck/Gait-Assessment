#!/usr/bin/env python3
"""Build all E_Extraction subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.E_Extraction.config import ExtractionConfig, INPUT_C_DIR, INPUT_D_DIR, OUTPUT_DIR  # noqa: E402
from scripts.E_Extraction.pipeline import process_all_subsets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all CARE-PD E_Extraction subsets.")
    parser.add_argument("--input-c-dir", default=str(INPUT_C_DIR), help="Input root for C_Representation outputs.")
    parser.add_argument("--input-d-dir", default=str(INPUT_D_DIR), help="Input root for D_Segmentation outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for E_Extraction outputs.")
    parser.add_argument("--max-trials-per-subset", type=int, default=None, help="Optional max trials to process per subset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExtractionConfig(
        input_c_dir=Path(args.input_c_dir),
        input_d_dir=Path(args.input_d_dir),
        output_dir=Path(args.output_dir),
    )
    summaries = process_all_subsets(
        config=config,
        input_c_dir=args.input_c_dir,
        input_d_dir=args.input_d_dir,
        output_dir=args.output_dir,
        max_trials_per_subset=args.max_trials_per_subset,
    )
    for summary in summaries:
        print(
            f"{summary['subset_name']}: processed {summary['processed_trials']} trial(s), "
            f"failed {summary['failed_trials']}"
        )
        print(f"- walk_features_csv: {summary['walk_features_csv']}")
        print(f"- turn_features_csv: {summary['turn_features_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())