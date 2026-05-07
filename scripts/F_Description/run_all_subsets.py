#!/usr/bin/env python3
"""Build all F_Description subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.F_Description.config import DescriptionConfig, INPUT_E_DIR, OUTPUT_DIR  # noqa: E402
from scripts.F_Description.pipeline import process_all_subsets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all CARE-PD F_Description subsets.")
    parser.add_argument("--input-e-dir", default=str(INPUT_E_DIR), help="Input root for E_Extraction outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for F_Description outputs.")
    parser.add_argument("--max-trials-per-subset", type=int, default=None, help="Optional max number of trials per subset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DescriptionConfig(
        input_e_dir=Path(args.input_e_dir),
        output_dir=Path(args.output_dir),
    )
    summaries = process_all_subsets(
        config=config,
        input_e_dir=args.input_e_dir,
        output_dir=args.output_dir,
        max_trials_per_subset=args.max_trials_per_subset,
    )
    print(f"Processed {len(summaries)} subset(s).")
    for summary in summaries:
        print(f"- {summary['subset_name']}: {summary['processed_trials']} trial(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
