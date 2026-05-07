#!/usr/bin/env python3
"""Canonicalize all A_Audition subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.B_Canonicalization.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, CanonicalizationConfig  # noqa: E402
from scripts.B_Canonicalization.pipeline import process_all_subsets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize all A_Audition subsets.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing A_Audition outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for B_Canonicalization outputs.")
    parser.add_argument("--max-trials-per-subset", type=int, help="Optional smoke-test limit per subset.")
    parser.add_argument("--subset", action="append", dest="subsets", help="Optional subset name to include; repeat as needed.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences whose NPZ, JSON, and PNG already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CanonicalizationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    summaries = process_all_subsets(
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_trials_per_subset=args.max_trials_per_subset,
        subset_names=set(args.subsets) if args.subsets else None,
        skip_existing=args.skip_existing,
    )
    total_failed = 0
    for summary in summaries:
        total_failed += int(summary["failed_trials"])
        print(
            f"{summary['subset_name']}: processed={summary['processed_trials']} "
            f"failed={summary['failed_trials']} skipped_existing={summary['skipped_existing_trials']} "
            f"inspected={summary['inspected_trials']}"
        )
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

