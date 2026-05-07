#!/usr/bin/env python3
"""Batch canonicalize one subset of A_Audition outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.B_Canonicalization.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, CanonicalizationConfig  # noqa: E402
from scripts.B_Canonicalization.pipeline import process_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize one A_Audition subset.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing A_Audition outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for B_Canonicalization outputs.")
    parser.add_argument("--subset", required=True, help="Subset name.")
    parser.add_argument("--subject-id", help="Optional subject filter.")
    parser.add_argument("--trial-id", action="append", dest="trial_ids", help="Optional repeated trial IDs to include.")
    parser.add_argument("--trial-contains", help="Optional substring filter on trial_id.")
    parser.add_argument("--max-trials", type=int, help="Optional limit after filtering.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences whose NPZ, JSON, and PNG already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CanonicalizationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    summary = process_subset(
        args.subset,
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        subject_filter=args.subject_id,
        trial_ids=set(args.trial_ids) if args.trial_ids else None,
        trial_contains=args.trial_contains,
        max_trials=args.max_trials,
        skip_existing=args.skip_existing,
    )
    print(
        f"{summary['subset_name']}: processed={summary['processed_trials']} "
        f"failed={summary['failed_trials']} skipped_existing={summary['skipped_existing_trials']} "
        f"inspected={summary['inspected_trials']}"
    )
    return 1 if summary["failed_trials"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

