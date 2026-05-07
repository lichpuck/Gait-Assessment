#!/usr/bin/env python3
"""Batch-convert one CARE-PD subset into Joint3D outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.A_Audition.config import DEFAULT_OUTPUT_DIR, RAW_DATA_DIR, AuditionConfig  # noqa: E402
from scripts.A_Audition.pipeline import process_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-convert one CARE-PD subset into Joint3D outputs.")
    parser.add_argument("--subset", default="BMCLab", help="Subset name like BMCLab or a direct .pkl path.")
    parser.add_argument("--raw-data-dir", default=str(RAW_DATA_DIR), help="Directory containing raw subset .pkl files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--subject-id", help="Optional subject filter.")
    parser.add_argument("--trial-contains", help="Optional substring filter on trial_id.")
    parser.add_argument("--trial-id", action="append", dest="trial_ids", help="Optional repeated trial IDs to include.")
    parser.add_argument("--max-trials", type=int, help="Optional limit after filtering.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences whose NPZ, JSON, and PNG already exist.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip PNG generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = AuditionConfig(
        raw_data_dir=Path(args.raw_data_dir),
        output_dir=output_dir,
        summary_csv=output_dir / "audition_summary.csv",
    )
    summary = process_subset(
        args.subset,
        config=config,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        subject_filter=args.subject_id,
        trial_ids=set(args.trial_ids) if args.trial_ids else None,
        trial_contains=args.trial_contains,
        max_trials=args.max_trials,
        generate_diagnostics=not args.no_diagnostics,
        skip_existing=args.skip_existing,
    )
    print(
        f"Processed {summary['processed_trials']} trial(s) from {summary['subset_name']} "
        f"(skipped={summary['skipped_trials']}, failed={summary['failed_trials']}, "
        f"inspected={summary['inspected_trials']})."
    )
    print(f"- summary: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
