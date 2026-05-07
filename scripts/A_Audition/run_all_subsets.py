#!/usr/bin/env python3
"""Convert all CARE-PD subsets into Joint3D outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.A_Audition.config import DEFAULT_OUTPUT_DIR, RAW_DATA_DIR, AuditionConfig  # noqa: E402
from scripts.A_Audition.pipeline import process_all_subsets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert all CARE-PD subsets into Joint3D outputs.")
    parser.add_argument("--raw-data-dir", default=str(RAW_DATA_DIR), help="Directory containing raw subset .pkl files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--max-trials-per-subset", type=int, help="Optional smoke-test limit per subset.")
    parser.add_argument("--subset", action="append", dest="subsets", help="Optional subset name to include; repeat as needed.")
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
    summaries = process_all_subsets(
        config=config,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        max_trials_per_subset=args.max_trials_per_subset,
        subset_names=set(args.subsets) if args.subsets else None,
        generate_diagnostics=not args.no_diagnostics,
        skip_existing=args.skip_existing,
    )
    for summary in summaries:
        print(
            f"{summary['subset_name']}: processed={summary['processed_trials']} "
            f"skipped={summary['skipped_trials']} failed={summary['failed_trials']} "
            f"inspected={summary['inspected_trials']} summary={summary['summary_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
