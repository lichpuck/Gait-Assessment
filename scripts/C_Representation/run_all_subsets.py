#!/usr/bin/env python3
"""Build C_Representation for all subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.C_Representation.config import INPUT_DIR, OUTPUT_DIR, RepresentationConfig  # noqa: E402
from scripts.C_Representation.pipeline import process_all_subsets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CARE-PD C_Representation for all subsets.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Input root for B_Canonicalization outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for C_Representation outputs.")
    parser.add_argument("--max-trials-per-subset", type=int, default=None, help="Optional cap per subset.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences with existing NPZ/JSON outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RepresentationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    summaries = process_all_subsets(
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_trials_per_subset=args.max_trials_per_subset,
        skip_existing=args.skip_existing,
    )
    processed = sum(int(item["processed_trials"]) for item in summaries)
    failed = sum(int(item["failed_trials"]) for item in summaries)
    skipped = sum(int(item["skipped_existing"]) for item in summaries)
    print(f"Processed subsets: {len(summaries)}")
    print(f"- processed trials: {processed}")
    print(f"- failed trials: {failed}")
    print(f"- skipped existing: {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
