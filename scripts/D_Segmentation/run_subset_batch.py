#!/usr/bin/env python3
"""Build D_Segmentation for a subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.D_Segmentation.config import INPUT_DIR, OUTPUT_DIR, SegmentationConfig  # noqa: E402
from scripts.D_Segmentation.pipeline import process_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CARE-PD D_Segmentation for a subset.")
    parser.add_argument("--subset", required=True, help="Subset name under outputs/C_Representation.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Input root for C_Representation outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for D_Segmentation outputs.")
    parser.add_argument("--trial-contains", default=None, help="Only process trials whose stem contains this text.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequences with existing NPZ/JSON/CSV outputs.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostic PNG generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SegmentationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    summary = process_subset(
        args.subset,
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trial_contains=args.trial_contains,
        max_trials=args.max_trials,
        generate_diagnostics=not args.no_diagnostics,
        skip_existing=args.skip_existing,
    )
    print(f"Subset: {summary['subset_name']}")
    print(f"- processed: {summary['processed_trials']}")
    print(f"- failed: {summary['failed_trials']}")
    print(f"- summary_path: {summary['summary_path']}")
    return 0 if int(summary["failed_trials"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
