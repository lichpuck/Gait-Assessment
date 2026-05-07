#!/usr/bin/env python3
"""Build one D_Segmentation sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.D_Segmentation.config import INPUT_DIR, OUTPUT_DIR, SegmentationConfig  # noqa: E402
from scripts.D_Segmentation.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD D_Segmentation sequence.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Input root for C_Representation outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for D_Segmentation outputs.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostic PNG generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SegmentationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    result = process_one_sequence(
        args.subset,
        args.subject_id,
        args.trial_id,
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        generate_diagnostics=not args.no_diagnostics,
    )
    print(f"Segmented {result.sequence.subset_name}/{result.sequence.subject_id}/{result.sequence.trial_id}")
    print(f"- frames: {result.sequence.num_frames}")
    print(f"- segmentation success: {result.segmentation_success}")
    print(f"- segments: {len(result.segments)}")
    print(f"- hesitation frames: {int(result.quality_metrics['hesitation_frame_count'])}")
    for name, path in sorted(result.output_paths.items()):
        print(f"- {name}: {path}")
    if result.warnings:
        print("- warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
