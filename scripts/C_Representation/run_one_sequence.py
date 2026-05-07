#!/usr/bin/env python3
"""Build one C_Representation sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.C_Representation.config import INPUT_DIR, OUTPUT_DIR, RepresentationConfig  # noqa: E402
from scripts.C_Representation.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one CARE-PD C_Representation sequence.")
    parser.add_argument("--subset", required=True, help="Subset name like 3DGait.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Input root for B_Canonicalization outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output root for C_Representation outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RepresentationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    try:
        result = process_one_sequence(
            args.subset,
            args.subject_id,
            args.trial_id,
            config=config,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    except Exception as error:
        print(f"Representation failed: {error}", file=sys.stderr)
        return 1

    print(f"Represented {result.sequence.subset_name}/{result.sequence.subject_id}/{result.sequence.trial_id}")
    print(f"- frames: {result.sequence.num_frames}")
    print(f"- representation success: {result.representation_success}")
    print(f"- valid frame ratio: {float(result.quality_metrics.get('valid_frame_ratio', 0.0)):.3f}")
    print(f"- cadence: {float(result.gait_summary.get('cadence_steps_per_min', 0.0)):.2f} steps/min")
    for name, path in sorted(result.output_paths.items()):
        print(f"- {name}: {path}")
    if result.warnings:
        print("- warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
