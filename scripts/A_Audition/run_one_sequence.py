#!/usr/bin/env python3
"""Convert one raw CARE-PD SMPL sequence into Joint3D outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.A_Audition.config import DEFAULT_OUTPUT_DIR, RAW_DATA_DIR, AuditionConfig  # noqa: E402
from scripts.A_Audition.pipeline import (  # noqa: E402
    AuditionSequenceResult,
    FailedSequenceResult,
    SkippedSequenceResult,
    process_one_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one raw CARE-PD SMPL sequence into Joint3D outputs.")
    parser.add_argument("--subset", required=True, help="Subset name like BMCLab or a direct .pkl path.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier, for example SUB01.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--raw-data-dir", default=str(RAW_DATA_DIR), help="Directory containing raw subset .pkl files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
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
    result = process_one_sequence(
        args.subset,
        args.subject_id,
        args.trial_id,
        config=config,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        generate_diagnostics=not args.no_diagnostics,
    )
    if isinstance(result, SkippedSequenceResult):
        print(f"Skipped {result.record.subset_name}/{result.record.subject_id}/{result.record.trial_id}")
        print(f"- reason: {result.skip_reason}")
        print(f"- duration_sec: {result.record.duration_sec:.3f}")
        if "summary_csv" in result.output_paths:
            print(f"- summary: {result.output_paths['summary_csv']}")
        return 0
    if isinstance(result, FailedSequenceResult):
        print(f"Failed {result.record.subset_name}/{result.record.subject_id}/{result.record.trial_id}")
        print(f"- reason: {result.failure_reason}")
        print(f"- message: {result.message}")
        if "summary_csv" in result.output_paths:
            print(f"- summary: {result.output_paths['summary_csv']}")
        return 0

    assert isinstance(result, AuditionSequenceResult)
    print(f"Converted {result.record.subset_name}/{result.record.subject_id}/{result.record.trial_id}")
    print(f"- joints_3d shape: {result.joints_3d.shape}")
    print(f"- trans_canonical shape: {result.trans_canonical.shape}")
    print(f"- R_total det: {result.axis.determinant:.3f}")
    print(f"- warnings: {', '.join(result.warnings) if result.warnings else 'none'}")
    for name, path in sorted(result.output_paths.items()):
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
