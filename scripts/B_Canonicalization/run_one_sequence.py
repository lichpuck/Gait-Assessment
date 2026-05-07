#!/usr/bin/env python3
"""Canonicalize one A_Audition sequence with one rigid B-stage transform."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.B_Canonicalization.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, CanonicalizationConfig  # noqa: E402
from scripts.B_Canonicalization.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize one A_Audition sequence.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing A_Audition outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for B_Canonicalization outputs.")
    parser.add_argument("--subset", required=True, help="Subset name.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CanonicalizationConfig(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
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
        print(f"Canonicalization failed: {error}", file=sys.stderr)
        return 1

    seq = result.sequence
    print(f"Canonicalized {seq.subset_name}/{seq.subject_id}/{seq.trial_id}")
    print(f"- joints_can shape: {result.joints_can.shape}")
    print(f"- trans_can shape: {result.trans_can.shape}")
    print(f"- det(R_global): {result.checks['det_R_global']:.6f}")
    print(
        f"- scale: enabled={result.transform.scale.enabled} "
        f"s_global={result.transform.s_global:.6f} "
        f"quality={result.transform.scale.quality_flag}"
    )
    print(
        f"- floor leveling: enabled={result.transform.floor.enabled} "
        f"tilt_before={result.transform.floor.tilt_before_deg:.3f} deg "
        f"tilt_after={result.transform.floor.tilt_after_deg:.3f} deg"
    )
    print(f"- first root XY norm: {result.checks['first_root_xy_norm_m']:.6g} m")
    print(
        f"- ground Z P{result.config.ground_percentile:g} after translation: "
        f"{result.checks['ground_z_percentile_after_translation_m']:.6g} m"
    )
    for name, path in sorted(result.output_paths.items()):
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
