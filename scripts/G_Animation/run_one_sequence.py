#!/usr/bin/env python3
"""Render one CARE-PD animation sequence from B/C/D/F outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.G_Animation.config import AnimationConfig  # noqa: E402
from scripts.G_Animation.pipeline import process_one_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one G_Animation composite MP4 from existing B/C/D/F outputs.")
    parser.add_argument("--subset", required=True, help="Subset name.")
    parser.add_argument("--subject-id", required=True, help="Subject identifier.")
    parser.add_argument("--trial-id", required=True, help="Trial identifier.")
    parser.add_argument("--input-b-dir", default=None, help="Optional B_Canonicalization input root override.")
    parser.add_argument("--input-c-dir", default=None, help="Optional C_Representation input root override.")
    parser.add_argument("--input-d-dir", default=None, help="Optional D_Segmentation input root override.")
    parser.add_argument("--input-f-dir", default=None, help="Optional F_Description input root override.")
    parser.add_argument("--output-dir", default=None, help="Optional G_Animation output root override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnimationConfig(
        input_b_dir=Path(args.input_b_dir) if args.input_b_dir else AnimationConfig().input_b_dir,
        input_c_dir=Path(args.input_c_dir) if args.input_c_dir else AnimationConfig().input_c_dir,
        input_d_dir=Path(args.input_d_dir) if args.input_d_dir else AnimationConfig().input_d_dir,
        input_f_dir=Path(args.input_f_dir) if args.input_f_dir else AnimationConfig().input_f_dir,
        output_dir=Path(args.output_dir) if args.output_dir else AnimationConfig().output_dir,
    )
    result = process_one_sequence(
        args.subset,
        args.subject_id,
        args.trial_id,
        config=config,
        output_dir=args.output_dir,
    )
    print(result.output_paths["mp4"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())