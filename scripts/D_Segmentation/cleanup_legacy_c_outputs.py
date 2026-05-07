#!/usr/bin/env python3
"""List or delete legacy C_Representation sequence artifacts.

The script is intentionally conservative: it treats an NPZ as strict C v2 only
when all required D input keys are present. For legacy NPZ files, it targets the
NPZ plus same-stem sidecars under the same subset directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.D_Segmentation.config import INPUT_DIR  # noqa: E402
from scripts.D_Segmentation.io_utils import REQUIRED_NPZ_KEYS  # noqa: E402


SIDECAR_SUFFIXES = (".npz", ".json", ".png", ".csv")
STALE_C_V2_PATTERNS = ("*.png", "*__segments.csv", "representation_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or delete legacy C_Representation sequence files.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="C_Representation root.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Default is dry-run list only.")
    return parser.parse_args()


def is_strict_v2_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return all(key in payload.files for key in REQUIRED_NPZ_KEYS)
    except Exception:
        return False


def legacy_sidecars(npz_path: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in SIDECAR_SUFFIXES:
        candidate = npz_path.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)

    old_segments = npz_path.with_name(f"{npz_path.stem}__segments.csv")
    if old_segments.exists() and old_segments.is_file():
        paths.append(old_segments)
    return sorted(set(paths))


def stale_non_contract_files(input_root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in STALE_C_V2_PATTERNS:
        paths.extend(path for path in input_root.glob(f"*/{pattern}") if path.is_file())
    return sorted(set(paths))


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    legacy_npzs = [path for path in sorted(input_root.glob("*/*.npz")) if not is_strict_v2_npz(path)]
    legacy_targets = [sidecar for npz_path in legacy_npzs for sidecar in legacy_sidecars(npz_path)]
    stale_targets = stale_non_contract_files(input_root)
    targets = sorted(set(legacy_targets + stale_targets))

    mode = "delete" if args.apply else "dry-run"
    print(f"[{mode}] legacy C sequence NPZs: {len(legacy_npzs)}")
    print(f"[{mode}] legacy sequence sidecars: {len(legacy_targets)}")
    print(f"[{mode}] stale non-contract C v2 sidecars: {len(stale_targets)}")
    print(f"[{mode}] targeted files: {len(targets)}")
    for path in targets:
        print(path)

    if args.apply:
        for path in targets:
            path.unlink()
        print(f"[delete] removed {len(targets)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
