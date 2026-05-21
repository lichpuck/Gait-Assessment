"""SMPL forward compatibility layer for B_Canonicalization.

B uses SMPL only to compute neutral beta=0 target bone lengths for scale
alignment.  The actual SMPL forward implementation lives in A_Audition so the
workflow has one local, tested SMPLX loader instead of depending on the removed
``care_pd_pipeline`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.A_Audition.smpl_forward import (
    load_smpl_model as _load_smpl_model,
    load_smplx_model as _load_smplx_model,
    smpl_to_joints as _smpl_to_joints,
)


@dataclass(frozen=True)
class SmplForwardResult:
    joints: np.ndarray
    backend_used: str = "smplx"
    backend_notes: tuple[str, ...] = ()


def load_smpl_model(path: str | Path):
    return _load_smpl_model(path)


def load_smpl_joint_template(path: str | Path):
    return _load_smpl_model(path)


def load_smplx_model(path: str | Path):
    return _load_smplx_model(path)


def smpl_to_joints(
    pose: np.ndarray,
    trans: np.ndarray,
    beta: np.ndarray,
    *,
    model_root: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    batch_size: int = 256,
) -> SmplForwardResult:
    result = _smpl_to_joints(
        pose,
        trans,
        beta,
        model_root=model_root,
        smpl_model_path=smpl_model_path,
        batch_size=batch_size,
    )
    return SmplForwardResult(
        joints=np.asarray(result.joints, dtype=np.float32),
        backend_used="smplx",
        backend_notes=tuple(result.backend_notes),
    )


__all__ = [
    "SmplForwardResult",
    "load_smpl_joint_template",
    "load_smpl_model",
    "load_smplx_model",
    "smpl_to_joints",
]
