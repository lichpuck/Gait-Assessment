"""Local SMPL forward wrapper shared by audition and canonicalization."""

from __future__ import annotations

from care_pd_pipeline.C_Sequence_Animation.smpl_forward import (
    SmplForwardResult,
    load_smpl_joint_template,
    load_smplx_model,
    smpl_to_joints,
)


def load_smpl_model(path: str):
    return load_smpl_joint_template(path)


__all__ = [
    "SmplForwardResult",
    "load_smpl_joint_template",
    "load_smpl_model",
    "load_smplx_model",
    "smpl_to_joints",
]