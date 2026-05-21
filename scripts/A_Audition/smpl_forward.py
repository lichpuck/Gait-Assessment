"""Local SMPL forward helpers for A_Audition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SmplForwardResult:
    joints: np.ndarray
    backend_notes: tuple[str, ...] = ()


def _patch_numpy_chumpy_aliases() -> None:
    """Patch NumPy 2.x aliases needed by legacy chumpy objects in SMPL pkl files."""

    alias_values = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": np.object_,
        "unicode": np.str_,
        "str": np.str_,
    }
    for name, value in alias_values.items():
        setattr(np, name, value)


def _import_torch_and_smpl() -> tuple[Any, Any]:
    _patch_numpy_chumpy_aliases()
    try:
        import torch
    except ImportError as exc:
        raise ImportError("A_Audition SMPL forward requires torch in the active environment") from exc
    try:
        from smplx.body_models import SMPL
    except ImportError as exc:
        raise ImportError("A_Audition SMPL forward requires smplx in the active environment") from exc
    return torch, SMPL


def _default_model_path(model_root: str | Path | None, smpl_model_path: str | Path | None) -> Path:
    if smpl_model_path is not None:
        return Path(smpl_model_path).expanduser().resolve()
    if model_root is None:
        model_root = Path(__file__).resolve().parents[2] / "body_models"
    return (Path(model_root).expanduser() / "smpl" / "SMPL_NEUTRAL.pkl").resolve()


@lru_cache(maxsize=4)
def _load_smpl_model_cached(model_path_str: str):
    torch, SMPL = _import_torch_and_smpl()
    model = SMPL(model_path_str, batch_size=1, gender="neutral")
    model.eval()
    model.to(torch.device("cpu"))
    return model


def load_smpl_model(path: str | Path):
    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return _load_smpl_model_cached(str(model_path))


def load_smplx_model(path: str | Path):
    return load_smpl_model(path)


def _normalize_pose_trans_beta(
    pose: np.ndarray,
    trans: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_array = np.asarray(pose, dtype=np.float32)
    trans_array = np.asarray(trans, dtype=np.float32)
    beta_array = np.asarray(beta, dtype=np.float32)

    if pose_array.ndim != 2 or pose_array.shape[1] != 72:
        raise ValueError(f"pose must have shape (T, 72), got {pose_array.shape}")
    if trans_array.shape != (pose_array.shape[0], 3):
        raise ValueError(f"trans must have shape ({pose_array.shape[0]}, 3), got {trans_array.shape}")
    if beta_array.shape == (10,):
        beta_array = beta_array.reshape(1, 10)
    if beta_array.shape == (1, 10):
        beta_array = np.repeat(beta_array, pose_array.shape[0], axis=0)
    elif beta_array.shape != (pose_array.shape[0], 10):
        raise ValueError(f"beta must have shape (1, 10), (10,), or ({pose_array.shape[0]}, 10), got {beta_array.shape}")

    if not np.all(np.isfinite(pose_array)):
        raise ValueError("pose contains non-finite values")
    if not np.all(np.isfinite(trans_array)):
        raise ValueError("trans contains non-finite values")
    if not np.all(np.isfinite(beta_array)):
        raise ValueError("beta contains non-finite values")
    return pose_array, trans_array, beta_array


def smpl_to_joints(
    pose: np.ndarray,
    trans: np.ndarray,
    beta: np.ndarray,
    *,
    model_root: str | Path | None = None,
    smpl_model_path: str | Path | None = None,
    batch_size: int = 256,
) -> SmplForwardResult:
    pose_array, trans_array, beta_array = _normalize_pose_trans_beta(pose, trans, beta)
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model_path = _default_model_path(model_root, smpl_model_path)
    model = load_smpl_model(model_path)
    torch, _ = _import_torch_and_smpl()

    joints_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, pose_array.shape[0], int(batch_size)):
            end = min(start + int(batch_size), pose_array.shape[0])
            pose_batch = torch.as_tensor(pose_array[start:end], dtype=torch.float32)
            trans_batch = torch.as_tensor(trans_array[start:end], dtype=torch.float32)
            beta_batch = torch.as_tensor(beta_array[start:end], dtype=torch.float32)
            output = model(
                global_orient=pose_batch[:, :3],
                body_pose=pose_batch[:, 3:],
                betas=beta_batch,
                transl=trans_batch,
                return_verts=False,
            )
            joints = output.joints[:, :24, :].detach().cpu().numpy()
            joints_batches.append(joints.astype(np.float32, copy=False))

    if not joints_batches:
        raise ValueError("SMPL forward received zero frames")
    joints_3d = np.concatenate(joints_batches, axis=0)
    return SmplForwardResult(joints=joints_3d.astype(np.float32, copy=False), backend_notes=())


__all__ = [
    "SmplForwardResult",
    "load_smpl_model",
    "load_smplx_model",
    "smpl_to_joints",
]
