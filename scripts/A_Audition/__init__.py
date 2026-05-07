"""A_Audition: convert raw CARE-PD SMPL sequences into Joint3D outputs."""

from .config import AuditionConfig
from .pipeline import (
    AuditionFailure,
    AuditionResult,
    AuditionSequenceResult,
    SkippedSequenceResult,
    process_all_subsets,
    process_one_sequence,
    process_subset,
)

__all__ = [
    "AuditionConfig",
    "AuditionFailure",
    "AuditionResult",
    "AuditionSequenceResult",
    "SkippedSequenceResult",
    "process_all_subsets",
    "process_one_sequence",
    "process_subset",
]
