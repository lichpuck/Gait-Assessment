"""CARE-PD C_Representation package."""

from .config import RepresentationConfig
from .pipeline import (
    SequenceRepresentationResult,
    process_all_subsets,
    process_one_sequence,
    process_subset,
    represent_sequence,
)

__all__ = [
    "RepresentationConfig",
    "SequenceRepresentationResult",
    "represent_sequence",
    "process_one_sequence",
    "process_subset",
    "process_all_subsets",
]
