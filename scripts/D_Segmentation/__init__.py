"""Independent D_Segmentation package built on C_Representation outputs."""

from .config import AUXILIARY_LABELS, PRIMARY_LABELS, PRIMARY_LABEL_TO_INDEX, SegmentationConfig
from .pipeline import process_all_subsets, process_one_sequence, process_subset, segment_sequence

__all__ = [
    "AUXILIARY_LABELS",
    "PRIMARY_LABELS",
    "PRIMARY_LABEL_TO_INDEX",
    "SegmentationConfig",
    "process_all_subsets",
    "process_one_sequence",
    "process_subset",
    "segment_sequence",
]
