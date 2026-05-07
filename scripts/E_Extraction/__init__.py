"""Public entry points for the E_Extraction module."""

from .config import ExtractionConfig
from .pipeline import SequenceExtractionResult, extract_sequence, process_all_subsets, process_one_sequence, process_subset

__all__ = [
    "ExtractionConfig",
    "SequenceExtractionResult",
    "extract_sequence",
    "process_one_sequence",
    "process_subset",
    "process_all_subsets",
]