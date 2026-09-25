from edge_ai_compression.compression.combined import SequentialCompression
from edge_ai_compression.compression.distillation import distill_student_inplace
from edge_ai_compression.compression.pruning import apply_global_unstructured_pruning

__all__ = [
    "SequentialCompression",
    "apply_global_unstructured_pruning",
    "distill_student_inplace",
]
