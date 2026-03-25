"""
Evaluation plugins for Goldcrest architecture (lazy import interface).

This module exposes evaluation classes without eagerly importing heavy
submodules to avoid side effects during test discovery (e.g., model
registry conflicts). Attributes are loaded on first access.
"""

from .base import (
    ModelEvaluationResult,
    ModelEvaluationPlugin,
    BaselineModelEvaluationPlugin,
    CompressedModelEvaluationPlugin,
)

__all__ = [
    # Base classes
    "ModelEvaluationResult",
    "ModelEvaluationPlugin",
    "BaselineModelEvaluationPlugin",
    "CompressedModelEvaluationPlugin",
    # Lazily imported plugin classes
    "UncompressedModelProfile",
    "CompressedModelProfile",
    "LMHarness",
    "LMEvaluator",
    "LMLMHarnessModelAdapter",
    "CSVLogger",
    "ResultComparator",
]


def __getattr__(name):  # PEP 562 lazy attribute loading
    if name in {"UncompressedModelProfile"}:
        from .baseline_eval import UncompressedModelProfile
        return locals()[name]
    if name in {"CompressedModelProfile"}:
        from .compressed_eval import CompressedModelProfile
        return locals()[name]
    if name in {"LMHarness", "LMEval", "LMEvaluator", "LMLMHarnessModelAdapter"}:
        from .lm_eval import LMHarness, LMEvaluator, LMLMHarnessModelAdapter
        return locals()[name]
    if name in {"CSVLogger", "ResultComparator"}:
        from .csv_logger import CSVLogger, ResultComparator
        return locals()[name]
    raise AttributeError(name)
