"""
Analysis plugins for Toggle architecture

This package contains plugins for analyzing model properties, weights,
and compression characteristics.
"""

from .activation_metrics import ActivationMetricsPlugin
from .memory_inference import MemoryInferenceProfiler
from .layer_svd_rank_decider import LayerSVDRankDecider
from .fisher_information import FisherInformationPlugin

# Lazy attribute access to avoid circular import with compression.pruning
def __getattr__(name):  # PEP 562
    if name == "PruningPlugin":
        from ..compression.pruning import PruningPlugin as _PP
        return _PP
    raise AttributeError(name)

__all__ = [
    "ActivationMetricsPlugin",
    "PruningPlugin",
    "MemoryInferenceProfiler",
    "LayerSVDRankDecider",
    "FisherInformationPlugin",
]
