"""
Compression plugins for Toggle architecture

This package contains plugins for tensor compression and model compression
using various decomposition methods like tensor-train, Tucker, and CP.
"""

from .base import (
    CompressedTensor,
    TensorCompressionResult,
    LowRankCompressionPlugin,
    ModelCompressionPlugin
)

# Import concrete plugin implementations
from .tensor_train import TensorTrain
from .tucker import Tucker
from .cp import CP
from .svd import SVD
from .tensorizer import Tensorizer
from .consolidator import ModelConsolidator
from .pruning import PruningPlugin
from .svd_activation import ActivationDrivenSVDWeightsCompressionPlugin

# ASVD and SVD-LLM plugins
from .calibration_collector import CalibrationCollectorPlugin
from .svd_activation_scaling import ActivationScalingPlugin
from .svd_data_whitening import DataWhiteningPlugin
from .svd_closed_form_update import ClosedFormUpdatePlugin
from .svd_ppl_sensitivity import PPLSensitivityPlugin
from .svd_binary_search_rank import BinarySearchRankPlugin
from .svdllm_pipeline import SVDLLMPipelinePlugin

# Plugin registration function
def register_compression_plugins(registry):
    """
    Register all compression plugins with the registry
    
    Args:
        registry: PluginRegistry instance to register plugins with
    """
    registry.register(TensorTrain, "TensorTrain")
    registry.register(Tucker, "Tucker")
    registry.register(CP, "CP")
    registry.register(SVD, "SVD")
    registry.register(Tensorizer, "Tensorizer")
    registry.register(ModelConsolidator, "ModelConsolidator")
    registry.register(PruningPlugin, "Pruning")
    registry.register(ActivationDrivenSVDWeightsCompressionPlugin, "ActivationSVDRankCompression")

    # ASVD and SVD-LLM plugins
    registry.register(CalibrationCollectorPlugin, "CalibrationCollector")
    registry.register(ActivationScalingPlugin, "ActivationScaling")
    registry.register(DataWhiteningPlugin, "DataWhitening")
    registry.register(ClosedFormUpdatePlugin, "ClosedFormUpdate")
    registry.register(PPLSensitivityPlugin, "PPLSensitivity")
    registry.register(BinarySearchRankPlugin, "BinarySearchRank")
    registry.register(SVDLLMPipelinePlugin, "SVDLLMPipeline")

__all__ = [
    'CompressedTensor',
    'TensorCompressionResult',
    'LowRankCompressionPlugin',
    'ModelCompressionPlugin',
    'TensorTrain',
    'Tucker',
    'CP',
    'SVD',
    'Tensorizer',
    'ModelConsolidator',
    'PruningPlugin',
    'ActivationDrivenSVDWeightsCompressionPlugin',
    # ASVD and SVD-LLM plugins
    'CalibrationCollectorPlugin',
    'ActivationScalingPlugin',
    'DataWhiteningPlugin',
    'ClosedFormUpdatePlugin',
    'PPLSensitivityPlugin',
    'BinarySearchRankPlugin',
    'SVDLLMPipelinePlugin',
    'register_compression_plugins'
]
