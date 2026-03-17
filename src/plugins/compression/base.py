"""
Compression Strategy Base Classes for Toggle Architecture

This module provides base classes for low-rank tensor compression and model compression
plugins within the event-driven pipeline system.
"""
import torch
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from ...framework.plugins import Plugin, PluginMetadata



@dataclass
class CompressedTensor:
    """
    Container for compressed tensor data
    """
    factors: Union[List[torch.Tensor], torch.Tensor, Any]
    method: str
    original_shape: tuple
    compression_ratio: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TensorCompressionResult:
    """
    Result container for tensor compression operations
    """
    compressed_tensors: Dict[str, CompressedTensor]
    total_compression_ratio: float
    compression_time: float
    method: str
    parameters: Dict[str, Any]


class LowRankCompressionPlugin(Plugin):
    """
    Base class for low-rank tensor compression strategies
    
    Combines Plugin functionality with Strategy interface to support
    both tensor-level and model-level compression operations.
    Provides standard interface for compressing tensors using various
    low-rank decomposition methods like tensor-train, Tucker, CP, SVD, etc.
    """
    
    def __init__(self, **config):
        """
        Initialize low-rank compression strategy
        
        Args:
            **config: Strategy-specific configuration
        """
        # Initialize Plugin
        super().__init__(**config)
        
        # Initialize strategy-like properties
        self.config = config
        self.logger = logging.getLogger(f"strategy.{self.__class__.__name__.lower()}")
        
        self._compression_stats = {
            "total_compressions": 0,
            "total_compression_time": 0.0,
            "average_compression_ratio": 0.0
        }
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Low-rank tensor compression plugin",
            category="compression"
        )
    
    def _execute_impl(self, tensor: torch.Tensor, **params) -> CompressedTensor:
        """
        Execute tensor compression
        
        Args:
            tensor: Tensor to compress
            **params: Compression parameters
            
        Returns:
            Compressed tensor result
        """
        import time
        start_time = time.time()
        
        # Validate tensor
        self._validate_tensor(tensor)
        
        # Emit compression started event
        if self.event_bus:
            self.event_bus.emit(
                "compression.started",
                {
                    "plugin": self.name,
                    "tensor_shape": list(tensor.shape),
                    "tensor_size": tensor.numel()
                },
                source=self.name
            )
        
        try:
            # Perform compression
            compressed = self.compress(tensor, **params)
            
            # Calculate timing and update stats
            compression_time = time.time() - start_time
            self._update_stats(compressed, compression_time)
            
            # Emit completion event
            if self.event_bus:
                self.event_bus.emit(
                    "compression.completed",
                    {
                        "plugin": self.name,
                        "compression_ratio": compressed.compression_ratio,
                        "compression_time": compression_time
                    },
                    source=self.name
                )
            
            return compressed
            
        except Exception as e:
            if self.event_bus:
                self.event_bus.emit(
                    "compression.failed",
                    {
                        "plugin": self.name,
                        "error": str(e)
                    },
                    source=self.name
                )
            raise
    
    # Strategy-level interface (for UnifiedStrategyFactory compatibility)
    def execute(self, context, model=None, target_modules=None, **params):
        """
        Strategy-level execution for model compression
        
        Args:
            context: Pipeline context
            model: Model to compress
            target_modules: List of target modules to compress
            **params: Strategy-specific parameters
            
        Returns:
            Model compression results
        """
        if not model:
            raise ValueError("Model is required for compression")
        
        self.logger.info(f"Applying {self.__class__.__name__} compression")
        return self.compress_model(model, target_modules=target_modules, **params)


# Backward-compat name expected by some tests
class TensorCompressionPlugin(LowRankCompressionPlugin):
    pass
    
    def compress_model(self, model, target_modules=None, **params):
        """
        Compress entire model using this low-rank strategy
        
        Args:
            model: Model to compress
            target_modules: List of target modules to compress
            **params: Compression parameters
            
        Returns:
            Model compression results
        """
        # Default implementation - can be overridden by subclasses
        # This provides model-level compression by applying tensor-level
        # compression to individual layers
        
        if target_modules is None:
            target_modules = self._get_default_target_modules()
        
        compressed_tensors = {}
        total_original_size = 0
        total_compressed_size = 0
        compression_stats = {}
        
        for module_name in target_modules:
            module = self._get_module_by_name(model, module_name)
            if module and hasattr(module, 'weight') and module.weight is not None:
                # Apply tensor-level compression to this module's weight
                compressed = self.compress(module.weight, **params)
                
                compressed_tensors[module_name] = compressed
                
                original_size = module.weight.numel()
                if hasattr(compressed, 'factors'):
                    if isinstance(compressed.factors, list):
                        compressed_size = sum(factor.numel() for factor in compressed.factors)
                    else:
                        compressed_size = compressed.factors.numel()
                elif hasattr(compressed, 'size'):
                    compressed_size = compressed.size()
                else:
                    compressed_size = original_size  # Fallback
                
                total_original_size += original_size
                total_compressed_size += compressed_size
                compression_stats[module_name] = {
                    "compression_ratio": original_size / compressed_size if compressed_size > 0 else 1.0,
                    "original_size": original_size,
                    "compressed_size": compressed_size
                }
        
        overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
        
        return TensorCompressionResult(
            compressed_tensors=compressed_tensors,
            total_compression_ratio=overall_compression_ratio,
            compression_time=0.0,  # Will be set by caller
            method=f"{self.__class__.__name__.lower()}",
            parameters={
                "target_modules": target_modules,
                "compression_stats": compression_stats,
                "total_original_size": total_original_size,
                "total_compressed_size": total_compressed_size
            }
        )
    
    def _get_module_by_name(self, model, module_name: str):
        """Get a module from model by name"""
        try:
            parts = module_name.split('.')
            current_module = model
            
            for part in parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    return None
            
            return current_module
        except Exception:
            return None
    
    def _get_default_target_modules(self) -> List[str]:
        """Get default list of module names to compress - can be overridden"""
        return [
            "transformer.wte",      # Word token embedding
            "transformer.wpe",      # Word position embedding
            "lm_head",              # Language model head
            "score",                # Score/classification head
            "classifier"             # Classifier head
        ]
    
    # Tensor-level interface (existing Plugin interface)
    @abstractmethod
    def compress(self, tensor: torch.Tensor, **params) -> CompressedTensor:
        """
        Compress tensor using the low-rank strategy
        
        Args:
            tensor: Input tensor to compress
            **params: Strategy-specific parameters
            
        Returns:
            Compressed tensor container
        """
        pass
    
    def create_layer(self, compressed_data, original_shape):
        """
        Create FactorEmbedding or FactorLinear from compressed data
        
        Args:
            compressed_data: CompressedTensor with factorized data
            original_shape: Original tensor shape [out_features, in_features] or [num_embeddings, embedding_dim]
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original shape
            
        Note:
            Default implementation - should be overridden by subclasses for method-specific layer creation
        """
        raise NotImplementedError(f"Layer creation not implemented for {self.__class__.__name__}")
    
    @abstractmethod
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress a compressed tensor
        
        Args:
            compressed: Compressed tensor container
            
        Returns:
            Reconstructed tensor
        """
        pass
    
    @abstractmethod
    def estimate_compression_ratio(self, tensor: torch.Tensor, **params) -> float:
        """
        Estimate compression ratio without performing actual compression
        
        Args:
            tensor: Input tensor
            **params: Strategy-specific parameters
            
        Returns:
            Estimated compression ratio
        """
        pass
    
    def validate_tensor_compatibility(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor is compatible with this compression strategy
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True if compatible, False otherwise
        """
        # Default implementation - can be overridden
        return len(tensor.shape) >= 2  # Most methods need at least 2D tensors
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self._compression_stats.copy()
    
    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        """Validate input tensor"""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if tensor.numel() == 0:
            raise ValueError("Tensor cannot be empty")
        
        if not self.validate_tensor_compatibility(tensor):
            raise ValueError(f"Tensor with shape {tensor.shape} is not compatible with {self.__class__.__name__}")
    
    def _update_stats(self, compressed: CompressedTensor, compression_time: float) -> None:
        """Update compression statistics"""
        self._compression_stats["total_compressions"] += 1
        self._compression_stats["total_compression_time"] += compression_time
        
        if compressed.compression_ratio:
            # Update average compression ratio
            count = self._compression_stats["total_compressions"]
            current_avg = self._compression_stats["average_compression_ratio"]
            new_avg = (current_avg * (count - 1) + compressed.compression_ratio) / count
            self._compression_stats["average_compression_ratio"] = new_avg


class ModelCompressionPlugin(Plugin):
    """
    Base class for full model compression plugins
    
    Handles compression of entire models by applying tensor compression
    strategies to specific layers or components.
    """
    
    def __init__(self, target_modules: Optional[List[str]] = None, **kwargs):
        """
        Initialize model compression strategy
        
        Args:
            target_modules: List of module names to compress (None for default)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.target_modules = target_modules or self._get_default_target_modules()
        self._compression_results = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Model compression plugin",
            category="compression"
        )
    
    def _execute_impl(self, model: torch.nn.Module, **params) -> TensorCompressionResult:
        """
        Execute model compression
        
        Args:
            model: Model to compress
            **params: Compression parameters
            
        Returns:
            Model compression results
        """
        import time
        start_time = time.time()
        
        # Emit compression started event
        if self.event_bus:
            self.event_bus.emit(
                "model_compression.started",
                {
                    "plugin": self.name,
                    "target_modules": self.target_modules,
                    "model_type": model.__class__.__name__
                },
                source=self.name
            )
        
        try:
            # Perform model compression
            result = self.compress_model(model, **params)
            
            # Calculate timing
            compression_time = time.time() - start_time
            result.compression_time = compression_time
            
            # Store results
            self._compression_results[model.__class__.__name__] = result
            
            # Emit completion event
            if self.event_bus:
                self.event_bus.emit(
                    "model_compression.completed",
                    {
                        "plugin": self.name,
                        "compression_ratio": result.total_compression_ratio,
                        "compression_time": compression_time,
                        "compressed_modules": list(result.compressed_tensors.keys())
                    },
                    source=self.name
                )
            
            return result
            
        except Exception as e:
            if self.event_bus:
                self.event_bus.emit(
                    "model_compression.failed",
                    {
                        "plugin": self.name,
                        "error": str(e)
                    },
                    source=self.name
                )
            raise
    
    @abstractmethod
    def compress_model(self, model: torch.nn.Module, **params) -> TensorCompressionResult:
        """
        Compress the entire model
        
        Args:
            model: Model to compress
            **params: Compression parameters
            
        Returns:
            Compression results
        """
        pass
    
    @abstractmethod
    def _get_default_target_modules(self) -> List[str]:
        """
        Get default list of module names to compress
        
        Returns:
            List of module names
        """
        pass
    
    def get_compression_history(self) -> Dict[str, TensorCompressionResult]:
        """Get compression history"""
        return self._compression_results.copy()
