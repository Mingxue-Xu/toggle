"""
Tucker Decomposition Plugin for Toggle Architecture

This plugin implements Tucker decomposition compression strategy, migrated
from the original factorization.py module.
"""
import time
import torch
import tensorly as tl
from typing import List, Dict, Any, Optional, Union

from .base import LowRankCompressionPlugin, CompressedTensor
from .tensorly_backend import set_tensorly_backend


class Tucker(LowRankCompressionPlugin):
    """
    Tucker decomposition compression plugin
    
    Migrates tucker_decomposition functionality from factorization.py
    to the new plugin-based architecture.
    """
    
    def __init__(self, tucker_ranks: Union[List[int], int], device: str = 'cpu', backend: str = 'pytorch', **kwargs):
        """
        Initialize Tucker plugin
        
        Args:
            tucker_ranks: Tucker ranks for decomposition (list or single int)
            device: Computation device ('cpu', 'cuda', etc.)
            backend: Tensorly backend ('pytorch', 'numpy')
            **kwargs: Additional configuration
        """
        super().__init__(name="Tucker", **kwargs)
        
        # Handle both int and list formats for tucker_ranks
        if isinstance(tucker_ranks, int):
            self.tucker_ranks = tucker_ranks
        elif isinstance(tucker_ranks, (list, tuple)):
            self.tucker_ranks = list(tucker_ranks)
        else:
            raise ValueError("tucker_ranks must be int or list of ints")
            
        self.device = device
        self.backend = backend
        
        # Set tensorly backend
        set_tensorly_backend(backend)
        
        # Validation
        if isinstance(self.tucker_ranks, list):
            if not all(isinstance(r, int) and r > 0 for r in self.tucker_ranks):
                raise ValueError("tucker_ranks must contain only positive integers")
        elif not isinstance(self.tucker_ranks, int) or self.tucker_ranks <= 0:
            raise ValueError("tucker_ranks must be a positive integer")
    
    def do_execute(self, context, **params):
        """Plugin interface implementation for Tucker compression"""
        tensor = params.get('tensor', None)
        if tensor is None:
            raise ValueError("Tucker compression requires a 'tensor' parameter")
        return self.compress(tensor)
    
    def compress(self, tensor: torch.Tensor, **params) -> CompressedTensor:
        """
        Compress tensor using Tucker decomposition
        
        Args:
            tensor: Input tensor to compress
            **params: Additional parameters (ignored for compatibility)
            
        Returns:
            CompressedTensor with Tucker factors (core tensor and factor matrices)
        """
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        
        # Perform Tucker decomposition
        tucker_factors = self._tucker_decomposition(tensor, self.tucker_ranks)
        
        # Calculate compression ratio
        original_size = tensor.numel()
        compressed_size = self._calculate_tucker_size(tucker_factors)
        compression_ratio = original_size / compressed_size
        
        return CompressedTensor(
            factors=tucker_factors,
            method="tucker",
            original_shape=tensor.shape,
            compression_ratio=compression_ratio,
            metadata={
                "tucker_ranks": self.tucker_ranks,
                "device": self.device,
                "backend": self.backend,
                "original_size": original_size,
                "compressed_size": compressed_size
            }
        )
    
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress tensor from Tucker factors
        
        Args:
            compressed: CompressedTensor with Tucker factors
            
        Returns:
            Reconstructed tensor
        """
        if compressed.method != "tucker":
            raise ValueError(f"Cannot decompress tensor compressed with method '{compressed.method}'")
        
        return self._reconstruct_from_tucker_factors(compressed.factors)
    
    def estimate_compression_ratio(self, tensor: torch.Tensor, **params) -> float:
        """
        Estimate compression ratio without performing actual compression
        
        Args:
            tensor: Input tensor
            **params: Additional parameters (ignored)
            
        Returns:
            Estimated compression ratio
        """
        original_size = tensor.numel()
        
        # Estimate Tucker factor sizes
        estimated_compressed_size = self._estimate_tucker_size(tensor.shape, self.tucker_ranks)
        
        return original_size / estimated_compressed_size if estimated_compressed_size > 0 else 0.0
    
    def validate_tensor_compatibility(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor is compatible with Tucker decomposition
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True if compatible, False otherwise
        """
        # Tucker requires at least 2D tensors
        if len(tensor.shape) < 2:
            return False
        
        # Check if tucker ranks are compatible with tensor dimensions
        if isinstance(self.tucker_ranks, list):
            if len(self.tucker_ranks) != len(tensor.shape):
                return False
            # All ranks should be positive and not exceed tensor dimensions
            if not all(1 <= rank <= dim for rank, dim in zip(self.tucker_ranks, tensor.shape)):
                return False
        else:
            # Single rank should be positive and reasonable
            if not (1 <= self.tucker_ranks <= min(tensor.shape)):
                return False
        
        return True
    
    def _tucker_decomposition(self, tensor: torch.Tensor, ranks: Union[List[int], int]) -> Any:
        """
        Internal Tucker decomposition implementation
        
        Migrated from original tucker_decomposition function
        
        Args:
            tensor: Input tensor
            ranks: Tucker ranks
            
        Returns:
            Tucker decomposition object (core tensor and factor matrices)
        """
        # Perform Tucker decomposition using tensorly
        tucker_factors = tl.decomposition.tucker(tensor, rank=ranks)
        
        return tucker_factors
    
    def _reconstruct_from_tucker_factors(self, tucker_factors: Any) -> torch.Tensor:
        """
        Reconstruct tensor from Tucker factors
        
        Args:
            tucker_factors: Tucker decomposition object
            
        Returns:
            Reconstructed tensor
        """
        # Use tensorly's reconstruction function
        if hasattr(tl, 'tucker_to_tensor'):
            return tl.tucker_to_tensor(tucker_factors)
        else:
            # Alternative approach for different tensorly versions
            return self._manual_tucker_reconstruction(tucker_factors)
    
    def _manual_tucker_reconstruction(self, tucker_factors) -> torch.Tensor:
        """
        Manual Tucker reconstruction for compatibility
        
        Args:
            tucker_factors: Tucker factors (core tensor and factor matrices)
            
        Returns:
            Reconstructed tensor
        """
        # Extract core tensor and factors
        if hasattr(tucker_factors, 'core') and hasattr(tucker_factors, 'factors'):
            core = tucker_factors.core
            factors = tucker_factors.factors
        elif isinstance(tucker_factors, (tuple, list)) and len(tucker_factors) >= 2:
            core = tucker_factors[0]
            factors = tucker_factors[1]
        else:
            raise ValueError("Unsupported Tucker factor format")
        
        # Reconstruct by contracting core with factor matrices
        result = core
        for i, factor in enumerate(factors):
            result = torch.tensordot(result, factor.T, dims=([0], [0]))
            # Move the contracted dimension to the end
            if i < len(factors) - 1:
                result = result.moveaxis(0, -1)
        
        return result
    
    def _calculate_tucker_size(self, tucker_factors: Any) -> int:
        """
        Calculate total size of Tucker factors
        
        Args:
            tucker_factors: Tucker decomposition object
            
        Returns:
            Total size of Tucker factors
        """
        total_size = 0
        
        # Extract core tensor and factors
        if hasattr(tucker_factors, 'core') and hasattr(tucker_factors, 'factors'):
            core = tucker_factors.core
            factors = tucker_factors.factors
        elif isinstance(tucker_factors, (tuple, list)) and len(tucker_factors) >= 2:
            core = tucker_factors[0]
            factors = tucker_factors[1]
        else:
            # Fallback estimation
            return 0
        
        # Add core tensor size
        total_size += core.numel()
        
        # Add factor matrix sizes
        if isinstance(factors, (list, tuple)):
            for factor in factors:
                total_size += factor.numel()
        else:
            total_size += factors.numel()
        
        return total_size
    
    def create_layer(self, compressed_data, original_shape):
        """
        Create FactorEmbedding or FactorLinear from Tucker data
        
        Args:
            compressed_data: CompressedTensor with Tucker factors (core + factor matrices)
            original_shape: Original tensor shape [out_features, in_features] or [num_embeddings, embedding_dim]
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original shape
        """
        from ...framework.layers import FactorLayer, FactorEmbedding, FactorLinear, Factor
        
        # Create factors from Tucker decomposition
        factors = []
        for factor_tensor in compressed_data.factors:
            factor = Factor(_weight=factor_tensor.clone())
            factors.append(factor)
        
        factor_layer = FactorLayer(
            factors=factors,
            func_name="tucker_contract"
        )
        
        # Determine layer type based on original shape
        if len(original_shape) == 2:
            if original_shape[0] > original_shape[1]:  # Likely embedding: [vocab_size, embedding_dim]
                return FactorEmbedding.from_pretrained(
                    factor_layer,
                    num_embeddings=original_shape[0] 
                )
            else:  # Linear layer: [out_features, in_features]
                return FactorLinear.from_pretrained(
                    factor_layer, 
                    in_features=original_shape[1],
                    out_features=original_shape[0]
                )
        else:
            raise ValueError(f"Unsupported tensor shape for layer creation: {original_shape}")
    
    def _estimate_tucker_size(self, shape: tuple, ranks: Union[List[int], int]) -> int:
        """
        Estimate total size of Tucker factors
        
        Args:
            shape: Original tensor shape
            ranks: Tucker ranks
            
        Returns:
            Estimated total factor size
        """
        if isinstance(ranks, int):
            # Single rank case - use same rank for all modes
            tucker_ranks = [min(ranks, dim) for dim in shape]
        else:
            tucker_ranks = ranks
        
        if len(tucker_ranks) != len(shape):
            return 0
        
        # Core tensor size
        core_size = 1
        for rank in tucker_ranks:
            core_size *= rank
        
        # Factor matrix sizes
        factor_sizes = sum(dim * rank for dim, rank in zip(shape, tucker_ranks))
        
        return core_size + factor_sizes
