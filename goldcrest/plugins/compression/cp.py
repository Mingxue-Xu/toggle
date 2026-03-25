"""
CP Decomposition Plugin for Goldcrest Architecture

This plugin implements CP (CANDECOMP/PARAFAC) decomposition compression strategy, 
migrated from the original factorization.py module.
"""
import torch
import tensorly as tl
from typing import List, Dict, Any, Optional, Union

from .base import LowRankCompressionPlugin, CompressedTensor
from .tensorly_backend import set_tensorly_backend


class CP(LowRankCompressionPlugin):
    """
    CP decomposition compression plugin
    
    Migrates cp_decomposition functionality from factorization.py
    to the new plugin-based architecture.
    """
    
    def __init__(self, cp_rank: int, device: str = 'cpu', backend: str = 'pytorch', **kwargs):
        """
        Initialize CP plugin
        
        Args:
            cp_rank: CP rank for decomposition
            device: Computation device ('cpu', 'cuda', etc.)
            backend: Tensorly backend ('pytorch', 'numpy')
            **kwargs: Additional configuration
        """
        super().__init__(name="CP", **kwargs)
        self.cp_rank = cp_rank
        self.device = device
        self.backend = backend
        
        # Set tensorly backend
        set_tensorly_backend(backend)
        
        # Validation
        if not isinstance(cp_rank, int) or cp_rank <= 0:
            raise ValueError("cp_rank must be a positive integer")
    
    def do_execute(self, context, **params):
        """Plugin interface implementation for CP compression"""
        tensor = params.get('tensor', None)
        if tensor is None:
            raise ValueError("CP compression requires a 'tensor' parameter")
        return self.compress(tensor)
    
    def compress(self, tensor: torch.Tensor, **params) -> CompressedTensor:
        """
        Compress tensor using CP decomposition
        
        Args:
            tensor: Input tensor to compress
            **params: Additional parameters (ignored for compatibility)
            
        Returns:
            CompressedTensor with CP factors
        """
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        
        # Perform CP decomposition
        cp_factors = self._cp_decomposition(tensor, self.cp_rank)
        
        # Calculate compression ratio
        original_size = tensor.numel()
        compressed_size = self._calculate_cp_size(cp_factors, tensor.shape)
        compression_ratio = original_size / compressed_size
        
        return CompressedTensor(
            factors=cp_factors,
            method="cp",
            original_shape=tensor.shape,
            compression_ratio=compression_ratio,
            metadata={
                "cp_rank": self.cp_rank,
                "device": self.device,
                "backend": self.backend,
                "original_size": original_size,
                "compressed_size": compressed_size
            }
        )
    
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress tensor from CP factors
        
        Args:
            compressed: CompressedTensor with CP factors
            
        Returns:
            Reconstructed tensor
        """
        if compressed.method != "cp":
            raise ValueError(f"Cannot decompress tensor compressed with method '{compressed.method}'")
        
        return self._reconstruct_from_cp_factors(compressed.factors)
    
    def validate_tensor_compatibility(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor is compatible with CP decomposition
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True if compatible, False otherwise
        """
        # CP requires at least 2D tensors (though it works best with higher-order tensors)
        if len(tensor.shape) < 2:
            return False
        
        # CP rank should not exceed the minimum tensor dimension
        if self.cp_rank > min(tensor.shape):
            return False
        
        return True
    
    def _cp_decomposition(self, tensor: torch.Tensor, rank: int) -> Any:
        """
        Internal CP decomposition implementation
        
        Migrated from original cp_decomposition function
        
        Args:
            tensor: Input tensor
            rank: CP rank
            
        Returns:
            CP decomposition object (weights and factor matrices)
        """
        # Perform CP decomposition using tensorly (parafac is the correct function name)
        cp_factors = tl.decomposition.parafac(tensor, rank=rank)
        
        return cp_factors
    
    def _reconstruct_from_cp_factors(self, cp_factors: Any) -> torch.Tensor:
        """
        Reconstruct tensor from CP factors
        
        Args:
            cp_factors: CP decomposition object
            
        Returns:
            Reconstructed tensor
        """
        # Use tensorly's reconstruction function
        if hasattr(tl, 'cp_to_tensor'):
            return tl.cp_to_tensor(cp_factors)
        else:
            # Alternative approach for different tensorly versions
            return self._manual_cp_reconstruction(cp_factors)
    
    def _manual_cp_reconstruction(self, cp_factors) -> torch.Tensor:
        """
        Manual CP reconstruction for compatibility
        
        Args:
            cp_factors: CP factors (weights and factor matrices)
            
        Returns:
            Reconstructed tensor
        """
        # Extract weights and factors
        if hasattr(cp_factors, 'weights') and hasattr(cp_factors, 'factors'):
            weights = cp_factors.weights
            factors = cp_factors.factors
        elif isinstance(cp_factors, (tuple, list)) and len(cp_factors) >= 2:
            weights = cp_factors[0]
            factors = cp_factors[1]
        else:
            raise ValueError("Unsupported CP factor format")
        
        # Reconstruct tensor using Khatri-Rao product and sum
        # For each rank component, compute outer product and weight
        reconstructed = None
        
        for r in range(self.cp_rank):
            # Extract r-th column from each factor matrix
            component_factors = [factor[:, r] for factor in factors]
            
            # Compute outer product of all factor vectors
            component_tensor = component_factors[0]
            for i in range(1, len(component_factors)):
                component_tensor = torch.outer(component_tensor.flatten(), component_factors[i])
                # Reshape to maintain tensor structure
                new_shape = component_tensor.shape[:-1] + factors[i].shape[:-1]
                component_tensor = component_tensor.view(new_shape)
            
            # Apply weight and accumulate
            if weights is not None and len(weights) > r:
                component_tensor = weights[r] * component_tensor
            
            if reconstructed is None:
                reconstructed = component_tensor
            else:
                reconstructed += component_tensor
        
        return reconstructed
    
    def _calculate_cp_size(self, cp_factors: Any, original_shape: tuple) -> int:
        """
        Calculate total size of CP factors
        
        Args:
            cp_factors: CP decomposition object
            original_shape: Original tensor shape
            
        Returns:
            Total size of CP factors
        """
        total_size = 0
        
        # Extract weights and factors
        if hasattr(cp_factors, 'weights') and hasattr(cp_factors, 'factors'):
            weights = cp_factors.weights
            factors = cp_factors.factors
        elif isinstance(cp_factors, (tuple, list)) and len(cp_factors) >= 2:
            weights = cp_factors[0]
            factors = cp_factors[1]
        else:
            # Fallback estimation
            return sum(dim * self.cp_rank for dim in original_shape)
        
        # Add weights size
        if weights is not None:
            if hasattr(weights, 'numel'):
                total_size += weights.numel()
            else:
                total_size += len(weights) if weights else 0
        
        # Add factor matrix sizes
        if isinstance(factors, (list, tuple)):
            for factor in factors:
                total_size += factor.numel()
        else:
            total_size += factors.numel()
        
        return total_size
    
    def _estimate_cp_size(self, shape: tuple, rank: int) -> int:
        """
        Estimate total size of CP factors
        
        Args:
            shape: Original tensor shape
            rank: CP rank
            
        Returns:
            Estimated total factor size
        """
        # Weights size
        weights_size = rank
        
        # Factor matrix sizes (each mode has a factor matrix of size mode_dim x rank)
        factor_sizes = sum(dim * rank for dim in shape)
        
        return weights_size + factor_sizes
    
    def create_layer(self, compressed_data, original_shape):
        """
        Create FactorEmbedding or FactorLinear from CP data
        
        Args:
            compressed_data: CompressedTensor with CP factors
            original_shape: Original tensor shape [out_features, in_features] or [num_embeddings, embedding_dim]
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original shape
        """
        from ...framework.layers import FactorLayer, FactorEmbedding, FactorLinear, Factor
        
        # Create factors from CP decomposition
        factors = []
        for factor_tensor in compressed_data.factors:
            factor = Factor(_weight=factor_tensor.clone())
            factors.append(factor)
        
        factor_layer = FactorLayer(
            factors=factors,
            func_name="cp_contract"
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
