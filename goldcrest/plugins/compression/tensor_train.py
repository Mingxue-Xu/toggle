"""
Tensor-Train Decomposition Plugin for Goldcrest Architecture

This plugin implements tensor-train decomposition compression strategy, migrated
from the original factorization.py module.
"""
import time
import torch
import tensorly as tl
from typing import List, Dict, Any, Optional, Union

from .base import LowRankCompressionPlugin, CompressedTensor
from .tensorly_backend import set_tensorly_backend


class TensorTrain(LowRankCompressionPlugin):
    """
    Tensor-Train decomposition compression plugin
    
    Migrates tensor_train_decomposition functionality from factorization.py
    to the new plugin-based architecture.
    """
    
    def __init__(self, tensor_ranks: List[int], device: str = 'cpu', backend: str = 'pytorch', **kwargs):
        """
        Initialize TensorTrain plugin
        
        Args:
            tensor_ranks: Tensor-train ranks for decomposition
            device: Computation device ('cpu', 'cuda', etc.)
            backend: Tensorly backend ('pytorch', 'numpy')
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.tensor_ranks = tensor_ranks
        self.device = device
        self.backend = backend
        
        # Set tensorly backend
        set_tensorly_backend(backend)
        
        # Validation
        if not tensor_ranks or not all(isinstance(r, int) and r > 0 for r in tensor_ranks):
            raise ValueError("tensor_ranks must be a list of positive integers")
    
    def do_execute(self, context, **params):
        """Plugin interface implementation for Tensor-Train compression"""
        tensor = params.get('tensor', None)
        if tensor is None:
            raise ValueError("Tensor-Train compression requires a 'tensor' parameter")
        return self.compress(tensor)
    
    def compress(self, tensor: torch.Tensor, **params) -> CompressedTensor:
        """
        Compress tensor using tensor-train decomposition
        
        Args:
            tensor: Input tensor to compress
            **params: Additional parameters (ignored for compatibility)
            
        Returns:
            CompressedTensor with tensor-train factors
        """
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        original_shape = tensor.shape
        
        # Handle 2D tensors by reshaping them to higher dimensions for TT
        if len(tensor.shape) == 2:
            # Reshape 2D tensor to 3D for TT decomposition
            h, w = tensor.shape
            # Find a suitable factorization for tensor-train
            if h > w:
                new_shape = (h // 2, 2, w) if h % 2 == 0 else (h // 3, 3, w) if h % 3 == 0 else (h, 1, w)
            else:
                new_shape = (h, w // 2, 2) if w % 2 == 0 else (h, w // 3, 3) if w % 3 == 0 else (h, w, 1)
            
            tensor = tensor.reshape(new_shape)
            
            # Adjust ranks for the reshaped tensor
            working_ranks = [1] + [min(4, dim) for dim in new_shape[:-1]] + [1]
        else:
            working_ranks = self.tensor_ranks
        
        # Perform tensor-train decomposition
        factors = self._tensor_train_decomposition(tensor, working_ranks)
        
        # Calculate compression ratio
        original_size = torch.numel(torch.zeros(original_shape))
        compressed_size = sum(factor.numel() for factor in factors)
        compression_ratio = original_size / compressed_size
        
        return CompressedTensor(
            factors=factors,
            method="tensor_train",
            original_shape=original_shape,
            compression_ratio=compression_ratio,
            metadata={
                "tensor_ranks": working_ranks,
                "device": self.device,
                "backend": self.backend,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "reshaped_for_tt": len(original_shape) == 2,
                "tt_tensor_shape": tensor.shape
            }
        )
    
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress tensor from tensor-train factors
        
        Args:
            compressed: CompressedTensor with tensor-train factors
            
        Returns:
            Reconstructed tensor
        """
        if compressed.method != "tensor_train":
            raise ValueError(f"Cannot decompress tensor compressed with method '{compressed.method}'")
        
        return self._reconstruct_from_factors(compressed.factors)
    
    def validate_tensor_compatibility(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor is compatible with tensor-train decomposition
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True if compatible, False otherwise
        """
        # Tensor-train requires at least 2D tensors
        if len(tensor.shape) < 2:
            return False
        
        # Check if tensor ranks are compatible with tensor dimensions
        # For 2D tensors, we need to reshape to higher dimensions first
        if len(tensor.shape) == 2:
            # For 2D tensors, we'll need to reshape them for tensor-train
            return True  # Let the compression method handle reshaping
        
        if len(self.tensor_ranks) != len(tensor.shape) + 1:
            return False
        
        # All ranks should be positive and reasonable
        if not all(1 <= rank <= min(tensor.shape) for rank in self.tensor_ranks[1:-1]):
            return False
        
        # First and last ranks should be 1 for standard tensor-train
        if self.tensor_ranks[0] != 1 or self.tensor_ranks[-1] != 1:
            return False
        
        return True
    
    def _tensor_train_decomposition(self, tensor: torch.Tensor, ranks: List[int]):
        """
        Internal tensor-train decomposition implementation

        Migrated from original tensor_train_decomposition function

        Args:
            tensor: Input tensor
            ranks: TT-ranks

        Returns:
            TTTensor object with factors
        """
        # Validate TT-ranks
        validated_ranks = tl.validate_tt_rank(tensor.shape, ranks)

        # Perform tensor-train decomposition — returns a TTTensor
        tt_tensor_obj = tl.decomposition.tensor_train(tensor, rank=validated_ranks)

        return tt_tensor_obj
    
    def _reconstruct_from_factors(self, factors: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct tensor from tensor-train factors
        
        Args:
            factors: List of tensor-train factors
            
        Returns:
            Reconstructed tensor
        """
        # Use tensorly's reconstruction function
        if hasattr(tl, 'tt_to_tensor'):
            return tl.tt_to_tensor(factors)
        else:
            # Fallback: manual reconstruction for compatibility
            return self._manual_tt_reconstruction(factors)
    
    def _manual_tt_reconstruction(self, factors: List[torch.Tensor]) -> torch.Tensor:
        """
        Manual tensor-train reconstruction for compatibility
        
        Args:
            factors: List of TT factors
            
        Returns:
            Reconstructed tensor
        """
        result = factors[0]
        
        for i in range(1, len(factors)):
            # Contract along shared dimensions
            result = torch.tensordot(result, factors[i], dims=([result.ndim-1], [0]))
        
        # Remove dummy dimensions (first and last rank dimensions)
        result = result.squeeze(0).squeeze(-1)
        
        return result
    
    def _estimate_factor_sizes(self, shape: tuple, ranks: List[int]) -> int:
        """
        Estimate total size of tensor-train factors
        
        Args:
            shape: Original tensor shape
            ranks: TT-ranks
            
        Returns:
            Estimated total factor size
        """
        if len(ranks) != len(shape) + 1:
            return 0
        
        total_size = 0
        for i, dim in enumerate(shape):
            factor_size = ranks[i] * dim * ranks[i + 1]
            total_size += factor_size
        
        return total_size
    
    def create_layer(self, compressed_data, original_shape):
        """
        Create FactorEmbedding or FactorLinear from tensor-train data
        
        Args:
            compressed_data: CompressedTensor with tensor-train factors
            original_shape: Original tensor shape [out_features, in_features] or [num_embeddings, embedding_dim]
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original shape
        """
        from ...framework.layers import FactorLayer, FactorEmbedding, FactorLinear, Factor
        
        # Create factors from tensor-train decomposition
        factors = []
        for factor_tensor in compressed_data.factors:
            factor = Factor(_weight=factor_tensor.clone())
            factors.append(factor)
        
        factor_layer = FactorLayer(
            factors=factors,
            func_name="tensor_train_contract"
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
