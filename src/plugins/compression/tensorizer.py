"""
Tensorizer Plugin for Toggle Architecture

This plugin implements tensorization functionality, migrated from the original 
Tensorizer class in factorization.py.
"""
import itertools
import torch
from typing import List, Dict, Any, Optional, Union, Tuple, Set

from .base import LowRankCompressionPlugin, CompressedTensor
from .tensorly_backend import set_tensorly_backend


class Tensorizer(LowRankCompressionPlugin):
    """
    Tensorizer plugin for reshaping tensors into higher-order formats
    
    Migrates Tensorizer class functionality from factorization.py to the
    new plugin-based architecture. This plugin focuses on tensor reshaping
    and tensor size analysis utilities.
    """
    
    def __init__(self, device: str = 'cpu', backend: str = 'pytorch', **kwargs):
        """
        Initialize Tensorizer plugin
        
        Args:
            device: Computation device ('cpu', 'cuda', etc.)
            backend: Tensorly backend ('pytorch', 'numpy')
            **kwargs: Additional configuration
        """
        super().__init__(name="Tensorizer", **kwargs)
        self.device = device
        self.backend = backend
        
        # Set tensorly backend
        set_tensorly_backend(backend)
    
    def compress(self, tensor: torch.Tensor, tensor_size: Union[List, Tuple], **params) -> CompressedTensor:
        """
        "Compress" tensor by reshaping it to specified tensor_size
        
        Note: This is not traditional compression but tensor restructuring
        
        Args:
            tensor: Input tensor to tensorize
            tensor_size: Target tensor shape
            **params: Additional parameters (ignored)
            
        Returns:
            CompressedTensor with reshaped tensor
        """
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        
        # Perform tensorization (reshaping)
        tensorized = self.tensorize(tensor_size, tensor)
        
        # Calculate "compression ratio" (actually just shape difference)
        original_size = tensor.numel()
        tensorized_size = tensorized.numel()
        compression_ratio = 1.0  # No actual compression, just reshaping
        
        return CompressedTensor(
            factors=tensorized,  # Single reshaped tensor
            method="tensorizer",
            original_shape=tensor.shape,
            compression_ratio=compression_ratio,
            metadata={
                "target_tensor_size": tensor_size,
                "device": self.device,
                "backend": self.backend,
                "original_size": original_size,
                "tensorized_size": tensorized_size,
                "tensorized_shape": tensorized.shape
            }
        )
    
    def decompress(self, compressed: CompressedTensor) -> torch.Tensor:
        """
        Decompress (reshape back) tensorized tensor
        
        Args:
            compressed: CompressedTensor with tensorized data
            
        Returns:
            Tensor reshaped back to original form
        """
        if compressed.method != "tensorizer":
            raise ValueError(f"Cannot decompress tensor compressed with method '{compressed.method}'")
        
        # Reshape back to original shape
        return compressed.factors.view(compressed.original_shape)
    
    
    def validate_tensor_compatibility(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor can be tensorized
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True (tensorizer works with any tensor)
        """
        return tensor.numel() > 0
    
    @staticmethod
    def tensorize(tensor_size: Optional[Union[Tuple, List]], arr: torch.Tensor) -> torch.Tensor:
        """
        Tensorize (reshape) tensor to specified size
        
        Migrated from original Tensorizer.tensorize static method
        
        Args:
            tensor_size: Target tensor shape
            arr: Input tensor
            
        Returns:
            Reshaped tensor
        """
        if tensor_size is None:
            return arr
        
        return arr.reshape(tensor_size)
    
    @staticmethod 
    def list_sizes(length: int = 768, min_length: int = 3, max_length: int = 4) -> List[Tuple]:
        """
        List possible tensor size factorizations for a given length
        
        Migrated from original Tensorizer.list_sizes static method
        
        Args:
            length: Total number of elements
            min_length: Minimum number of factors
            max_length: Maximum number of factors (-1 for no limit)
            
        Returns:
            List of possible factorizations as tuples
        """
        def find_factors(n: int) -> List[int]:
            """Find factors of n"""
            factors = set()
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    factors.add(i)
                    factors.add(n // i)
            # Remove 1 and the number itself to avoid trivial factorizations
            factors = [element for element in factors if element not in [1, length]]
            return sorted(factors)
        
        def find_combinations_with_repetition(factors: List[int], target: int, 
                                            min_length: int = min_length, 
                                            max_length: int = max_length) -> Set[Tuple]:
            """Find combinations of factors that multiply to target"""
            result = set()
            
            if max_length == -1:
                max_length = len(factors) + 1
            
            for r in range(min_length, max_length + 1):
                for combo in itertools.product(factors, repeat=r):
                    if evaluate_combo(combo, target):
                        result.add(tuple(sorted(combo)))  # Sort to avoid duplicates
            return result
        
        def evaluate_combo(combo: Tuple, gt: int) -> bool:
            """Check if combination multiplies to ground truth"""
            prod = 1
            for number in combo:
                prod *= number
                if prod > gt:
                    return False
            return prod == gt
        
        # Find factors and combinations
        factors = find_factors(n=length)
        combinations_of_factors = sorted(find_combinations_with_repetition(factors, target=length))
        
        return combinations_of_factors
    
    def get_factorizations(self, length: int, min_length: int = 3, max_length: int = 4) -> List[Tuple]:
        """
        Get possible tensor factorizations for given length
        
        Args:
            length: Total number of elements  
            min_length: Minimum number of factors
            max_length: Maximum number of factors
            
        Returns:
            List of possible factorizations
        """
        return self.list_sizes(length=length, min_length=min_length, max_length=max_length)
    
    def find_optimal_tensorization(self, tensor: torch.Tensor, 
                                 target_factors: Optional[int] = None,
                                 prefer_square: bool = True) -> Tuple[int, ...]:
        """
        Find optimal tensorization for a given tensor
        
        Args:
            tensor: Input tensor
            target_factors: Target number of factors (None for auto)
            prefer_square: Prefer square-like factorizations
            
        Returns:
            Optimal tensor size factorization
        """
        total_elements = tensor.numel()
        
        # Get possible factorizations
        factorizations = self.get_factorizations(
            length=total_elements,
            min_length=target_factors or 2,
            max_length=target_factors or 4
        )
        
        if not factorizations:
            # Fallback: use original shape if no good factorizations found
            return tensor.shape
        
        if prefer_square:
            # Score factorizations by how "square" they are (factors close to each other)
            def square_score(factors: Tuple) -> float:
                if not factors:
                    return float('inf')
                mean_factor = sum(factors) / len(factors)
                return sum((f - mean_factor) ** 2 for f in factors) / len(factors)
            
            optimal = min(factorizations, key=square_score)
        else:
            # Just use the first valid factorization
            optimal = factorizations[0]
        
        return optimal
    
    def analyze_tensorization_options(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze tensorization options for a given tensor
        
        Args:
            tensor: Input tensor to analyze
            
        Returns:
            Dictionary with tensorization analysis
        """
        total_elements = tensor.numel()
        original_shape = tensor.shape
        
        # Get all possible factorizations
        factorizations = self.get_factorizations(length=total_elements)
        
        analysis = {
            "original_shape": original_shape,
            "total_elements": total_elements,
            "num_factorizations": len(factorizations),
            "factorizations": factorizations[:10],  # Limit to first 10 for readability
            "recommended": None
        }
        
        if factorizations:
            # Find recommended tensorization
            recommended = self.find_optimal_tensorization(tensor)
            analysis["recommended"] = recommended
        
        return analysis
    
    def do_execute(self, context, **params):
        """
        Execute tensorization as a plugin
        
        Args:
            context: Pipeline context (not used for tensorizer)
            **params: Parameters including:
                - tensor: Tensor to tensorize
                - tensor_size: Target tensor shape
                
        Returns:
            Tensorization result
        """
        tensor = params.get('tensor')
        tensor_size = params.get('tensor_size')
        
        if tensor is None:
            raise ValueError("Tensorizer requires 'tensor' parameter")
        
        if tensor_size is None:
            # Auto-determine optimal tensorization
            tensor_size = self.find_optimal_tensorization(tensor)
        
        # Perform compression (tensorization)
        compressed = self.compress(tensor, tensor_size)
        
        return {
            'compressed_tensor': compressed,
            'original_shape': tensor.shape,
            'tensorized_shape': tensor_size,
            'analysis': self.analyze_tensorization_options(tensor)
        }
