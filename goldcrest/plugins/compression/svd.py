"""
SVD (Singular Value Decomposition) Compression Plugin

This plugin implements truncated SVD matrix decomposition for neural network compression.
SVD decomposes a matrix A into A = U * S * V^T where U and V are orthogonal matrices
and S is a diagonal matrix of singular values. Truncation keeps only the top-k singular values.
"""

import torch
from typing import Dict, Any, Optional, Union, List
import uuid
from ...framework.context import PipelineContext
from .base import LowRankCompressionPlugin
from .svd_backend import build_svd_backend


class CompressedSVDTensor:
    """Container for SVD-compressed tensor data"""
    
    def __init__(self, u: torch.Tensor, s: torch.Tensor, vt: torch.Tensor, 
                 original_shape: torch.Size, compression_ratio: float):
        self.u = u  # Left singular vectors
        self.s = s  # Singular values
        self.vt = vt  # Right singular vectors (transposed)
        self.original_shape = original_shape
        self.compression_ratio = compression_ratio
        
    def size(self) -> int:
        """Total number of parameters in compressed representation"""
        return self.u.numel() + self.s.numel() + self.vt.numel()


class SVD(LowRankCompressionPlugin):
    """
    SVD compression plugin for matrix factorization using truncated SVD.

    Parameters:
        rank: Number of singular values to keep (truncation rank)
        preserve_energy: Alternative to rank - keep components that preserve this fraction of energy
        use_activation_scaling: Enable ASVD activation-aware scaling (requires ActivationScalingPlugin)
        use_data_whitening: Enable SVD-LLM data whitening (requires DataWhiteningPlugin)
        use_closed_form_update: Enable SVD-LLM closed-form U update (requires ClosedFormUpdatePlugin)
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        preserve_energy: Optional[float] = None,
        svd_backend: str = "torch",
        svd_backend_config: Optional[Dict[str, Any]] = None,
        cola: Optional[Dict[str, Any]] = None,
        use_activation_scaling: bool = False,
        use_data_whitening: bool = False,
        use_closed_form_update: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.plugin_name = "SVD"
        self.plugin_id = str(uuid.uuid4())
        self.category = "compression"

        if rank is not None and preserve_energy is not None:
            raise ValueError("Cannot specify both rank and preserve_energy")
        if rank is None and preserve_energy is None:
            raise ValueError("Must specify either rank or preserve_energy")

        self.rank = rank
        self.preserve_energy = preserve_energy
        backend_config = dict(svd_backend_config or {})
        if cola and not backend_config:
            backend_config = dict(cola)
        self.svd_backend_name = svd_backend
        self.svd_backend_config = backend_config
        self.svd_backend = build_svd_backend(self.svd_backend_name, self.svd_backend_config)
        self.svd_which = backend_config.get("which", "LM")

        # ASVD and SVD-LLM enhancement flags
        self.use_activation_scaling = use_activation_scaling
        self.use_data_whitening = use_data_whitening
        self.use_closed_form_update = use_closed_form_update
        
    def initialize(self, context: PipelineContext):
        """Initialize the SVD plugin"""
        super().initialize(context)
        self.context.event_bus.emit(
            "plugin.loaded",
            {
                "plugin_name": self.plugin_name,
                "plugin_id": self.plugin_id,
                "category": self.category
            }
        )
    
    def do_execute(self, context, **params):
        """Plugin interface implementation for SVD compression"""
        tensor = params.get('tensor', None)
        if tensor is None:
            raise ValueError("SVD compression requires a 'tensor' parameter")
        return self.compress(tensor)
    
    def _determine_truncation_rank(self, singular_values: torch.Tensor) -> int:
        """Determine the truncation rank based on configuration"""
        if self.rank is not None:
            return min(self.rank, len(singular_values))
        
        if self.preserve_energy is not None:
            # Calculate cumulative energy (normalized squared singular values)
            energy = singular_values ** 2
            total_energy = energy.sum()
            cumulative_energy = torch.cumsum(energy, dim=0) / total_energy
            
            # Find minimum rank that preserves desired energy
            preserved_indices = (cumulative_energy >= self.preserve_energy).nonzero(as_tuple=True)[0]
            if len(preserved_indices) > 0:
                return preserved_indices[0].item() + 1
            else:
                return len(singular_values)  # Fallback to full rank
        
        return len(singular_values)  # Should not reach here
    
    def compress(self, tensor: torch.Tensor, layer_name: str = None) -> CompressedSVDTensor:
        """
        Compress tensor using truncated SVD with optional ASVD/SVD-LLM transforms.

        Args:
            tensor: Input tensor to compress
            layer_name: Optional layer name for fetching pre-computed transforms

        Returns:
            CompressedSVDTensor: SVD-compressed representation
        """
        original_shape = tensor.shape

        # SVD works on 2D matrices, so reshape if necessary
        if len(tensor.shape) > 2:
            # Flatten all dimensions except the last one
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor

        weight = tensor_2d.clone()
        scaling_factor = None
        whitening_L = None
        whitening_L_inv = None

        # Apply activation scaling if available and enabled
        if self.use_activation_scaling and self.state_manager and layer_name:
            scaling_factor = self.state_manager.state.get(f"svd.scaling.{layer_name}")
            if scaling_factor is not None:
                from .svd_activation_scaling import ActivationScalingPlugin
                weight = ActivationScalingPlugin.apply_scaling(weight, scaling_factor.to(weight.device))

        # Apply data whitening if available and enabled
        if self.use_data_whitening and self.state_manager and layer_name:
            whitening_L = self.state_manager.state.get(f"svd.whitening.L.{layer_name}")
            whitening_L_inv = self.state_manager.state.get(f"svd.whitening.L_inv.{layer_name}")
            if whitening_L is not None:
                from .svd_data_whitening import DataWhiteningPlugin
                weight = DataWhiteningPlugin.apply_whitening(
                    weight,
                    whitening_L.to(device=weight.device, dtype=weight.dtype),
                )

        # Get rank from StateManager if available (from BinarySearchRankPlugin)
        rank = self.rank
        if self.state_manager and layer_name:
            stored_rank = self.state_manager.state.get(f"svd.ranks.{layer_name}")
            if stored_rank is not None:
                rank = stored_rank

        svd_k = rank if rank is not None and self.preserve_energy is None else None
        which = "LM" if self.preserve_energy is not None else self.svd_which

        # Perform SVD via selected backend
        U, S, Vt = self.svd_backend.compute_svd(
            weight,
            k=svd_k,
            which=which,
            full=False,
        )

        # Determine truncation rank
        k = self._determine_truncation_rank(S) if rank is None else min(rank, len(S))

        # Truncate to top k components
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        # Apply inverse transforms to V in reverse order

        # Apply inverse whitening first (if whitening was applied)
        if whitening_L_inv is not None:
            from .svd_data_whitening import DataWhiteningPlugin
            # Vt_k is (k x n), need to transform back
            Vt_k = DataWhiteningPlugin.inverse_whitening(
                Vt_k.T,
                whitening_L_inv.to(device=Vt_k.device, dtype=Vt_k.dtype),
            ).T

        # Apply inverse scaling (if scaling was applied)
        if scaling_factor is not None:
            from .svd_activation_scaling import ActivationScalingPlugin
            Vt_k = ActivationScalingPlugin.inverse_scaling(Vt_k.T, scaling_factor.to(Vt_k.device)).T

        # Optional closed-form U refinement
        if self.use_closed_form_update and self.strategy_factory and layer_name:
            try:
                cfu_plugin = self.strategy_factory.get_compression_strategy("closed_form_update")
                if hasattr(cfu_plugin, 'initialize') and self.context:
                    cfu_plugin.initialize(self.context)
                U_k, S_k, Vt_k = cfu_plugin.refine_svd(layer_name, U_k, S_k, Vt_k)
            except Exception as e:
                # Log warning but continue without refinement
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Closed-form update failed: {e}")

        # Calculate compression ratio
        original_size = tensor.numel()
        compressed_size = U_k.numel() + S_k.numel() + Vt_k.numel()
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

        return CompressedSVDTensor(
            u=U_k, s=S_k, vt=Vt_k,
            original_shape=original_shape,
            compression_ratio=compression_ratio
        )
    
    def decompress(self, compressed: CompressedSVDTensor) -> torch.Tensor:
        """
        Reconstruct tensor from SVD compression
        
        Args:
            compressed: SVD-compressed tensor data
            
        Returns:
            torch.Tensor: Reconstructed tensor
        """
        # Reconstruct matrix: A_approx = U_k * S_k * Vt_k
        reconstructed_2d = compressed.u @ torch.diag(compressed.s) @ compressed.vt
        
        # Reshape back to original shape if necessary
        if len(compressed.original_shape) > 2:
            return reconstructed_2d.view(compressed.original_shape)
        else:
            return reconstructed_2d
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get compression configuration information"""
        return {
            "method": "svd",
            "rank": self.rank,
            "preserve_energy": self.preserve_energy,
            "svd_backend": self.svd_backend_name,
            "svd_backend_config": self.svd_backend_config,
            "plugin_id": self.plugin_id,
            "use_activation_scaling": self.use_activation_scaling,
            "use_data_whitening": self.use_data_whitening,
            "use_closed_form_update": self.use_closed_form_update,
        }
    
    def supports_tensor_shape(self, shape: torch.Size) -> bool:
        """Check if tensor shape is supported for SVD compression"""
        # SVD requires at least 2D tensors (can reshape higher dims)
        return len(shape) >= 2 and min(shape) > 1
    
    def create_layer(self, compressed_data, original_shape):
        """
        Create FactorEmbedding or FactorLinear from SVD data
        
        Args:
            compressed_data: CompressedSVDTensor with SVD factors (U, S, V)
            original_shape: Original tensor shape [out_features, in_features] or [num_embeddings, embedding_dim]
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original shape
        """
        from ...framework.layers import FactorLayer, FactorEmbedding, FactorLinear, Factor
        
        # Create factors from SVD decomposition (U, S, V)
        factors = [
            Factor(_weight=compressed_data.u.clone()),
            Factor(_weight=torch.diag(compressed_data.s).clone()),
            Factor(_weight=compressed_data.vt.clone())
        ]
        factor_layer = FactorLayer(_factors=factors)
        factor_layer.func_name = 'svd'
        
        # Determine layer type based on original shape
        if len(original_shape) == 2:
            if original_shape[0] > original_shape[1]:  # Likely embedding: [vocab_size, embedding_dim]
                # For embedding-like shape (vocab_size, emb_dim) it's unusual for SVD
                # but return a FactorLinear with matching dims for consistency
                return FactorLinear(
                    in_features=original_shape[1],
                    out_features=original_shape[0],
                    _weight=factor_layer,
                    bias=False,
                    _func_name='svd'
                )
            else:  # Linear layer: [out_features, in_features]
                return FactorLinear(
                    in_features=original_shape[1],
                    out_features=original_shape[0],
                    _weight=factor_layer,
                    bias=True,
                    _func_name='svd'
                )
        else:
            raise ValueError(f"Unsupported tensor shape for layer creation: {original_shape}")
