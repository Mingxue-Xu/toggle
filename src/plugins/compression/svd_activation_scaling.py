"""
ASVD Activation Scaling Plugin.

Computes per-layer activation scaling factors for ASVD-style compression.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class ActivationScalingPlugin(Plugin):
    """
    ASVD-style activation-aware scaling factor computation.

    Computes per-layer scaling factors from calibration data and stores
    them in StateManager for use by SVD compression plugins.

    Mathematical Background:
        Standard SVD minimizes: ||W - W_approx||_F^2
        ASVD minimizes:         ||S(W - W_approx)||_F^2

        where S = diag(activation_scaling)^alpha

    Reads from:
        - calibration.activations.<layer_name> (from CalibrationCollectorPlugin)
        - calibration.fisher.<layer_name> (optional, from FisherInformationPlugin)
        - calibration.layer_names: List of layer names with collected data

    Writes to:
        - svd.scaling.<layer_name>: Scaling vector for each layer

    Usage:
        plugin = ActivationScalingPlugin(method="abs_mean", alpha=0.5)
        plugin.initialize(context)
        result = plugin.execute()
    """

    def __init__(
        self,
        method: str = "abs_mean",
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize ActivationScalingPlugin.

        Args:
            method: Scaling method - "abs_mean", "abs_max", or "fisher"
            alpha: Scaling exponent (0.0-1.0). Higher values = more aggressive scaling
        """
        super().__init__(**kwargs)
        self.method = method
        self.alpha = alpha

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="ASVD activation scaling computation",
            category="compression"
        )

    def do_execute(
        self,
        target_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute activation scaling factors from calibration data.

        Args:
            target_layers: Optional list of layer name patterns to process

        Returns:
            Dict with method, alpha, and layers_processed count
        """
        # Validate calibration data exists
        if not self.state_manager:
            raise RuntimeError("StateManager not available")

        if not self.state_manager.state.get("calibration.collected"):
            raise RuntimeError(
                "Calibration data not available. "
                "Ensure CalibrationCollectorPlugin runs before ActivationScalingPlugin."
            )

        self.emit_event("activation_scaling.started", {
            "method": self.method,
            "alpha": self.alpha
        })

        scaling_factors: Dict[str, torch.Tensor] = {}
        layers_processed = 0

        # Get list of layers with calibration data
        layer_names = self.state_manager.state.get("calibration.layer_names") or []

        total_layers = len(layer_names)
        for idx, layer_name in enumerate(layer_names):
            if target_layers and not any(t in layer_name for t in target_layers):
                continue

            activations = self.state_manager.state.get(f"calibration.activations.{layer_name}")
            if activations is None:
                continue

            # Flatten to (N, hidden_dim) if needed
            if len(activations.shape) > 2:
                activations = activations.view(-1, activations.shape[-1])

            # Compute scaling based on method
            if self.method == "abs_mean":
                scale = activations.abs().mean(dim=0)
            elif self.method == "abs_max":
                scale = activations.abs().max(dim=0).values
            elif self.method == "fisher":
                # Fisher info should be pre-computed by FisherInformationPlugin
                fisher = self.state_manager.state.get(f"calibration.fisher.{layer_name}")
                if fisher is None:
                    self.logger.warning(
                        f"Fisher info not found for {layer_name}, falling back to abs_mean"
                    )
                    scale = activations.abs().mean(dim=0)
                else:
                    scale = fisher.sqrt()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")

            # Apply alpha exponent and clamp for numerical stability
            scale = scale.float().clamp(min=1e-6) ** self.alpha

            # Store in StateManager
            self.state_manager.state.set(f"svd.scaling.{layer_name}", scale)
            scaling_factors[layer_name] = scale
            layers_processed += 1

            self.update_progress(100.0 * (idx + 1) / max(1, total_layers))

        self.emit_event("activation_scaling.completed", {"layers": layers_processed})

        return {
            "method": self.method,
            "alpha": self.alpha,
            "layers_processed": layers_processed,
        }

    def compute_scaling_factors(
        self,
        activations: Dict[str, torch.Tensor],
        method: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scaling factors from activation tensors.

        This is a standalone utility method that can be called without
        initializing the full plugin context.

        Args:
            activations: Dict mapping layer names to activation tensors
            method: Scaling method - "abs_mean", "abs_max", or "fisher"
                   (defaults to self.method)
            alpha: Scaling exponent (defaults to self.alpha)

        Returns:
            Dict mapping layer names to scaling factor tensors
        """
        method = method if method is not None else self.method
        alpha = alpha if alpha is not None else self.alpha

        scaling_factors: Dict[str, torch.Tensor] = {}

        for layer_name, acts in activations.items():
            # Flatten to (N, hidden_dim) if needed
            if len(acts.shape) > 2:
                acts = acts.view(-1, acts.shape[-1])

            # Compute scaling based on method
            if method == "abs_mean":
                scale = acts.abs().mean(dim=0)
            elif method == "abs_max":
                scale = acts.abs().max(dim=0).values
            else:
                # Default to abs_mean for unknown methods
                scale = acts.abs().mean(dim=0)

            # Apply alpha exponent and clamp for numerical stability
            scale = scale.float().clamp(min=1e-6) ** alpha
            scaling_factors[layer_name] = scale

        return scaling_factors

    @staticmethod
    def apply_scaling(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling to weight matrix before SVD.

        Transforms: W_scaled = W * S (element-wise column scaling)

        Args:
            weight: Weight matrix (out_features x in_features)
            scale: Scaling vector (in_features,)

        Returns:
            Scaled weight matrix
        """
        return weight * scale.view(1, -1)

    @staticmethod
    def inverse_scaling(V: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse scaling to V matrix after SVD.

        After SVD of scaled weights, the V matrix needs to be unscaled:
        V_unscaled = V / S

        Args:
            V: Right singular vectors (in_features x rank)
            scale: Scaling vector (in_features,)

        Returns:
            Unscaled V matrix
        """
        return V / scale.view(-1, 1)
