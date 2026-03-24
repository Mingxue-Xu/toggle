"""
SVD-LLM Data Whitening Plugin.

Computes Cholesky-based whitening matrices for truncation-aware SVD.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class DataWhiteningPlugin(Plugin):
    """
    SVD-LLM-style truncation-aware data whitening.

    Computes Cholesky-based whitening matrices from X^T X calibration data
    and stores them in StateManager for use by SVD compression plugins.

    Mathematical Background:
        Given calibration inputs X, we compute:
            X^T X = L @ L^T  (Cholesky decomposition)

        The whitening transform is:
            W_white = W @ L

        After SVD and truncation, the inverse is:
            V_original = L^{-1} @ V_truncated

    Reads from:
        - calibration.xtx.<layer_name> (from CalibrationCollectorPlugin)
        - calibration.layer_names: List of layer names with collected data

    Writes to:
        - svd.whitening.L.<layer_name>: Cholesky factor L
        - svd.whitening.L_inv.<layer_name>: Inverse of L for reconstruction

    Usage:
        plugin = DataWhiteningPlugin(regularization=1e-6)
        plugin.initialize(context)
        result = plugin.execute()
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        **kwargs
    ):
        """
        Initialize DataWhiteningPlugin.

        Args:
            regularization: Diagonal regularization for numerical stability.
                           Added to diagonal of X^T X before Cholesky.
        """
        super().__init__(**kwargs)
        self.regularization = regularization

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="SVD-LLM data whitening computation",
            category="compression"
        )

    def do_execute(
        self,
        target_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute whitening matrices from calibration data.

        Args:
            target_layers: Optional list of layer name patterns to process

        Returns:
            Dict with layers_processed and failed_layers counts
        """
        if not self.state_manager:
            raise RuntimeError("StateManager not available")

        if not self.state_manager.state.get("calibration.collected"):
            raise RuntimeError(
                "Calibration data not available. "
                "Ensure CalibrationCollectorPlugin runs before DataWhiteningPlugin."
            )

        self.emit_event("data_whitening.started", {
            "regularization": self.regularization
        })

        layers_processed = 0
        failed_layers = []

        # Get list of layers with calibration data
        layer_names = self.state_manager.state.get("calibration.layer_names") or []
        total_layers = len(layer_names)

        for idx, layer_name in enumerate(layer_names):
            if target_layers and not any(t in layer_name for t in target_layers):
                continue

            xtx = self.state_manager.state.get(f"calibration.xtx.{layer_name}")
            if xtx is None:
                continue

            try:
                # Compute the factorization in float64. Some under-conditioned
                # X^T X matrices still fail after the eigenvalue shift in float32.
                target_dtype = xtx.dtype if xtx.is_floating_point() else torch.float32
                xtx_work = xtx.to(dtype=torch.float64)
                n = xtx_work.shape[0]
                eye = torch.eye(n, dtype=torch.float64, device=xtx_work.device)
                xtx_reg = 0.5 * (xtx_work + xtx_work.T)
                xtx_reg = xtx_reg + self.regularization * eye

                # Compute Cholesky decomposition: X^T X = L @ L^T
                try:
                    L = torch.linalg.cholesky(xtx_reg)
                except RuntimeError:
                    min_eig = torch.min(torch.linalg.eigvalsh(xtx_reg)).item()
                    shift = max(0.0, -min_eig) + self.regularization
                    L = torch.linalg.cholesky(xtx_reg + shift * eye)

                # Compute inverse for reconstruction, then cast back to the
                # original floating dtype to avoid inflating persisted state.
                L_inv = torch.linalg.inv(L)
                L = L.to(dtype=target_dtype)
                L_inv = L_inv.to(dtype=target_dtype)

                # Store in StateManager
                self.state_manager.state.set(f"svd.whitening.L.{layer_name}", L)
                self.state_manager.state.set(f"svd.whitening.L_inv.{layer_name}", L_inv)
                layers_processed += 1

            except RuntimeError as e:
                self.logger.warning(f"Cholesky failed for {layer_name}: {e}")
                failed_layers.append(layer_name)

            self.update_progress(100.0 * (idx + 1) / max(1, total_layers))

        self.emit_event("data_whitening.completed", {
            "layers": layers_processed,
            "failed": len(failed_layers),
        })

        return {
            "layers_processed": layers_processed,
            "failed_layers": failed_layers,
        }

    @staticmethod
    def apply_whitening(weight: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform to weight matrix before SVD.

        Transforms: W_white = W @ L

        Args:
            weight: Weight matrix (out_features x in_features)
            L: Cholesky factor (in_features x in_features)

        Returns:
            Whitened weight matrix
        """
        return weight @ L

    @staticmethod
    def inverse_whitening(V: torch.Tensor, L_inv: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse whitening to V matrix after SVD.

        After SVD of whitened weights, the V matrix needs inverse transform:
        V_original = L^{-1} @ V

        Args:
            V: Right singular vectors (in_features x rank)
            L_inv: Inverse Cholesky factor (in_features x in_features)

        Returns:
            Original-space V matrix
        """
        return L_inv @ V
