"""
SVD-LLM Closed-Form Update Plugin.

Refines U matrix after SVD truncation using least-squares optimization.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class ClosedFormUpdatePlugin(Plugin):
    """
    SVD-LLM-style closed-form U matrix refinement.

    After SVD truncation, refines U to minimize reconstruction error
    using calibration input/output pairs.

    Mathematical Background:
        Given truncated SVD: W_approx = U @ diag(S) @ Vt
        And calibration pairs (X, Y) where Y = X @ W^T

        We solve for optimal U:
            U_opt = argmin ||X @ Vt.T @ diag(S) @ U.T - Y||_F^2

        This is a linear least-squares problem with closed-form solution.

    Reads from:
        - calibration.activations.<layer_name>: Input X
        - calibration.outputs.<layer_name>: Original output Y
        - calibration.layer_names: List of layer names with collected data

    This plugin provides a `refine_svd` method that can be called by
    the main SVD compression plugin after truncation.

    Usage:
        plugin = ClosedFormUpdatePlugin()
        plugin.initialize(context)

        # After SVD truncation:
        U_refined, S, Vt = plugin.refine_svd(layer_name, U, S, Vt)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="SVD-LLM closed-form U refinement",
            category="compression"
        )

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        """
        Validate that calibration data is available.

        This plugin is typically called via refine_svd() method.
        do_execute validates that required calibration data exists.

        Returns:
            Dict with status and layers_with_outputs count
        """
        if not self.state_manager:
            raise RuntimeError("StateManager not available")

        if not self.state_manager.state.get("calibration.collected"):
            raise RuntimeError(
                "Calibration data not available. "
                "Ensure CalibrationCollectorPlugin runs first."
            )

        # Check that outputs were collected
        layer_names = self.state_manager.state.get("calibration.layer_names") or []
        layers_with_outputs = sum(
            1 for name in layer_names
            if self.state_manager.state.get(f"calibration.outputs.{name}") is not None
        )

        if layers_with_outputs == 0:
            self.logger.warning(
                "No calibration outputs found. "
                "Ensure CalibrationCollectorPlugin was run with collect_outputs=True"
            )

        return {
            "status": "ready",
            "layers_with_outputs": layers_with_outputs,
        }

    def refine_svd(
        self,
        layer_name: str,
        U: torch.Tensor,
        S: torch.Tensor,
        Vt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refine U matrix using closed-form least-squares update.

        Given truncated SVD factors and calibration data, solves:
            U_opt = argmin ||X @ Vt.T @ diag(S) @ U.T - Y||_F^2

        Args:
            layer_name: Name of the layer (for fetching calibration data)
            U: Left singular vectors (out_features x rank)
            S: Singular values (rank,)
            Vt: Right singular vectors transposed (rank x in_features)

        Returns:
            Tuple of (U_refined, S, Vt) - refined SVD factors
        """
        if not self.state_manager:
            self.logger.warning("StateManager not available, skipping refinement")
            return U, S, Vt

        X = self.state_manager.state.get(f"calibration.activations.{layer_name}")
        Y = self.state_manager.state.get(f"calibration.outputs.{layer_name}")

        if X is None or Y is None:
            self.logger.warning(
                f"Calibration data not found for {layer_name}, skipping refinement"
            )
            return U, S, Vt

        # Flatten batch dimensions: (batch * seq_len, hidden_dim)
        X = X.view(-1, X.shape[-1]).float()
        Y = Y.view(-1, Y.shape[-1]).float()

        # Move to same device as U
        device = U.device
        dtype = U.dtype
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            # Compute X @ Vt.T @ diag(S)
            # Note: Vt.T = V (in_features x rank)
            V = Vt.T.float()
            VS = V @ torch.diag(S.float())  # (in_features x rank)
            XVS = X @ VS                     # (batch x rank)

            try:
                # Solve least-squares: find U_opt such that XVS @ U_opt.T ≈ Y
                # Rearranging: (XVS)^T @ XVS @ U_opt.T = (XVS)^T @ Y
                # Using torch.linalg.lstsq: solve XVS @ A = Y for A = U_opt.T

                U_opt_T, residuals, rank_out, singular = torch.linalg.lstsq(XVS, Y)
                U_refined = U_opt_T.T  # (out_features x rank)

                # Ensure orthonormality via QR decomposition
                U_refined, R = torch.linalg.qr(U_refined)

                # Absorb R into S: new singular values
                # R is (rank x rank), diagonal contains scale factors
                S_refined = torch.abs(torch.diag(R)) * S.float()

                # Handle sign: ensure S is positive
                signs = torch.sign(torch.diag(R))
                U_refined = U_refined * signs.view(1, -1)

                return U_refined.to(dtype), S_refined.to(dtype), Vt

            except RuntimeError as e:
                self.logger.warning(f"Closed-form update failed for {layer_name}: {e}")
                return U, S, Vt

    def update_u_matrix(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vt: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update U matrix using closed-form least-squares optimization.

        This is a standalone utility method that can be called without
        StateManager - it takes calibration data (X, Y) directly.

        Args:
            U: Left singular vectors (out_features x rank)
            S: Singular values (rank,)
            Vt: Right singular vectors transposed (rank x in_features)
            X: Input activations (batch x in_features)
            Y: Original outputs (batch x out_features)

        Returns:
            Tuple of (U_refined, S_refined, Vt) - refined SVD factors
        """
        # Flatten batch dimensions if needed
        X = X.view(-1, X.shape[-1]).float()
        Y = Y.view(-1, Y.shape[-1]).float()

        # Move to same device as U
        device = U.device
        dtype = U.dtype
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            # Compute X @ Vt.T @ diag(S)
            V = Vt.T.float()
            VS = V @ torch.diag(S.float())  # (in_features x rank)
            XVS = X @ VS                     # (batch x rank)

            try:
                # Solve least-squares: find U_opt such that XVS @ U_opt.T ≈ Y
                U_opt_T, residuals, rank_out, singular = torch.linalg.lstsq(XVS, Y)
                U_refined = U_opt_T.T  # (out_features x rank)

                # Ensure orthonormality via QR decomposition
                U_refined, R = torch.linalg.qr(U_refined)

                # Absorb R into S: new singular values
                S_refined = torch.abs(torch.diag(R)) * S.float()

                # Handle sign: ensure S is positive
                signs = torch.sign(torch.diag(R))
                U_refined = U_refined * signs.view(1, -1)

                return U_refined.to(dtype), S_refined.to(dtype), Vt

            except RuntimeError:
                return U, S, Vt
