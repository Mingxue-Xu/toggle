"""
ASVD PPL-based Sensitivity Analysis Plugin.

Computes per-layer sensitivity via perplexity evaluation at various compression ratios.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class PPLSensitivityPlugin(Plugin):
    """
    ASVD-style perplexity-based layer sensitivity analysis.

    Computes sensitivity by measuring PPL impact when compressing
    each layer at various ratios. Results stored in StateManager
    for use by rank allocation plugins.

    Mathematical Background:
        Layer sensitivity is measured as:
            delta_PPL_l(k) = PPL(M_l,k) - PPL(M)

        where M_l,k is the model with layer l compressed to rank k.

    Writes to:
        - svd.sensitivity.ppl.<layer_name>: Dict of ratio -> ppl_delta
        - svd.sensitivity.order: Layers sorted by sensitivity (most to least)
        - svd.sensitivity.baseline_ppl: Baseline perplexity before compression

    Usage:
        plugin = PPLSensitivityPlugin(param_ratios=[0.3, 0.5, 0.7, 0.9])
        plugin.initialize(context)
        result = plugin.execute(model=model, eval_dataloader=dataloader)
    """

    def __init__(
        self,
        param_ratios: Optional[List[float]] = None,
        cache_results: bool = True,
        use_activation_scaling: bool = False,
        svd_backend: str = "torch",
        svd_backend_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PPLSensitivityPlugin.

        Args:
            param_ratios: List of compression ratios to evaluate (0.0-1.0)
            cache_results: Whether to cache results in StateManager
            use_activation_scaling: Whether to use the same activation-aware
                SVD transform as final compression during temporary probes
            svd_backend: SVD backend used for temporary probes
            svd_backend_config: Optional backend configuration for probes
        """
        super().__init__(**kwargs)
        self.param_ratios = param_ratios or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.cache_results = cache_results
        self.use_activation_scaling = use_activation_scaling
        self.svd_backend = svd_backend
        self.svd_backend_config = dict(svd_backend_config or {})

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="ASVD PPL-based sensitivity analysis",
            category="analysis"
        )

    def do_execute(
        self,
        model: nn.Module = None,
        tokenizer: Any = None,
        eval_dataloader: Any = None,
        target_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute per-layer sensitivity via perplexity evaluation.

        Args:
            model: The model to analyze
            tokenizer: Tokenizer for the model
            eval_dataloader: DataLoader for perplexity evaluation
            target_layers: Optional list of layer name patterns to analyze

        Returns:
            Dict with baseline_ppl, sensitivity_map, and sensitivity_order
        """
        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("PPLSensitivityPlugin requires a model")
        if eval_dataloader is None:
            raise ValueError("PPLSensitivityPlugin requires an eval_dataloader")

        self.emit_event(
            "ppl_sensitivity.started",
            {
                "ratios": self.param_ratios,
                "use_activation_scaling": self.use_activation_scaling,
            },
        )

        # Get baseline PPL
        baseline_ppl = self._compute_ppl(model, tokenizer, eval_dataloader)

        sensitivity_map: Dict[str, Dict[float, float]] = {}

        # Find target linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers and not any(t in name for t in target_layers):
                    continue
                linear_layers.append((name, module))

        total_layers = len(linear_layers)

        for idx, (name, module) in enumerate(linear_layers):
            self.update_progress(100.0 * idx / max(1, total_layers))

            original_weight = module.weight.data.clone()
            m, n = original_weight.shape

            layer_sensitivity: Dict[float, float] = {}

            for ratio in self.param_ratios:
                # Compute rank for this ratio
                k = max(1, int(ratio * min(m, n)))

                # Apply SVD compression temporarily
                try:
                    compressed = self._reconstruct_probe_weight(
                        original_weight,
                        rank=k,
                        layer_name=name,
                    )

                    module.weight.data = compressed.to(original_weight.dtype)

                    # Measure PPL
                    compressed_ppl = self._compute_ppl(model, tokenizer, eval_dataloader)
                    ppl_delta = compressed_ppl - baseline_ppl

                    layer_sensitivity[ratio] = ppl_delta

                    # Restore original weight
                    module.weight.data = original_weight

                except Exception as e:
                    self.logger.warning(f"SVD failed for {name} at ratio {ratio}: {e}")
                    module.weight.data = original_weight
                    layer_sensitivity[ratio] = float('inf')

            sensitivity_map[name] = layer_sensitivity

            # Store in StateManager
            if self.state_manager and self.cache_results:
                self.state_manager.state.set(f"svd.sensitivity.ppl.{name}", layer_sensitivity)

        # Compute sensitivity ordering (by max PPL delta)
        layer_max_sensitivity = {
            name: max(sens.values()) for name, sens in sensitivity_map.items()
        }
        sorted_layers = sorted(
            layer_max_sensitivity.keys(),
            key=lambda x: layer_max_sensitivity[x],
            reverse=True
        )

        if self.state_manager:
            self.state_manager.state.set("svd.sensitivity.order", sorted_layers)
            self.state_manager.state.set("svd.sensitivity.baseline_ppl", baseline_ppl)
            self.state_manager.state.set(
                "svd.sensitivity.mode",
                "act_aware_ppl" if self.use_activation_scaling else "plain_ppl",
            )

        self.emit_event("ppl_sensitivity.completed", {"layers": total_layers})

        return {
            "baseline_ppl": baseline_ppl,
            "sensitivity_map": sensitivity_map,
            "sensitivity_order": sorted_layers,
            "use_activation_scaling": self.use_activation_scaling,
        }

    def _reconstruct_probe_weight(
        self,
        weight: torch.Tensor,
        rank: int,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Reconstruct a dense probe weight using the same SVD path as final compression.

        Using the production SVD strategy here keeps temporary sensitivity probes
        aligned with the final decomposition logic, including activation-aware scaling.
        """
        from .svd import SVD

        probe_svd = SVD(
            rank=rank,
            svd_backend=self.svd_backend,
            svd_backend_config=self.svd_backend_config,
            use_activation_scaling=self.use_activation_scaling,
            name=f"{self.name}_probe_svd",
        )
        if self.context is not None:
            probe_svd.initialize(self.context)

        rank_key = f"svd.ranks.{layer_name}" if layer_name else None
        stored_rank = None
        had_stored_rank = False
        if rank_key and self.state_manager is not None:
            stored_rank = self.state_manager.state.get(rank_key)
            had_stored_rank = stored_rank is not None
            if had_stored_rank:
                self.state_manager.state.delete(rank_key)

        try:
            compressed = probe_svd.compress(weight, layer_name=layer_name)
        finally:
            if rank_key and self.state_manager is not None and had_stored_rank:
                self.state_manager.state.set(rank_key, stored_rank)

        return probe_svd.decompress(compressed)

    def _compute_ppl(self, model: nn.Module, tokenizer: Any, dataloader: Any) -> float:
        """
        Compute perplexity on evaluation data.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer (unused but kept for interface consistency)
            dataloader: DataLoader providing evaluation batches

        Returns:
            Perplexity value
        """
        # Use the shared evaluation strategy only when no explicit dataloader is provided.
        if dataloader is None and self.strategy_factory:
            try:
                eval_strategy = self.strategy_factory.get_evaluation_strategy(["perplexity"])
                result = eval_strategy.execute(self.context, model=model, tokenizer=tokenizer)
                return result.get("perplexity", {}).get("ppl", float("inf"))
            except Exception:
                pass  # Fall back to manual computation

        if dataloader is None:
            return float("inf")

        # Fallback: simple cross-entropy loss computation
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                try:
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                        num_tokens = batch.get("input_ids", batch.get("labels")).numel()
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                        outputs = model(inputs)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        num_tokens = inputs.numel()
                    else:
                        inputs = batch.to(device)
                        outputs = model(inputs)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        num_tokens = inputs.numel()

                    if hasattr(loss, 'item'):
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens

                except Exception as e:
                    self.logger.warning(f"Error computing loss for batch: {e}")
                    continue

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()

    def compute_layer_sensitivity(
        self,
        layer: nn.Module,
        param_ratios: Optional[List[float]] = None,
        baseline_ppl: float = 0.0,
        eval_fn: Optional[Any] = None,
        layer_name: Optional[str] = None,
    ) -> Dict[float, float]:
        """
        Compute sensitivity for a single layer at various compression ratios.

        This is a standalone utility method that can be called without
        initializing the full plugin context.

        Args:
            layer: The layer module (nn.Linear) to analyze
            param_ratios: List of compression ratios to evaluate (defaults to self.param_ratios)
            baseline_ppl: Baseline perplexity for delta computation
            eval_fn: Optional function(model) -> ppl to compute perplexity
                     If not provided, uses a simple proxy based on weight reconstruction error
            layer_name: Optional dotted layer name for state-backed transforms

        Returns:
            Dict mapping compression ratio to sensitivity score (PPL delta or error)
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError("layer must be nn.Linear")

        ratios = param_ratios if param_ratios is not None else self.param_ratios

        original_weight = layer.weight.data.clone()
        m, n = original_weight.shape

        sensitivity: Dict[float, float] = {}

        for ratio in ratios:
            k = max(1, int(ratio * min(m, n)))

            try:
                compressed = self._reconstruct_probe_weight(
                    original_weight,
                    rank=k,
                    layer_name=layer_name,
                )

                if eval_fn is not None:
                    # Use provided evaluation function
                    layer.weight.data = compressed.to(original_weight.dtype)
                    try:
                        ppl = eval_fn()
                        sensitivity[ratio] = ppl - baseline_ppl
                    finally:
                        layer.weight.data = original_weight
                else:
                    # Use reconstruction error as proxy for sensitivity
                    error = torch.norm(original_weight.float() - compressed.float()) / torch.norm(original_weight.float())
                    sensitivity[ratio] = float(error.item())

            except Exception:
                sensitivity[ratio] = float('inf')

        return sensitivity
