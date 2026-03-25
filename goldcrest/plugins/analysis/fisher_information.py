"""
Fisher Information Analysis Plugin.

Computes gradient-based importance weights for each layer.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class FisherInformationPlugin(Plugin):
    """
    Compute Fisher information (gradient-based importance) for each layer.

    Fisher information approximates the importance of each weight/activation
    dimension based on gradient magnitudes during forward/backward passes.

    Mathematical Background:
        Fisher information for parameter theta is:
            F(theta) = E[(d log p(x|theta) / d theta)^2]

        For neural networks, this is approximated as the expected squared
        gradient of the loss with respect to inputs/activations.

    Writes to:
        - calibration.fisher.<layer_name>: Fisher info vector for each layer

    Usage:
        plugin = FisherInformationPlugin(n_samples=32)
        plugin.initialize(context)
        result = plugin.execute(model=model, dataloader=dataloader)
    """

    def __init__(
        self,
        n_samples: int = 32,
        **kwargs
    ):
        """
        Initialize FisherInformationPlugin.

        Args:
            n_samples: Number of forward/backward samples for gradient accumulation
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Fisher information computation",
            category="analysis"
        )

    def do_execute(
        self,
        model: nn.Module = None,
        dataloader: Any = None,
        target_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute Fisher information for each layer.

        Args:
            model: The model to analyze
            dataloader: DataLoader for gradient computation (required)
            target_layers: Optional list of layer name patterns to analyze

        Returns:
            Dict with layers_processed count
        """
        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("FisherInformationPlugin requires a model")
        if dataloader is None:
            raise ValueError("FisherInformationPlugin requires a dataloader")

        self.emit_event("fisher.started", {"n_samples": self.n_samples})

        # Accumulators for squared gradients
        fisher_accum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}

        # Register backward hooks
        hooks = []

        def make_hook(name: str):
            def hook_fn(module, grad_input, grad_output):
                if grad_input[0] is not None:
                    g = grad_input[0].detach()
                    # Flatten batch dimension and compute mean squared gradient
                    g_flat = g.view(-1, g.shape[-1])
                    g_sq = (g_flat ** 2).mean(dim=0)

                    if name not in fisher_accum:
                        fisher_accum[name] = torch.zeros_like(g_sq)
                        counts[name] = 0

                    fisher_accum[name] += g_sq.cpu()
                    counts[name] += 1
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers and not any(t in name for t in target_layers):
                    continue
                hook = module.register_full_backward_hook(make_hook(name))
                hooks.append(hook)

        # Run forward/backward passes
        model.train()  # Need gradients
        device = next(model.parameters()).device
        samples = 0

        for batch in dataloader:
            if samples >= self.n_samples:
                break

            model.zero_grad()

            try:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                else:
                    inputs = batch.to(device)
                    outputs = model(inputs)
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss

                if hasattr(loss, 'backward'):
                    loss.backward()

                samples += 1
                self.update_progress(100.0 * samples / self.n_samples)

            except Exception as e:
                self.logger.warning(f"Error in forward/backward pass: {e}")
                continue

        # Remove hooks
        for h in hooks:
            h.remove()

        model.eval()

        # Normalize and store
        if self.state_manager:
            for name, fisher in fisher_accum.items():
                fisher_normalized = fisher / max(1, counts[name])
                self.state_manager.state.set(f"calibration.fisher.{name}", fisher_normalized)

        self.emit_event("fisher.completed", {"layers": len(fisher_accum)})

        return {
            "layers_processed": len(fisher_accum),
        }

    def compute_fisher_diagonal(
        self,
        layer: nn.Module,
        inputs: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Compute diagonal Fisher information for a single layer.

        This method computes the Fisher information diagonal approximation
        based on gradient magnitudes for the given inputs.

        Args:
            layer: The layer module (typically nn.Linear)
            inputs: Input tensor to the layer
            num_samples: Number of samples for estimation

        Returns:
            Diagonal Fisher information tensor
        """
        if not isinstance(layer, nn.Module):
            raise TypeError("layer must be a torch.nn.Module")

        fisher_diag = None
        count = 0

        layer.train()
        original_requires_grad = layer.weight.requires_grad
        layer.weight.requires_grad = True

        for _ in range(num_samples):
            layer.zero_grad()

            # Forward pass
            output = layer(inputs)

            # Use sum as pseudo-loss for gradient computation
            loss = output.sum()
            loss.backward()

            if layer.weight.grad is not None:
                grad_sq = (layer.weight.grad ** 2).detach()
                if fisher_diag is None:
                    fisher_diag = torch.zeros_like(grad_sq)
                fisher_diag += grad_sq
                count += 1

        layer.weight.requires_grad = original_requires_grad
        layer.eval()

        if fisher_diag is not None and count > 0:
            fisher_diag = fisher_diag / count

        return fisher_diag if fisher_diag is not None else torch.tensor([])

    def compute_importance_scores(
        self,
        fisher_info: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute importance scores from Fisher information.

        Args:
            fisher_info: Dictionary mapping layer names to Fisher diagonal tensors

        Returns:
            Dictionary mapping layer names to scalar importance scores
        """
        scores: Dict[str, float] = {}

        for name, fisher in fisher_info.items():
            if fisher is None or fisher.numel() == 0:
                scores[name] = 0.0
            else:
                # Importance = sum of Fisher diagonal elements (total information)
                scores[name] = float(fisher.sum().item())

        # Normalize to [0, 1] range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores
