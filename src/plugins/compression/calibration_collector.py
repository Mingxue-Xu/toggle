"""
Calibration data collection plugin for ASVD and SVD-LLM.

Collects activations, outputs, and X^TX matrices in a single forward pass
for consumption by ASVD and SVD-LLM plugins.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class CalibrationCollectorPlugin(Plugin):
    """
    Collects calibration data (activations, inputs, outputs) in a single forward pass.

    Stores collected data in StateManager for consumption by other plugins:
    - calibration.activations.<layer_name>: Input activations per layer
    - calibration.outputs.<layer_name>: Output tensors per layer
    - calibration.xtx.<layer_name>: X^T @ X matrices for whitening
    - calibration.layer_names: List of layer names with collected data
    - calibration.collected: Boolean flag indicating calibration completed
    - calibration.sample_count: Number of samples collected

    This eliminates redundant forward passes across multiple plugins.

    Usage:
        plugin = CalibrationCollectorPlugin(n_samples=256, collect_xtx=True)
        plugin.initialize(context)
        result = plugin.execute(model=model, dataloader=dataloader)
    """

    def __init__(
        self,
        n_samples: int = 256,
        collect_activations: bool = True,
        collect_outputs: bool = True,
        collect_xtx: bool = True,
        collect_fisher: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize CalibrationCollectorPlugin.

        Args:
            n_samples: Number of calibration samples to collect
            collect_activations: Whether to store input activations
            collect_outputs: Whether to store layer outputs
            collect_xtx: Whether to compute and store X^T @ X matrices
            collect_fisher: Whether to collect Fisher information (requires gradients)
            target_modules: List of module name patterns to target (None = all Linear)
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.collect_activations = collect_activations
        self.collect_outputs = collect_outputs
        self.collect_xtx = collect_xtx
        self.collect_fisher = collect_fisher
        self.target_modules = target_modules

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Collects calibration data for ASVD/SVD-LLM plugins",
            category="calibration"
        )

    def do_execute(
        self,
        model: nn.Module = None,
        dataloader: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute calibration data collection.

        Args:
            model: The model to collect activations from
            dataloader: DataLoader providing calibration data

        Returns:
            Dict with layers_collected and samples count
        """
        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("CalibrationCollector requires a model")
        if dataloader is None:
            raise ValueError("CalibrationCollector requires a dataloader")

        # Emit start event
        self.emit_event("calibration.started", {
            "n_samples": self.n_samples,
            "collect_activations": self.collect_activations,
            "collect_xtx": self.collect_xtx,
        })

        # Storage for collected data
        activations: Dict[str, List[torch.Tensor]] = {}
        outputs: Dict[str, List[torch.Tensor]] = {}
        xtx_accum: Dict[str, torch.Tensor] = {}
        xtx_counts: Dict[str, int] = {}

        hooks = []

        def make_hook(name: str):
            def hook_fn(module, inp, out):
                x = inp[0].detach()

                if self.collect_activations:
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())

                if self.collect_outputs:
                    if name not in outputs:
                        outputs[name] = []
                    out_tensor = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
                    outputs[name].append(out_tensor.cpu())

                if self.collect_xtx:
                    # Flatten to 2D: (batch * seq_len, hidden_dim)
                    x_flat = x.view(-1, x.shape[-1]).float()
                    xtx = x_flat.T @ x_flat
                    if name not in xtx_accum:
                        xtx_accum[name] = torch.zeros_like(xtx)
                        xtx_counts[name] = 0
                    xtx_accum[name] += xtx.cpu()
                    xtx_counts[name] += x_flat.shape[0]

            return hook_fn

        # Register hooks on target modules
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_modules and not any(t in name for t in self.target_modules):
                continue
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

        # Run forward passes
        model.eval()
        device = next(model.parameters()).device
        samples_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= self.n_samples:
                    break

                remaining = max(0, self.n_samples - samples_collected)
                model_inputs, batch_examples = self._prepare_batch_inputs(
                    batch=batch,
                    device=device,
                    max_examples=remaining,
                )

                if isinstance(model_inputs, dict):
                    model(**model_inputs)
                else:
                    model(model_inputs)

                samples_collected += batch_examples
                self.update_progress(100.0 * samples_collected / self.n_samples)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Store in StateManager
        if self.state_manager:
            if self.collect_activations:
                for name, acts in activations.items():
                    # Store concatenated activations
                    self.state_manager.state.set(
                        f"calibration.activations.{name}",
                        torch.cat(acts, dim=0)
                    )

            if self.collect_outputs:
                for name, outs in outputs.items():
                    self.state_manager.state.set(
                        f"calibration.outputs.{name}",
                        torch.cat(outs, dim=0)
                    )

            if self.collect_xtx:
                for name, xtx in xtx_accum.items():
                    # Normalize by total token count
                    self.state_manager.state.set(
                        f"calibration.xtx.{name}",
                        xtx / xtx_counts[name]
                    )

            self.state_manager.state.set("calibration.collected", True)
            self.state_manager.state.set("calibration.sample_count", samples_collected)

            # Store list of layer names for downstream plugins
            all_layer_names = list(set(
                list(activations.keys()) +
                list(outputs.keys()) +
                list(xtx_accum.keys())
            ))
            self.state_manager.state.set("calibration.layer_names", all_layer_names)

        # Emit completion event
        layers_collected = len(activations) if activations else len(xtx_accum)
        self.emit_event("calibration.completed", {
            "layers_collected": layers_collected,
            "samples": samples_collected,
        })

        return {
            "layers_collected": layers_collected,
            "samples": samples_collected,
        }

    def _prepare_batch_inputs(
        self,
        batch: Any,
        device: torch.device,
        max_examples: Optional[int] = None,
    ) -> tuple[Any, int]:
        """Move a batch to device, preserve structure, and optionally trim it."""
        if isinstance(batch, dict):
            prepared = {
                key: self._maybe_trim_tensor(value, max_examples).to(device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            return prepared, self._infer_batch_size(prepared)

        if isinstance(batch, (list, tuple)):
            tensors = [
                self._maybe_trim_tensor(value, max_examples).to(device)
                if isinstance(value, torch.Tensor)
                else value
                for value in batch
            ]
            if not tensors:
                raise ValueError("Calibration batch cannot be empty")

            if len(tensors) == 1:
                tensor = tensors[0]
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("Single-item calibration batch must contain a tensor")
                return tensor, self._infer_batch_size(tensor)

            if len(tensors) <= 3 and all(
                value is None or isinstance(value, torch.Tensor) for value in tensors
            ):
                keys = ("input_ids", "attention_mask", "labels")
                prepared = {
                    key: value
                    for key, value in zip(keys, tensors)
                    if isinstance(value, torch.Tensor)
                }
                return prepared, self._infer_batch_size(prepared)

            raise ValueError(
                "Ambiguous tuple/list calibration batch. Expected "
                "(input_ids,), (input_ids, attention_mask), or "
                "(input_ids, attention_mask, labels)."
            )

        if isinstance(batch, torch.Tensor):
            tensor = self._maybe_trim_tensor(batch, max_examples).to(device)
            return tensor, self._infer_batch_size(tensor)

        raise ValueError(f"Unsupported calibration batch type: {type(batch).__name__}")

    def _infer_batch_size(self, batch: Any) -> int:
        """Infer example count from the leading tensor dimension."""
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0] if value.ndim > 0 else 1
            return 0

        if isinstance(batch, torch.Tensor):
            return batch.shape[0] if batch.ndim > 0 else 1

        raise ValueError(f"Cannot infer batch size from type: {type(batch).__name__}")

    def _maybe_trim_tensor(
        self,
        value: torch.Tensor,
        max_examples: Optional[int],
    ) -> torch.Tensor:
        """Trim tensor batches so collection stops on true example count."""
        if max_examples is None or max_examples <= 0 or value.ndim == 0:
            return value
        if value.shape[0] <= max_examples:
            return value
        return value[:max_examples]
