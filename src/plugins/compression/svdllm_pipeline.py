"""
SVD-LLM orchestration plugin.

Composes calibration, whitening, rank selection, SVD compression, and optional
closed-form update into a single production-oriented layer-by-layer pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class SVDLLMPipelinePlugin(Plugin):
    """
    Execute the SVD-LLM algorithm using Toggle's existing plugin components.

    The pipeline runs target layers sequentially so each later layer is
    calibrated against the current model state after earlier layers have
    already been compressed.
    """

    def __init__(
        self,
        *,
        target_modules: Optional[List[str]] = None,
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        min_rank: int = 1,
        regularization: float = 1e-6,
        svd_backend: str = "torch",
        svd_backend_config: Optional[Dict[str, Any]] = None,
        use_closed_form_update: bool = True,
        continue_on_error: bool = False,
        clear_intermediate_state: bool = False,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_modules = list(target_modules or [])
        self.rank = rank
        self.rank_ratio = rank_ratio
        self.min_rank = min_rank
        self.regularization = regularization
        self.svd_backend = svd_backend
        self.svd_backend_config = dict(svd_backend_config or {})
        self.use_closed_form_update = use_closed_form_update
        self.continue_on_error = continue_on_error
        self.clear_intermediate_state = clear_intermediate_state
        self.n_samples = n_samples

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Layerwise SVD-LLM orchestration using Toggle plugins",
            category="compression",
        )

    def _resolve_n_samples(self, dataloader: Any, n_samples: Optional[int]) -> int:
        if n_samples is not None:
            return int(n_samples)

        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None:
            try:
                return int(len(dataset))
            except TypeError:
                pass

        raise ValueError(
            "SVDLLMPipelinePlugin requires n_samples when dataloader.dataset length "
            "is unavailable."
        )

    def _expand_targets(
        self,
        model: nn.Module,
        target_modules: Sequence[str],
    ) -> List[Tuple[str, str]]:
        from .consolidator import ModelConsolidator

        helper = ModelConsolidator(compression_method="svd", target_modules=list(target_modules))
        expanded = helper._expand_target_modules(model, list(target_modules))
        return [
            (surgery_name, helper._normalize_state_layer_name(surgery_name))
            for surgery_name in expanded
        ]

    def _resolve_linear_module(self, model: nn.Module, surgery_name: str) -> nn.Module:
        from .consolidator import ModelConsolidator

        helper = ModelConsolidator(compression_method="svd", target_modules=[surgery_name])
        module = helper._get_layer_by_name(model, surgery_name)
        if module is None:
            raise ValueError(f"Target layer '{surgery_name}' was not found")
        if not hasattr(module, "weight") or getattr(module, "weight", None) is None:
            raise ValueError(f"Target layer '{surgery_name}' has no weight tensor")
        weight = module.weight
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(
                f"Target layer '{surgery_name}' is not a 2D weight-bearing linear-like module"
            )
        return module

    def _resolve_rank(
        self,
        *,
        module: nn.Module,
        state_name: str,
        rank: Optional[int],
        rank_ratio: Optional[float],
        min_rank: int,
    ) -> int:
        if rank is not None and rank_ratio is not None:
            raise ValueError("Specify either rank or rank_ratio, not both")

        stored_rank = None
        if self.state_manager is not None:
            stored_rank = self.state_manager.state.get(f"svd.ranks.{state_name}")

        m, n = tuple(int(dim) for dim in module.weight.shape)
        max_rank = min(m, n)

        if rank is not None:
            chosen_rank = int(rank)
        elif rank_ratio is not None:
            chosen_rank = int((m * n * float(rank_ratio)) / max(1, (m + n)))
        elif stored_rank is not None:
            chosen_rank = int(stored_rank)
        else:
            raise ValueError(
                "SVDLLMPipelinePlugin requires rank, rank_ratio, or pre-populated "
                f"svd.ranks.{state_name}"
            )

        chosen_rank = max(int(min_rank), min(int(chosen_rank), max_rank))
        if self.state_manager is not None:
            self.state_manager.state.set(f"svd.ranks.{state_name}", chosen_rank)
        return chosen_rank

    def _clear_layer_state(self, state_name: str) -> None:
        if self.state_manager is None:
            return

        keys = [
            f"calibration.activations.{state_name}",
            f"calibration.outputs.{state_name}",
            f"calibration.xtx.{state_name}",
        ]
        for key in keys:
            self.state_manager.state.delete(key)

    def do_execute(
        self,
        model: nn.Module = None,
        dataloader: Any = None,
        target_modules: Optional[List[str]] = None,
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        min_rank: Optional[int] = None,
        regularization: Optional[float] = None,
        svd_backend: Optional[str] = None,
        svd_backend_config: Optional[Dict[str, Any]] = None,
        use_closed_form_update: Optional[bool] = None,
        continue_on_error: Optional[bool] = None,
        clear_intermediate_state: Optional[bool] = None,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from .calibration_collector import CalibrationCollectorPlugin
        from .consolidator import ModelConsolidator
        from .svd_data_whitening import DataWhiteningPlugin

        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("SVDLLMPipelinePlugin requires a model")
        if dataloader is None:
            raise ValueError("SVDLLMPipelinePlugin requires a dataloader")

        target_modules = list(target_modules or self.target_modules)
        if not target_modules:
            raise ValueError("SVDLLMPipelinePlugin requires target_modules")

        rank = self.rank if rank is None else rank
        rank_ratio = self.rank_ratio if rank_ratio is None else rank_ratio
        min_rank = self.min_rank if min_rank is None else int(min_rank)
        regularization = self.regularization if regularization is None else float(regularization)
        svd_backend = self.svd_backend if svd_backend is None else svd_backend
        svd_backend_config = dict(self.svd_backend_config if svd_backend_config is None else svd_backend_config)
        use_closed_form_update = (
            self.use_closed_form_update
            if use_closed_form_update is None
            else bool(use_closed_form_update)
        )
        continue_on_error = self.continue_on_error if continue_on_error is None else bool(continue_on_error)
        clear_intermediate_state = (
            self.clear_intermediate_state
            if clear_intermediate_state is None
            else bool(clear_intermediate_state)
        )
        n_samples = self._resolve_n_samples(dataloader, self.n_samples if n_samples is None else n_samples)

        targets = self._expand_targets(model, target_modules)
        if not targets:
            raise ValueError(f"No target layers matched: {target_modules}")

        self.emit_event(
            "svdllm_pipeline.started",
            {
                "targets": [state_name for _, state_name in targets],
                "rank": rank,
                "rank_ratio": rank_ratio,
                "use_closed_form_update": use_closed_form_update,
            },
        )

        layer_results: Dict[str, Dict[str, Any]] = {}
        failures: Dict[str, str] = {}
        layers_replaced: List[str] = []

        for idx, (surgery_name, state_name) in enumerate(targets):
            try:
                module = self._resolve_linear_module(model, surgery_name)
                chosen_rank = self._resolve_rank(
                    module=module,
                    state_name=state_name,
                    rank=rank,
                    rank_ratio=rank_ratio,
                    min_rank=min_rank,
                )

                collector = CalibrationCollectorPlugin(
                    n_samples=n_samples,
                    collect_activations=use_closed_form_update,
                    collect_outputs=use_closed_form_update,
                    collect_xtx=True,
                    target_modules=[state_name],
                    name=f"{self.name}_calibration",
                )
                collector.initialize(self.context)
                calibration_result = collector.execute(model=model, dataloader=dataloader)

                whitening = DataWhiteningPlugin(
                    regularization=regularization,
                    name=f"{self.name}_whitening",
                )
                whitening.initialize(self.context)
                whitening_result = whitening.execute(target_layers=[state_name])

                if self.state_manager is None:
                    raise RuntimeError("StateManager not available")

                whitening_L = self.state_manager.state.get(f"svd.whitening.L.{state_name}")
                whitening_L_inv = self.state_manager.state.get(f"svd.whitening.L_inv.{state_name}")
                if whitening_L is None or whitening_L_inv is None:
                    raise RuntimeError(
                        f"Whitening factors were not produced for layer '{state_name}'"
                    )

                consolidator = ModelConsolidator(
                    compression_method="svd",
                    target_modules=[surgery_name],
                    rank=chosen_rank,
                    svd_backend=svd_backend,
                    svd_backend_config=svd_backend_config,
                    use_data_whitening=True,
                    use_closed_form_update=use_closed_form_update,
                )
                consolidator.initialize(self.context)
                compression_result = consolidator.compress_model_with_surgery(model)

                replaced = list(compression_result.parameters.get("layers_replaced", []))
                layers_replaced.extend(replaced)
                layer_results[state_name] = {
                    "surgery_name": surgery_name,
                    "state_name": state_name,
                    "rank": chosen_rank,
                    "calibration": calibration_result,
                    "whitening": whitening_result,
                    "compression": compression_result.parameters,
                }

                if clear_intermediate_state:
                    self._clear_layer_state(state_name)

            except Exception as exc:
                failures[state_name] = str(exc)
                if not continue_on_error:
                    raise

            self.update_progress(100.0 * (idx + 1) / max(1, len(targets)))

        result = {
            "targets": [state_name for _, state_name in targets],
            "layer_results": layer_results,
            "layers_processed": len(layer_results),
            "layers_replaced": layers_replaced,
            "failures": failures,
            "use_closed_form_update": use_closed_form_update,
        }

        self.emit_event(
            "svdllm_pipeline.completed",
            {
                "layers_processed": result["layers_processed"],
                "failures": len(failures),
            },
        )

        return result
