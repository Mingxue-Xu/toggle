from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..analysis.layer_svd_rank_decider import LayerSVDRankDecider
from ...framework.plugins import Plugin, PluginMetadata


class ActivationDrivenSVDWeightsCompressionPlugin(Plugin):
    """
    Activation-driven SVD compression with per-layer ranks and model surgery.

    Steps:
      1) Resolve activation report (from config.report_path or state).
      2) Compute per-layer ranks via LayerSVDRankDecider.
      3) Use ModelConsolidator with method_overrides to apply SVD with ranks.
    """

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Activation-driven SVD compression with model surgery",
            category="compression",
        )

    def _resolve_report(self, svd_cfg: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Explicit path
        report_path = svd_cfg.get("report_path")
        if report_path:
            from ..analysis.report_loader import load_report
            return load_report(report_path)
        # 2) From state
        if self.state_manager:
            rep = self.state_manager.state.get("analysis.activation_metrics")
            if isinstance(rep, dict) and rep.get("per_layer") is not None:
                return rep
        raise ValueError("No activation report available for SVD compression")

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        config = dict(self._configuration or {})
        # Config section can be either top-level 'svd_compression' or 'compression'.get('svd', {})
        svd_cfg = dict(config.get("svd_compression") or {})
        if not svd_cfg:
            svd_cfg = dict((config.get("compression") or {}).get("svd", {}) or {})
        if not svd_cfg:
            raise ValueError("Missing svd_compression (or compression.svd) configuration")

        # Resolve model
        model = kwargs.get("model")
        if model is None:
            model = self.get_model("current")

        # Load report
        report = kwargs.get("report") or self._resolve_report(svd_cfg)

        # Decide ranks
        decider = LayerSVDRankDecider(svd_cfg)
        decider_result = decider.decide(model, report)
        layer_ranks: Dict[str, int] = decider_result.get("layer_ranks", {})
        energy_thresholds: Dict[str, float] = decider_result.get("energy_thresholds", {})
        overrides: List[Dict[str, Any]] = []
        target_modules: List[str] = []

        if layer_ranks:
            target_modules = list(layer_ranks.keys())
            for path, k in layer_ranks.items():
                overrides.append({
                    "pattern": path,
                    "func_name": "svd",
                    "rank": int(k),
                    "granularity": "matrix",
                })
        elif energy_thresholds:
            target_modules = list(energy_thresholds.keys())
            for path, tau in energy_thresholds.items():
                overrides.append({
                    "pattern": path,
                    "func_name": "svd",
                    "preserve_energy": float(tau),
                    "granularity": "matrix",
                })
        else:
            raise RuntimeError("LayerSVDRankDecider returned no targets (neither ranks nor energy thresholds)")

        # Use ModelConsolidator to apply surgery
        from .consolidator import ModelConsolidator
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=target_modules,
            **config  # pass through top-level options (e.g., device/backend if any)
        )
        consolidator.initialize(self.context)
        result = consolidator.compress_model_with_surgery(
            model,
            overrides=overrides,
            granularity="matrix",
        )

        out = {
            "decider": decider_result,
            "compression": {
                "layers_replaced": result.parameters.get("layers_replaced", []),
                "surgery_stats": result.parameters.get("surgery_stats", {}),
                "overall_ratio": result.total_compression_ratio,
            },
        }
        # Save selection to state for debugging
        if self.state_manager:
            self.state_manager.state.set("compression.svd.last", out)
        return out
