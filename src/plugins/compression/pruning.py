from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from ..analysis.report_loader import load_report
from ..analysis.layer_selector import LayerMetricSelector
from ...framework.plugins import Plugin, PluginMetadata

# Configuration dataclass and loader (merged from pruning_config.py)
from dataclasses import dataclass
from src.config.loader import ConfigurationLoader


@dataclass(frozen=True)
class PruningConfig:
    report_path: Path
    selection_metric_type: str
    group_prefix: str = "layers"
    include_names: Optional[List[str]] = None
    module_types: Optional[List[str]] = None
    aggregator: str = "mean"
    weights: str = "uniform"
    num_deleted_block: Optional[int] = None
    ratio_deleted_block: Optional[float] = None
    min_keep: int = 1
    container_path: str = "model.layers"


def load_pruning_config(config_path: Path | str) -> PruningConfig:
    loader = ConfigurationLoader()
    cfg = loader.load_config(Path(config_path))
    pruning: Dict[str, Any] = (cfg or {}).get("pruning") or {}

    report_path = Path(pruning.get("report_path"))
    if not report_path:
        raise ValueError("pruning.report_path is required")

    return PruningConfig(
        report_path=report_path,
        selection_metric_type=str(pruning.get("selection_metric_type")),
        group_prefix=str(pruning.get("group_prefix", "layers")),
        include_names=list(pruning.get("include_names") or []) or None,
        module_types=list(pruning.get("module_types") or []) or None,
        aggregator=str(pruning.get("aggregator", "mean")),
        weights=str(pruning.get("weights", "uniform")),
        num_deleted_block=pruning.get("num_deleted_block"),
        ratio_deleted_block=pruning.get("ratio_deleted_block"),
        min_keep=int(pruning.get("min_keep", 1)),
        container_path=str(pruning.get("container_path", "model.layers")),
    )


# Plugin implementation (merged from pruning_plugin.py)

class PruningPlugin(Plugin):
    """Selects transformer blocks by activation metrics and prunes them in-place."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Selects blocks by activation metrics and prunes them in-place",
            category="compression",
        )

    def _resolve_report(self, pruning_cfg: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Explicit report_path wins
        report_path = pruning_cfg.get("report_path")
        if report_path:
            return load_report(report_path)

        # 2) Prefer report previously stored by ActivationMetricsPlugin
        if self.state_manager:
            rep = self.state_manager.state.get("analysis.activation_metrics")
            if isinstance(rep, dict) and rep.get("per_layer") is not None:
                return rep

        raise ValueError(
            "No activation report available: provide pruning.report_path or run ActivationMetricsPlugin first"
        )

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        config = dict(self._configuration or {})
        pruning_cfg = dict(config.get("pruning") or {})

        # Validate selector configuration
        if not pruning_cfg.get("selection_metric_type"):
            raise ValueError("pruning.selection_metric_type is required")

        report = self._resolve_report(pruning_cfg)

        selector = LayerMetricSelector(
            selection_metric_type=str(pruning_cfg.get("selection_metric_type")),
            group_prefix=str(pruning_cfg.get("group_prefix", "layers")),
            include_names=pruning_cfg.get("include_names"),
            module_types=pruning_cfg.get("module_types"),
            aggregator=str(pruning_cfg.get("aggregator", "mean")),
            weights=str(pruning_cfg.get("weights", "uniform")),
        )
        block_scores = selector.compute_scores(report)
        if not block_scores.scores:
            # Relax selection if filters were too strict
            self.logger.warning(
                "No block scores computed with provided filters; relaxing selection (ignore include_names/module_types)"
            )
            relaxed = LayerMetricSelector(
                selection_metric_type=str(pruning_cfg.get("selection_metric_type")),
                group_prefix=str(pruning_cfg.get("group_prefix", "layers")),
                include_names=None,
                module_types=None,
                aggregator=str(pruning_cfg.get("aggregator", "mean")),
                weights=str(pruning_cfg.get("weights", "uniform")),
            )
            block_scores = relaxed.compute_scores(report)
        if not block_scores.scores:
            # As a final fallback, switch to a basic built-in metric to ensure progress
            fallback_metric = "l2_norm.median"
            self.logger.warning(
                f"No block scores available for metric '{pruning_cfg.get('selection_metric_type')}'. "
                f"Falling back to '{fallback_metric}'."
            )
            relaxed2 = LayerMetricSelector(
                selection_metric_type=fallback_metric,
                group_prefix=str(pruning_cfg.get("group_prefix", "layers")),
                include_names=None,
                module_types=None,
                aggregator=str(pruning_cfg.get("aggregator", "mean")),
                weights=str(pruning_cfg.get("weights", "uniform")),
            )
            block_scores = relaxed2.compute_scores(report)

        # Determine how many to prune
        num_deleted_block = pruning_cfg.get("num_deleted_block")
        ratio_deleted_block = pruning_cfg.get("ratio_deleted_block")
        min_keep = int(pruning_cfg.get("min_keep", 1))
        selected = LayerMetricSelector.choose_to_prune(
            block_scores.scores,
            num_deleted_block=num_deleted_block,
            ratio_deleted_block=ratio_deleted_block,
            min_keep=min_keep,
        )

        # Perform pruning on provided model or context model
        model = kwargs.get("model")
        if model is None:
            model = self.get_model("current")

        container_path = str(pruning_cfg.get("container_path", "model.layers"))

        from .consolidator import ModelConsolidator

        consolidator = ModelConsolidator()
        pruning_stats = consolidator.remove_transformer_blocks(model, container_path, selected)

        result = {
            "scores": block_scores.scores,
            "selected_indices": selected,
            "container_path": container_path,
            "pruning_stats": pruning_stats,
        }

        if self.state_manager:
            self.state_manager.state.set("pruning.last_selection", result)

        return result


def select_blocks_from_config(cfg_path: Path | str) -> Dict[str, object]:
    """
    Load pruning configuration and compute block scores and selected indices (no model required).

    Returns a dict with selection and pruning stats.
    """
    cfg: PruningConfig = load_pruning_config(cfg_path)

    data = load_report(cfg.report_path)
    selector = LayerMetricSelector(
        selection_metric_type=cfg.selection_metric_type,
        group_prefix=cfg.group_prefix,
        include_names=cfg.include_names,
        module_types=cfg.module_types,
        aggregator=cfg.aggregator,
        weights=cfg.weights,
    )
    block_scores = selector.compute_scores(data)
    selected = LayerMetricSelector.choose_to_prune(
        block_scores.scores,
        num_deleted_block=cfg.num_deleted_block,
        ratio_deleted_block=cfg.ratio_deleted_block,
        min_keep=cfg.min_keep,
    )
    return {
        "scores": block_scores.scores,
        "selected_indices": selected,
        "container_path": cfg.container_path,
    }


def prune_model_from_config(model: object, cfg_path: Path | str) -> Dict[str, object]:
    """
    Load pruning configuration, compute block scores, select blocks, and remove them from the model.
    """
    sel = select_blocks_from_config(cfg_path)
    selected = sel["selected_indices"]

    # Import consolidator lazily to avoid importing torch at module import time
    from .consolidator import ModelConsolidator

    consolidator = ModelConsolidator()
    stats = consolidator.remove_transformer_blocks(model, sel["container_path"], selected)
    out = dict(sel)
    out["pruning_stats"] = stats
    return out
