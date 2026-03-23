"""
Weight metrics analysis plugin and analyzer.

Computes configured metrics over model weight tensors and builds a report
using WeightsStatisticsManager.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch


from ...framework.plugins import Plugin, PluginMetadata
from ...framework.statistics import WeightsStatisticsManager
from ...framework.model_manager import ModelManager

from .metric_utils import BasicMetricsBackend, ExternalMetricsBackend


class WeightMetricsAnalyzer:
    """Pure-logic analyzer to compute metrics per selected tensor."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def analyze_model(
        self,
        model: torch.nn.Module,
        metrics_to_use: List[str],
        selection: Dict[str, Any],
        compute: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        move_to_cpu = bool(compute.get("move_to_cpu", True))
        max_layers = selection.get("max_layers")
        include_names = selection.get("include_names") or []
        exclude_names = selection.get("exclude_names") or []
        module_types = set(selection.get("module_types") or [])
        param_kinds = set(selection.get("parameter_kinds") or ["weight"])

        # Build mapping from param full name to module type
        module_type_map: Dict[str, str] = {}
        for mod_name, module in model.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full = f"{mod_name}.{pname}" if mod_name else pname
                module_type_map[full] = type(module).__name__

        selected: List[str] = []
        for name, param in model.named_parameters():
            kind = name.split(".")[-1]
            if param_kinds and kind not in param_kinds:
                continue
            if include_names and not any(pat in name for pat in include_names):
                continue
            if exclude_names and any(pat in name for pat in exclude_names):
                continue
            mtype = module_type_map.get(name, "")
            if module_types and mtype not in module_types:
                continue
            selected.append(name)
        if max_layers is not None:
            selected = selected[: int(max_layers)]

        results: List[Dict[str, Any]] = []
        with torch.no_grad():
            for name in selected:
                tensor = dict(model.named_parameters())[name]
                t = tensor.detach()
                if move_to_cpu and t.device.type != "cpu":
                    t = t.to("cpu")
                values: Dict[str, Any] = {}
                for mname in metrics_to_use:
                    try:
                        fn = self.backend.get(mname)
                    except KeyError:
                        continue
                    out = fn(t)
                    if isinstance(out, dict):
                        # flatten
                        for k, v in out.items():
                            values[f"{mname}.{k}"] = float(v) if isinstance(v, (int, float)) else v
                    else:
                        values[mname] = float(out) if isinstance(out, (int, float)) else out
                results.append(
                    {
                        "name": name,
                        "module_type": module_type_map.get(name, ""),
                        "shape": list(t.shape),
                        "numel": int(t.numel()),
                        "device": str(tensor.device),
                        "dtype": str(tensor.dtype),
                        "values": values,
                    }
                )
        return results


class WeightMetricsPlugin(Plugin):
    """Plugin to compute weight metrics and persist a statistics report."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Compute metrics on model weights and generate a report",
            category="analysis",
        )

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        start = time.time()
        config = dict(self._configuration)
        analysis_cfg = config.get("analysis", {})
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        compute_cfg = analysis_cfg.get("compute", {})
        output_cfg = analysis_cfg.get("output", {})

        # Resolve model
        model = kwargs.get("model")
        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("No model available for weight metrics analysis")

        # Resolve metrics backend
        ext_import = (metrics_cfg.get("import", {}) if isinstance(metrics_cfg, dict) else {})
        backend = ExternalMetricsBackend(
            module_path=ext_import.get("module"),
            file_path=ext_import.get("file"),
            name_prefix=ext_import.get("name_prefix"),
        )
        provenance = backend.provenance
        if not provenance.get("found", False):
            backend = BasicMetricsBackend(include_advanced=True)

        # Determine metrics list
        requested = metrics_cfg.get("names", []) if isinstance(metrics_cfg, dict) else []
        if requested == "all":
            metrics_to_use = list(backend.list_metrics().keys())
        else:
            metrics_to_use = [str(n).lower() for n in (requested or list(backend.list_metrics().keys()))]

        analyzer = WeightMetricsAnalyzer(backend)
        raw_results = analyzer.analyze_model(
            model=model,
            metrics_to_use=metrics_to_use,
            selection=selection_cfg,
            compute=compute_cfg,
        )

        # Build report
        stats_mgr = WeightsStatisticsManager()
        model_info = self.model_manager.get_model_info(model) if self.model_manager else {}
        timings = {
            "runtime_sec": time.time() - start,
            "external_metrics": provenance,
        }
        report = stats_mgr.build_report(
            model_info=model_info, raw_results=raw_results, config=config, timings=timings
        )

        # Persist
        report_dir = output_cfg.get("report_dir") or f"analysis/weight_metrics"
        stem = output_cfg.get("stem", "report")
        # Append timestamp to distinguish runs unless explicitly disabled
        if output_cfg.get("timestamp", True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"{stem}_{ts}"
        base_path = stats_mgr.build_report_path(self.context, report_dir, stem)
        if output_cfg.get("save_json", True):
            stats_mgr.save_json(report, base_path.with_suffix(".json"))
        if output_cfg.get("save_csv", False):
            rows = stats_mgr.extract_csv_rows(report)
            stats_mgr.save_csv(rows, base_path.with_suffix(".csv"))

        # Store in state
        if self.state_manager:
            self.state_manager.state.set("analysis.weight_metrics", report)

        return report
