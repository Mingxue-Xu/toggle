"""
Statistics/report managers for analysis outputs.

Provides a base StatisticsManager with load/save and path helpers,
and a WeightsStatisticsManager that shapes weight-metrics reports.
"""
from __future__ import annotations

import json
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class StatisticsManager:
    """Base manager for saving/loading analysis reports and collecting statistics."""

    schema_version: str = "1.0"

    def __init__(self):
        """Initialize the statistics manager with empty data stores."""
        self._data: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Any] = {}

    def add(self, metric_name: str, value: float) -> None:
        """
        Add a metric value to the statistics collection.

        Args:
            metric_name: Name of the metric (e.g., "loss", "accuracy").
            value: The value to record.
        """
        if metric_name not in self._data:
            self._data[metric_name] = []
        self._data[metric_name].append(float(value))

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get aggregated statistics for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Dictionary with mean, std, min, max, median, count.
        """
        import statistics as stats_module

        values = self._data.get(metric_name, [])
        if not values:
            return {"count": 0}

        result: Dict[str, float] = {
            "count": len(values),
            "mean": stats_module.mean(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            result["std"] = stats_module.pstdev(values)
            result["median"] = stats_module.median(values)
        else:
            result["std"] = 0.0
            result["median"] = values[0]
        return result

    def aggregate(self, metric_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Aggregate statistics for multiple metrics.

        Args:
            metric_names: List of metric names to aggregate. If None, aggregates all.

        Returns:
            Dictionary mapping metric names to their statistics.
        """
        names = metric_names if metric_names is not None else list(self._data.keys())
        return {name: self.get_stats(name) for name in names}

    def clear(self) -> None:
        """Clear all collected statistics."""
        self._data.clear()
        self._metadata.clear()

    def build_report_path(self, context, subdir: str, stem: str) -> Path:
        """Build a path under the context workspace for a report file.

        Args:
            context: PipelineContext with workspace_dir
            subdir: subdirectory under workspace (e.g., "analysis/weight_metrics")
            stem: filename stem without extension
        """
        # Allow absolute output paths; otherwise, resolve under workspace
        p = Path(subdir)
        base = p if p.is_absolute() else context.get_workspace_path(subdir)
        base.mkdir(parents=True, exist_ok=True)
        return base / stem

    def save_json(self, report: Dict[str, Any], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON report to {path}")
        return path

    def save_csv(self, rows: List[Dict[str, Any]], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            # Write empty file with no header
            path.write_text("")
            logger.info(f"Saved empty CSV to {path}")
            return path
        # Collect headers from union of keys
        headers = sorted({k for row in rows for k in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info(f"Saved CSV report to {path}")
        return path

    def load_json(self, path: Path) -> Dict[str, Any]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class WeightsStatisticsManager(StatisticsManager):
    """Report builder for weight-only metrics analysis."""

    schema_version: str = "1.0"

    def __init__(self):
        """Initialize the weights statistics manager."""
        super().__init__()
        self._weight_stats: List[Dict[str, Any]] = []

    def collect_from_model(
        self,
        model: Any,
        metrics: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect weight statistics from a PyTorch model.

        Args:
            model: PyTorch model to analyze.
            metrics: List of metrics to compute (e.g., ["l2_norm", "mean", "std"]).
                     If None, computes basic metrics.
            include_patterns: Optional list of name patterns to include.
            exclude_patterns: Optional list of name patterns to exclude.

        Returns:
            List of dictionaries containing per-parameter statistics.
        """
        import torch
        import re

        default_metrics = ["l2_norm", "l1_norm", "mean", "std", "sparsity", "max_abs"]
        metrics_to_use = metrics if metrics is not None else default_metrics

        results: List[Dict[str, Any]] = []

        for name, param in model.named_parameters():
            # Apply include/exclude filters
            if include_patterns:
                if not any(re.search(p, name) for p in include_patterns):
                    continue
            if exclude_patterns:
                if any(re.search(p, name) for p in exclude_patterns):
                    continue

            if not isinstance(param, torch.Tensor):
                continue

            data = param.detach().float()
            values: Dict[str, float] = {}

            if "l2_norm" in metrics_to_use:
                values["l2_norm"] = float(torch.norm(data, p=2).item())
            if "l1_norm" in metrics_to_use:
                values["l1_norm"] = float(torch.norm(data, p=1).item())
            if "mean" in metrics_to_use:
                values["mean"] = float(data.mean().item())
            if "std" in metrics_to_use:
                values["std"] = float(data.std().item()) if data.numel() > 1 else 0.0
            if "sparsity" in metrics_to_use:
                values["sparsity"] = float((data == 0).sum().item() / data.numel())
            if "max_abs" in metrics_to_use:
                values["max_abs"] = float(data.abs().max().item())

            entry = {
                "name": name,
                "module_type": type(param).__name__,
                "shape": list(param.shape),
                "numel": param.numel(),
                "device": str(param.device),
                "dtype": str(param.dtype),
                "values": values,
            }
            results.append(entry)

            # Also add to internal tracking
            for metric_name, metric_value in values.items():
                self.add(f"{name}.{metric_name}", metric_value)

        self._weight_stats = results
        return results

    def build_report(
        self,
        model_info: Dict[str, Any],
        raw_results: List[Dict[str, Any]],
        config: Dict[str, Any],
        timings: Dict[str, Any],
    ) -> Dict[str, Any]:
        analysis_cfg = config.get("analysis", {})
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        aggregation_cfg = analysis_cfg.get("aggregation", {})
        compute_cfg = analysis_cfg.get("compute", {})

        per_layer = []
        metric_names: List[str] = []
        for item in raw_results:
            values: Dict[str, Any] = item.get("values", {})
            metric_names.extend(list(values.keys()))
            per_layer.append(
                {
                    "name": item.get("name"),
                    "module_type": item.get("module_type", ""),
                    "shape": list(item.get("shape", [])),
                    "numel": int(item.get("numel", 0)),
                    "device": str(item.get("device", "")),
                    "dtype": str(item.get("dtype", "")),
                    "statistics": self._values_to_statistics(values),
                }
            )

        metric_names = sorted(set(metric_names))
        reductions = aggregation_cfg.get(
            "reductions", ["mean", "std", "min", "max", "median"]
        )

        report: Dict[str, Any] = {
            "version": self.schema_version,
            "model": {
                "type": model_info.get("type"),
                "parameter_count": model_info.get("parameter_count"),
                "device": model_info.get("device"),
                "is_compressed": model_info.get("is_compressed", False),
            },
            "analysis": {
                "type": "weight_metrics",
                "metrics": metrics_cfg.get("names", []),
                "selection": selection_cfg,
                "aggregation": {"reductions": reductions},
                "compute": compute_cfg,
            },
            "summary": {
                "layers_analyzed": len(per_layer),
                "parameters_analyzed": sum(pl.get("numel", 0) for pl in per_layer),
                "metrics": metric_names,
                "reductions": reductions,
                "runtime_sec": float(timings.get("runtime_sec", 0.0)),
            },
            "per_layer": per_layer,
            "by_module_type": self.aggregate_by_module_type(per_layer, reductions),
            "global": self.aggregate_global(per_layer, reductions),
            "provenance": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "seed": compute_cfg.get("seed"),
                "framework_versions": {},
                "external_metrics": timings.get("external_metrics", {}),
            },
        }
        return report

    def _values_to_statistics(self, values: Dict[str, Any]) -> Dict[str, Any]:
        # Values are already scalar metrics; place them under a single "value" reduction
        return {k: {"value": v} for k, v in values.items()}

    def aggregate_by_module_type(
        self, per_layer: List[Dict[str, Any]], reductions: List[str]
    ) -> Dict[str, Any]:
        # Since we only have scalar "value" per metric, aggregation just reduces across layers
        from statistics import mean

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in per_layer:
            groups.setdefault(item.get("module_type", ""), []).append(item)

        result: Dict[str, Any] = {}
        for mtype, items in groups.items():
            stats_acc: Dict[str, List[float]] = {}
            for it in items:
                for metric_name, metric_obj in it.get("statistics", {}).items():
                    val = metric_obj.get("value")
                    if isinstance(val, (int, float)):
                        stats_acc.setdefault(metric_name, []).append(float(val))
            module_stats: Dict[str, Any] = {}
            for metric_name, values in stats_acc.items():
                if not values:
                    continue
                module_stats[metric_name] = self._reduce(values, reductions)
            result[mtype] = {
                "count": len(items),
                "total_numel": sum(it.get("numel", 0) for it in items),
                "statistics": module_stats,
            }
        return result

    def aggregate_global(
        self, per_layer: List[Dict[str, Any]], reductions: List[str]
    ) -> Dict[str, Any]:
        stats_acc: Dict[str, List[float]] = {}
        for it in per_layer:
            for metric_name, metric_obj in it.get("statistics", {}).items():
                val = metric_obj.get("value")
                if isinstance(val, (int, float)):
                    stats_acc.setdefault(metric_name, []).append(float(val))
        global_stats: Dict[str, Any] = {}
        for metric_name, values in stats_acc.items():
            if not values:
                continue
            global_stats[metric_name] = self._reduce(values, reductions)
        return global_stats

    def _reduce(self, values: List[float], reductions: List[str]) -> Dict[str, float]:
        import math
        import statistics

        out: Dict[str, float] = {}
        if not values:
            return out
        v = sorted(values)
        if "mean" in reductions:
            out["mean"] = float(statistics.mean(v))
        if "std" in reductions:
            out["std"] = float(statistics.pstdev(v)) if len(v) > 1 else 0.0
        if "min" in reductions:
            out["min"] = float(v[0])
        if "max" in reductions:
            out["max"] = float(v[-1])
        if "median" in reductions:
            out["median"] = float(statistics.median(v))
        if "p25" in reductions:
            out["p25"] = float(self._percentile(v, 25))
        if "p75" in reductions:
            out["p75"] = float(self._percentile(v, 75))
        return out

    def _percentile(self, v: List[float], p: float) -> float:
        if not v:
            return 0.0
        if len(v) == 1:
            return float(v[0])
        k = (len(v) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(v) - 1)
        if f == c:
            return float(v[int(k)])
        d0 = v[f] * (c - k)
        d1 = v[c] * (k - f)
        return float(d0 + d1)

    def extract_csv_rows(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in report.get("per_layer", []):
            base = {
                "name": item.get("name"),
                "module_type": item.get("module_type"),
                "shape": str(item.get("shape")),
                "numel": item.get("numel"),
                "device": item.get("device"),
                "dtype": item.get("dtype"),
            }
            for metric_name, obj in item.get("statistics", {}).items():
                base[f"{metric_name}_value"] = obj.get("value")
            rows.append(base)
        return rows


class ActivationsStatisticsManager(StatisticsManager):
    """Report builder for activation metrics analysis."""

    schema_version: str = "1.0"

    def __init__(self):
        """Initialize the activations statistics manager."""
        super().__init__()
        self._activation_records: Dict[str, Dict[str, Any]] = {}

    def add_activation(
        self,
        layer_name: str,
        activation: Any,
        module_type: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Add an activation tensor and compute its statistics.

        Args:
            layer_name: Name of the layer this activation came from.
            activation: The activation tensor (PyTorch tensor or numpy array).
            module_type: Optional type of the module.
            metrics: List of metrics to compute. If None, uses defaults.
        """
        import torch
        import numpy as np

        default_metrics = ["mean", "std", "min", "max", "l2_norm"]
        metrics_to_use = metrics if metrics is not None else default_metrics

        # Convert to float tensor
        if isinstance(activation, np.ndarray):
            data = torch.from_numpy(activation).float()
        elif isinstance(activation, torch.Tensor):
            data = activation.detach().float()
        else:
            data = torch.tensor(activation, dtype=torch.float32)

        # Flatten if needed
        if data.dim() > 1:
            data_flat = data.reshape(-1)
        else:
            data_flat = data

        # Initialize record if needed
        if layer_name not in self._activation_records:
            self._activation_records[layer_name] = {
                "series": {m: [] for m in metrics_to_use},
                "module_type": module_type or "",
                "sample_count": 0,
                "last_shape": [],
            }

        record = self._activation_records[layer_name]
        record["sample_count"] += 1
        record["last_shape"] = list(data.shape)

        # Compute and store metrics
        if "mean" in metrics_to_use:
            val = float(data_flat.mean().item())
            record["series"]["mean"].append(val)
            self.add(f"{layer_name}.mean", val)
        if "std" in metrics_to_use:
            val = float(data_flat.std().item()) if data_flat.numel() > 1 else 0.0
            record["series"]["std"].append(val)
            self.add(f"{layer_name}.std", val)
        if "min" in metrics_to_use:
            val = float(data_flat.min().item())
            record["series"]["min"].append(val)
            self.add(f"{layer_name}.min", val)
        if "max" in metrics_to_use:
            val = float(data_flat.max().item())
            record["series"]["max"].append(val)
            self.add(f"{layer_name}.max", val)
        if "l2_norm" in metrics_to_use:
            val = float(torch.norm(data_flat, p=2).item())
            record["series"]["l2_norm"].append(val)
            self.add(f"{layer_name}.l2_norm", val)

    def get_activation_records(self) -> List[Dict[str, Any]]:
        """
        Get collected activation records in the format expected by build_report.

        Returns:
            List of activation record dictionaries.
        """
        return [
            {
                "name": name,
                "module_type": rec["module_type"],
                "series": rec["series"],
                "last_shape": rec["last_shape"],
                "sample_count": rec["sample_count"],
            }
            for name, rec in self._activation_records.items()
        ]

    def build_report(
        self,
        model_info: Dict[str, Any],
        raw_results: List[Dict[str, Any]],
        config: Dict[str, Any],
        timings: Dict[str, Any],
    ) -> Dict[str, Any]:
        analysis_cfg = config.get("analysis", {})
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        aggregation_cfg = analysis_cfg.get("aggregation", {})
        compute_cfg = analysis_cfg.get("compute", {})
        lm_eval_cfg = config.get("lm_eval", {})

        reductions = aggregation_cfg.get(
            "reductions", ["mean", "std", "min", "max", "median"]
        )

        per_layer: List[Dict[str, Any]] = []
        metric_names: List[str] = []
        total_samples = 0
        for item in raw_results:
            series: Dict[str, List[float]] = item.get("series", {})
            for k in series.keys():
                metric_names.append(k)
            total_samples += int(item.get("sample_count", 0))
            per_layer.append(
                {
                    "name": item.get("name"),
                    "module_type": item.get("module_type", ""),
                    "last_shape": list(item.get("last_shape", [])),
                    "sample_count": int(item.get("sample_count", 0)),
                    "statistics": self._series_to_statistics(series, reductions, item.get("result_shape_counts", {})),
                    "result_shape_counts": item.get("result_shape_counts", {}),
                }
            )

        metric_names = sorted(set(metric_names))

        report: Dict[str, Any] = {
            "version": self.schema_version,
            "model": {
                "type": model_info.get("type"),
                "parameter_count": model_info.get("parameter_count"),
                "device": model_info.get("device"),
                "is_compressed": model_info.get("is_compressed", False),
            },
            "analysis": {
                "type": "activation_metrics",
                "metrics": metrics_cfg.get("names", []),
                "selection": selection_cfg,
                "aggregation": {"reductions": reductions},
                "compute": compute_cfg,
                "lm_eval": lm_eval_cfg,
            },
            "summary": {
                "layers_analyzed": len(per_layer),
                "samples_analyzed": total_samples,
                "metrics": metric_names,
                "reductions": reductions,
                "runtime_sec": float(timings.get("runtime_sec", 0.0)),
            },
            "per_layer": per_layer,
            "by_module_type": self.aggregate_by_module_type(raw_results, reductions),
            "global": self.aggregate_global(raw_results, reductions),
            "provenance": {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "seed": compute_cfg.get("seed"),
                "framework_versions": {},
                "external_metrics": timings.get("external_metrics", {}),
            },
        }
        return report

    def _series_to_statistics(
        self,
        series: Dict[str, List[float]],
        reductions: List[str],
        shape_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        shape_counts = shape_counts or {}
        suffixes = {"mean", "std", "min", "max", "median", "p25", "p75"}
        for metric_name, values in series.items():
            # sanitize values to floats
            clean: List[float] = []
            for v in values:
                try:
                    clean.append(float(v))
                except Exception:
                    continue
            entry = self._reduce(clean, reductions)
            # attach representative result_shape if available from shape_counts
            base = metric_name
            if "." in base:
                parts = base.rsplit(".", 1)
                if len(parts) == 2 and parts[1] in suffixes:
                    base = parts[0]
            shapes = shape_counts.get(base, {})
            if shapes:
                # choose most frequent shape representation
                shape_str = sorted(shapes.items(), key=lambda kv: -kv[1])[0][0]
                try:
                    # normalize to tuple for readability
                    import ast
                    shp = ast.literal_eval(shape_str)
                    if isinstance(shp, (list, tuple)):
                        entry = {"result_shape": tuple(shp), **entry}
                    else:
                        entry = {"result_shape": shape_str, **entry}
                except Exception:
                    entry = {"result_shape": shape_str, **entry}
            stats[metric_name] = entry
        return stats

    def aggregate_by_module_type(
        self, raw_results: List[Dict[str, Any]], reductions: List[str]
    ) -> Dict[str, Any]:
        groups: Dict[str, List[List[float]]] = {}
        metrics: Dict[str, None] = {}
        # Build metric-wise lists grouped by module type
        grouped_series: Dict[str, Dict[str, List[float]]] = {}
        counts: Dict[str, int] = {}
        total_samples: Dict[str, int] = {}
        for item in raw_results:
            mtype = str(item.get("module_type", ""))
            counts[mtype] = counts.get(mtype, 0) + 1
            total_samples[mtype] = total_samples.get(mtype, 0) + int(item.get("sample_count", 0))
            for metric_name, values in item.get("series", {}).items():
                metrics[metric_name] = None
                grouped_series.setdefault(mtype, {}).setdefault(metric_name, []).extend([float(v) for v in values])

        result: Dict[str, Any] = {}
        for mtype, per_metric in grouped_series.items():
            out_stats: Dict[str, Any] = {}
            for metric_name, values in per_metric.items():
                out_stats[metric_name] = self._reduce(values, reductions)
            result[mtype] = {
                "count": counts.get(mtype, 0),
                "total_samples": total_samples.get(mtype, 0),
                "statistics": out_stats,
            }
        return result

    def aggregate_global(
        self, raw_results: List[Dict[str, Any]], reductions: List[str]
    ) -> Dict[str, Any]:
        # Concatenate all series across layers
        acc: Dict[str, List[float]] = {}
        for item in raw_results:
            for metric_name, values in item.get("series", {}).items():
                acc.setdefault(metric_name, []).extend([float(v) for v in values])
        global_stats: Dict[str, Any] = {}
        for metric_name, values in acc.items():
            global_stats[metric_name] = self._reduce(values, reductions)
        return global_stats

    def extract_csv_rows(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in report.get("per_layer", []):
            base = {
                "name": item.get("name"),
                "module_type": item.get("module_type"),
                "last_shape": str(item.get("last_shape")),
                "sample_count": item.get("sample_count"),
            }
            for metric_name, red in item.get("statistics", {}).items():
                for rname, rvalue in red.items():
                    base[f"{metric_name}_{rname}"] = rvalue
            rows.append(base)
        return rows

    def _reduce(self, values: List[float], reductions: List[str]) -> Dict[str, float]:
        import numpy as _np
        out: Dict[str, float] = {}
        if not values:
            return out
        a = _np.asarray(values, dtype=float)
        if "mean" in reductions:
            out["mean"] = float(_np.mean(a))
        if "std" in reductions:
            out["std"] = float(_np.std(a, ddof=0)) if a.size > 1 else 0.0
        if "min" in reductions:
            out["min"] = float(_np.min(a))
        if "max" in reductions:
            out["max"] = float(_np.max(a))
        if "median" in reductions:
            out["median"] = float(_np.median(a))
        if "p25" in reductions:
            out["p25"] = float(_np.percentile(a, 25))
        if "p75" in reductions:
            out["p75"] = float(_np.percentile(a, 75))
        return out

    def _percentile(self, v: List[float], p: float) -> float:
        if not v:
            return 0.0
        if len(v) == 1:
            return float(v[0])
        k = (len(v) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(v) - 1)
        if f == c:
            return float(v[int(k)])
        d0 = v[f] * (c - k)
        d1 = v[c] * (k - f)
        return float(d0 + d1)
