"""
Report loader utilities for activation metrics analysis.

This module provides helpers to:
- Discover activation-metrics report JSON files under a root directory
- Load an individual report JSON
- Parse a concise summary from the report
- Flatten per-layer statistics into row-like dictionaries

Designed for reuse by analysis/pruning tools and tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


logger = logging.getLogger(__name__)


# -----------------------------
# Discovery and file loading
# -----------------------------

def find_reports(
    root: Path | str = Path("test_logs/analysis/activation_metrics"),
    pattern: str = "report_*.json",
) -> Iterator[Path]:
    """Yield paths to activation-metrics report JSONs under a root.

    - Searches recursively for files matching `pattern`.
    - Yields in sorted order for stable output.
    """
    root_path = Path(root)
    if not root_path.exists():
        logger.warning("Reports root does not exist: %s", root_path)
        return iter(())

    files = sorted(root_path.rglob(pattern))
    for p in files:
        if p.is_file():
            yield p


def load_report(path: Path | str) -> Dict[str, Any]:
    """Load a single report JSON file into a dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Parsing helpers and dataclasses
# -----------------------------


@dataclass(frozen=True)
class ModelInfo:
    type: str
    parameter_count: Optional[int] = None
    device: Optional[str] = None
    is_compressed: Optional[bool] = None


@dataclass(frozen=True)
class AnalysisInfo:
    type: str
    metrics: Any
    task: Optional[str]
    reductions: Tuple[str, ...]


@dataclass(frozen=True)
class SummaryInfo:
    layers_analyzed: int
    samples_analyzed: int
    metrics: Tuple[str, ...]
    reductions: Tuple[str, ...]
    runtime_sec: float


@dataclass(frozen=True)
class ActivationReportSummary:
    path: Path
    created_from_filename: Optional[str]
    model_type: str
    model: ModelInfo
    analysis: AnalysisInfo
    summary: SummaryInfo
    per_layer_count: int


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_model(model: Dict[str, Any]) -> ModelInfo:
    return ModelInfo(
        type=str(model.get("type", "")),
        parameter_count=model.get("parameter_count"),
        device=model.get("device"),
        is_compressed=model.get("is_compressed"),
    )


def _parse_analysis(analysis: Dict[str, Any]) -> AnalysisInfo:
    reductions = tuple(_safe_get(analysis, "aggregation", "reductions", default=[]) or [])
    task = _safe_get(analysis, "lm_eval", "task")
    return AnalysisInfo(
        type=str(analysis.get("type", "")),
        metrics=analysis.get("metrics"),
        task=str(task) if task is not None else None,
        reductions=reductions,
    )


def _parse_summary(summary: Dict[str, Any]) -> SummaryInfo:
    return SummaryInfo(
        layers_analyzed=int(summary.get("layers_analyzed", 0)),
        samples_analyzed=int(summary.get("samples_analyzed", 0)),
        metrics=tuple(summary.get("metrics", []) or ()),
        reductions=tuple(summary.get("reductions", []) or ()),
        runtime_sec=float(summary.get("runtime_sec", 0.0)),
    )


def parse_activation_report(data: Dict[str, Any], path: Path | str | None = None) -> ActivationReportSummary:
    """Parse a raw activation-metrics report dict into a concise summary."""
    path_obj = Path(path) if path is not None else Path("")
    filename = path_obj.name or None

    model = _parse_model(data.get("model", {}))
    analysis = _parse_analysis(data.get("analysis", {}))
    summary = _parse_summary(data.get("summary", {}))

    per_layer_count = 0
    try:
        per_layer = data.get("per_layer")
        if isinstance(per_layer, list):
            per_layer_count = len(per_layer)
    except Exception:
        per_layer_count = 0

    return ActivationReportSummary(
        path=path_obj,
        created_from_filename=filename,
        model_type=model.type,
        model=model,
        analysis=analysis,
        summary=summary,
        per_layer_count=per_layer_count,
    )


def flatten_layer_rows(
    data: Dict[str, Any], select_metrics: Iterable[str] | None = None
) -> List[Dict[str, Any]]:
    """Flatten per-layer statistics into row dicts.

    Each row contains:
      - name, module_type, sample_count, last_shape
      - for each metric, two columns with suffixes ".mean" and ".median" if present
    """
    if select_metrics is None:
        select_metrics = tuple((data.get("summary") or {}).get("metrics") or ())

    per_layer = data.get("per_layer") or []
    rows: List[Dict[str, Any]] = []
    for entry in per_layer:
        row: Dict[str, Any] = {
            "name": entry.get("name"),
            "module_type": entry.get("module_type"),
            "sample_count": entry.get("sample_count"),
            "last_shape": entry.get("last_shape"),
        }
        stats = entry.get("statistics", {}) or {}
        for key in select_metrics:
            val = stats.get(key)
            mean_key = f"{key}.mean"
            median_key = f"{key}.median"
            if isinstance(val, dict):
                row[mean_key] = val.get("mean")
                row[median_key] = val.get("median")
            else:
                row[mean_key] = None
                row[median_key] = None
        rows.append(row)
    return rows


def _print_summaries(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            data = load_report(p)
            summary = parse_activation_report(data, path=p)
            task = summary.analysis.task or "-"
            print(
                f"{p}: model={summary.model_type}, task={task}, "
                f"layers={summary.summary.layers_analyzed}, per_layer={summary.per_layer_count}, "
                f"metrics={len(summary.summary.metrics)}, runtime={summary.summary.runtime_sec:.2f}s"
            )
        except Exception as e:
            logger.exception("Failed to parse %s: %s", p, e)


# if __name__ == "__main__":
#     _print_summaries(find_reports())

