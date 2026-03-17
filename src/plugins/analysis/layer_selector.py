"""
Layer selection utilities based on activation metrics reports.

Provides a LayerMetricSelector that computes per-block scores from per-layer
statistics in an activation metrics report, then selects blocks to prune.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Number = float


@dataclass(frozen=True)
class BlockScores:
    scores: List[Tuple[int, Number]]  # (block_idx, score), ascending
    block_to_layers: Dict[int, List[str]]


class LayerMetricSelector:
    def __init__(
        self,
        selection_metric_type: str,
        *,
        group_prefix: str = "layers",
        include_names: Optional[Sequence[str]] = None,
        module_types: Optional[Sequence[str]] = None,
        aggregator: str = "mean",  # mean | median | min | max
        weights: str = "uniform",  # uniform | sample_count
    ) -> None:
        self.selection_metric_type = selection_metric_type
        self.group_prefix = group_prefix
        self.include_names = list(include_names) if include_names else None
        self.module_types = list(module_types) if module_types else None
        self.aggregator = aggregator
        self.weights = weights

        metric_base, reducer = self._parse_metric(selection_metric_type)
        self.metric_base = metric_base
        self.reducer = reducer

    @staticmethod
    def _parse_metric(selection_metric_type: str) -> Tuple[str, str]:
        if not (selection_metric_type.endswith(".mean") or selection_metric_type.endswith(".median")):
            raise ValueError(
                "selection_metric_type must end with .mean or .median (e.g., 'compute_lda_matrix.singular_values.max.median')"
            )
        base, reducer = selection_metric_type.rsplit(".", 1)
        return base, reducer

    def _match_filters(self, entry: Mapping[str, Any]) -> bool:
        name: str = entry.get("name", "") or ""
        # Accept group prefix anywhere in the dotted path (supports 'layers' or 'model.layers')
        if f"{self.group_prefix}." not in name:
            return False

        if self.include_names is not None:
            if not any(tok in name for tok in self.include_names):
                return False

        if self.module_types is not None:
            if entry.get("module_type") not in self.module_types:
                return False

        return True

    def _extract_block_index(self, name: str) -> Optional[int]:
        """
        Extract block index from a dotted module name.

        Supports group prefixes like 'layers' or nested like 'model.layers'.
        Finds the last occurrence of '<group_prefix>.' and parses the following
        token as the block index.
        """
        try:
            anchor = f"{self.group_prefix}."
            pos = name.rfind(anchor)
            if pos < 0:
                return None
            rest = name[pos + len(anchor):]
            idx_token = rest.split(".", 1)[0]
            return int(idx_token)
        except Exception:
            return None

    def compute_scores(self, data: Mapping[str, Any]) -> BlockScores:
        per_layer = data.get("per_layer") or []

        # Collect values by block index
        values_by_block: Dict[int, List[Tuple[Number, int, str]]] = {}
        layers_by_block: Dict[int, List[str]] = {}

        for entry in per_layer:
            if not self._match_filters(entry):
                continue

            name = entry.get("name", "") or ""
            block_idx = self._extract_block_index(name)
            if block_idx is None:
                continue

            stats = (entry.get("statistics") or {}).get(self.metric_base)
            if not isinstance(stats, dict):
                continue
            val = stats.get(self.reducer)
            if not isinstance(val, (int, float)):
                continue
            # Skip non-finite values (e.g., NaN/Inf) to avoid contaminating aggregates
            import math
            if not math.isfinite(float(val)):
                continue

            weight = 1
            if self.weights == "sample_count":
                try:
                    weight = int(entry.get("sample_count") or 0) or 1
                except Exception:
                    weight = 1

            values_by_block.setdefault(block_idx, []).append((float(val), weight, name))
            layers_by_block.setdefault(block_idx, []).append(name)

        # Aggregate per block
        scores: List[Tuple[int, Number]] = []
        for b, items in values_by_block.items():
            vals = [v for v, _w, _n in items]
            if not vals:
                continue
            if self.aggregator == "mean":
                if self.weights == "sample_count":
                    wsum = sum(w for _v, w, _n in items)
                    vsum = sum(v * w for v, w, _n in items)
                    score = vsum / wsum if wsum > 0 else float("nan")
                else:
                    score = sum(vals) / len(vals)
            elif self.aggregator == "median":
                score = float(median(vals))
            elif self.aggregator == "min":
                score = min(vals)
            elif self.aggregator == "max":
                score = max(vals)
            else:
                raise ValueError(f"Unsupported aggregator: {self.aggregator}")

            scores.append((b, float(score)))

        scores.sort(key=lambda x: x[1])  # ascending
        return BlockScores(scores=scores, block_to_layers=layers_by_block)

    @staticmethod
    def choose_to_prune(
        scores: Sequence[Tuple[int, Number]],
        *,
        num_deleted_block: Optional[int] = None,
        ratio_deleted_block: Optional[float] = None,
        min_keep: int = 1,
    ) -> List[int]:
        if (num_deleted_block is None) == (ratio_deleted_block is None):
            raise ValueError("Provide exactly one of num_deleted_block or ratio_deleted_block")

        B = len(scores)
        if B <= 0:
            return []

        if ratio_deleted_block is not None:
            if ratio_deleted_block < 0:
                raise ValueError("ratio_deleted_block must be >= 0")
            k = int(ratio_deleted_block * B)
        else:
            if num_deleted_block < 0:
                raise ValueError("num_deleted_block must be >= 0")
            k = int(num_deleted_block)

        k = max(0, min(k, max(0, B - max(1, min_keep))))
        return [b for b, _s in list(scores)[:k]]
