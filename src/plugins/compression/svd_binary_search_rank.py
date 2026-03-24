"""
ASVD Binary Search Rank Allocation Plugin.

Finds optimal per-layer ranks via binary search to meet target PPL or parameter ratio.
"""

import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from ...framework.plugins import Plugin, PluginMetadata


class BinarySearchRankPlugin(Plugin):
    """
    ASVD-style binary search for optimal per-layer rank allocation.

    Given a target (PPL threshold or parameter ratio), searches for
    the optimal rank for each layer that meets the target.

    Mathematical Background:
        Given target parameter ratio tau, find ranks {k_l} such that:
            sum_l k_l * (m_l + n_l + 1) <= tau * sum_l m_l * n_l

        Ranks are allocated based on layer sensitivity scores.

    Reads from:
        - svd.sensitivity.ppl.<layer_name> (from PPLSensitivityPlugin)
        - OR uses stable_rank heuristic from activation analysis

    Writes to:
        - svd.ranks.<layer_name>: Optimal rank for each layer

    Usage:
        plugin = BinarySearchRankPlugin(
            target_mode="param_ratio",
            param_ratio_target=0.9
        )
        plugin.initialize(context)
        result = plugin.execute(model=model)
    """

    def __init__(
        self,
        target_mode: str = "param_ratio",
        ppl_target: Optional[float] = None,
        param_ratio_target: float = 0.9,
        sensitivity_source: str = "stable_rank",
        min_rank: int = 8,
        **kwargs
    ):
        """
        Initialize BinarySearchRankPlugin.

        Args:
            target_mode: "ppl" or "param_ratio" - determines optimization target
            ppl_target: Target PPL threshold (when target_mode="ppl")
            param_ratio_target: Target parameter ratio (when target_mode="param_ratio")
            sensitivity_source: "ppl" or "stable_rank" - source of sensitivity scores
            min_rank: Minimum rank to preserve for any layer
        """
        super().__init__(**kwargs)
        self.target_mode = target_mode
        self.ppl_target = ppl_target
        self.param_ratio_target = param_ratio_target
        self.sensitivity_source = sensitivity_source
        self.min_rank = min_rank

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="ASVD binary search rank allocation",
            category="compression"
        )

    def do_execute(
        self,
        model: nn.Module = None,
        target_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Allocate optimal ranks to each layer.

        Args:
            model: The model to allocate ranks for
            target_layers: Optional list of layer name patterns to process

        Returns:
            Dict with ranks, total_original_params, and total_compressed_params
        """
        if model is None:
            model = self.get_model("current")
        if model is None:
            raise ValueError("BinarySearchRankPlugin requires a model")

        self.emit_event("binary_search_rank.started", {
            "target_mode": self.target_mode,
            "param_ratio_target": self.param_ratio_target,
        })

        # Collect layer info
        layer_info: Dict[str, Dict[str, int]] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers and not any(t in name for t in target_layers):
                    continue
                m, n = module.weight.shape
                layer_info[name] = {"m": m, "n": n, "params": m * n}

        total_params = sum(info["params"] for info in layer_info.values())

        ppl_sensitivity_map = self._get_ppl_sensitivity_map(layer_info)

        # Allocate ranks based on target mode and sensitivity source
        if self.sensitivity_source == "ppl" and ppl_sensitivity_map:
            if self.target_mode == "param_ratio":
                ranks = self._allocate_by_param_ratio_from_ppl_map(
                    layer_info,
                    ppl_sensitivity_map,
                    total_params,
                )
            else:
                ranks = self._allocate_by_ppl_from_ppl_map(layer_info, ppl_sensitivity_map)
        else:
            sensitivity_scores = self._get_sensitivity_scores(layer_info)
            if self.target_mode == "param_ratio":
                ranks = self._allocate_by_param_ratio(layer_info, sensitivity_scores, total_params)
            else:
                ranks = self._allocate_by_ppl(layer_info, sensitivity_scores)

        # Store in StateManager
        if self.state_manager:
            for name, rank in ranks.items():
                self.state_manager.state.set(f"svd.ranks.{name}", rank)

        # Calculate compressed params
        total_compressed_params = sum(
            ranks[n] * (layer_info[n]["m"] + layer_info[n]["n"] + 1)
            for n in ranks
        )

        self.emit_event("binary_search_rank.completed", {"layers": len(ranks)})

        return {
            "ranks": ranks,
            "total_original_params": total_params,
            "total_compressed_params": total_compressed_params,
            "compression_ratio": total_params / max(1, total_compressed_params),
        }

    def _get_ppl_sensitivity_map(self, layer_info: Dict[str, Dict[str, int]]) -> Dict[str, Dict[float, float]]:
        """
        Read the full per-layer PPL sensitivity surface from state.

        This preserves the layer/ratio candidate structure needed for a
        closer-to-upstream binary search instead of collapsing everything
        to a single scalar sensitivity score.
        """
        if not self.state_manager:
            return {}

        sensitivity_map: Dict[str, Dict[float, float]] = {}
        for name in layer_info:
            raw_map = self.state_manager.state.get(f"svd.sensitivity.ppl.{name}")
            if not isinstance(raw_map, dict):
                continue

            normalized: Dict[float, float] = {}
            for ratio, score in raw_map.items():
                try:
                    ratio_f = float(ratio)
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue

                if ratio_f <= 0.0 or math.isnan(ratio_f) or math.isnan(score_f):
                    continue

                normalized[ratio_f] = score_f

            if normalized:
                sensitivity_map[name] = dict(sorted(normalized.items()))

        return sensitivity_map

    def _get_sensitivity_scores(self, layer_info: Dict) -> Dict[str, float]:
        """
        Get sensitivity scores from StateManager or compute stable rank.

        Args:
            layer_info: Dict mapping layer names to dimension info

        Returns:
            Dict mapping layer names to sensitivity scores
        """
        scores = {}

        # Use stable rank heuristic
        for name in layer_info:
            m, n = layer_info[name]["m"], layer_info[name]["n"]
            # Stable rank approximation: use 50% of min dimension
            scores[name] = 0.5 * min(m, n)

        return scores

    def _rank_from_ratio(self, layer_dims: Dict[str, int], ratio: float) -> int:
        """
        Convert a probe ratio back into the rank convention used by Toggle's SVD plugin.
        """
        min_dim = min(layer_dims["m"], layer_dims["n"])
        raw_rank = int(ratio * min_dim)
        return min(min_dim, max(self.min_rank, raw_rank))

    def _compressed_param_count(self, layer_dims: Dict[str, int], rank: int) -> int:
        return rank * (layer_dims["m"] + layer_dims["n"] + 1)

    def _build_ppl_candidate_list(
        self,
        sensitivity_map: Dict[str, Dict[float, float]],
    ) -> List[tuple[str, float, float]]:
        def sort_score(score: float) -> float:
            if math.isinf(score):
                return float("-inf") if score > 0 else float("inf")
            return -score

        candidates: List[tuple[str, float, float]] = []
        for layer_name, ratio_map in sensitivity_map.items():
            for ratio, score in ratio_map.items():
                candidates.append((layer_name, ratio, score))

        candidates.sort(
            key=lambda item: (
                sort_score(item[2]),
                item[1],
                item[0],
            )
        )
        return candidates

    def _ranks_from_candidate_threshold(
        self,
        layer_info: Dict[str, Dict[str, int]],
        candidates: List[tuple[str, float, float]],
        start_idx: int,
    ) -> Dict[str, int]:
        """
        Apply the upstream-style candidate-threshold idea:
        tuples before the threshold are treated as too sensitive to compress.
        """
        selected_ratios = {name: 1.0 for name in layer_info}
        for layer_name, ratio, _score in candidates[start_idx:]:
            if layer_name in selected_ratios:
                selected_ratios[layer_name] = min(selected_ratios[layer_name], ratio)

        return {
            name: self._rank_from_ratio(layer_info[name], selected_ratios[name])
            for name in layer_info
        }

    def _allocate_by_param_ratio_from_ppl_map(
        self,
        layer_info: Dict[str, Dict[str, int]],
        sensitivity_map: Dict[str, Dict[float, float]],
        total_params: int,
    ) -> Dict[str, int]:
        """
        Allocate ranks from the full per-layer PPL surface using a global threshold
        over (layer, ratio, score) tuples, which is closer to the upstream search.
        """
        candidates = self._build_ppl_candidate_list(sensitivity_map)
        if not candidates:
            return self._allocate_by_param_ratio(
                layer_info,
                self._get_sensitivity_scores(layer_info),
                total_params,
            )

        target_params = self.param_ratio_target * total_params
        best_idx = 0
        lo = 0
        hi = len(candidates) - 1

        while lo <= hi:
            mid = (lo + hi) // 2
            ranks = self._ranks_from_candidate_threshold(layer_info, candidates, mid)
            compressed_params = sum(
                self._compressed_param_count(layer_info[name], rank)
                for name, rank in ranks.items()
            )

            if compressed_params <= target_params:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return self._ranks_from_candidate_threshold(layer_info, candidates, best_idx)

    def _allocate_by_ppl_from_ppl_map(
        self,
        layer_info: Dict[str, Dict[str, int]],
        sensitivity_map: Dict[str, Dict[float, float]],
    ) -> Dict[str, int]:
        """
        Choose the most compressed ratio per layer whose measured PPL delta
        remains below the requested threshold.
        """
        if self.ppl_target is None:
            raise ValueError("ppl_target must be set when target_mode='ppl'")

        ranks: Dict[str, int] = {}
        for name, dims in layer_info.items():
            ratio_map = sensitivity_map.get(name) or {}
            chosen_ratio = 1.0
            for ratio, score in sorted(ratio_map.items(), key=lambda item: item[0]):
                if score < self.ppl_target:
                    chosen_ratio = ratio
                    break
            ranks[name] = self._rank_from_ratio(dims, chosen_ratio)

        return ranks

    def _allocate_by_param_ratio(
        self,
        layer_info: Dict,
        sensitivity: Dict[str, float],
        total_params: int
    ) -> Dict[str, int]:
        """
        Allocate ranks to achieve target parameter ratio.

        Uses binary search to find scaling factor that achieves target.

        Args:
            layer_info: Dict mapping layer names to dimension info
            sensitivity: Dict mapping layer names to sensitivity scores
            total_params: Total original parameter count

        Returns:
            Dict mapping layer names to allocated ranks
        """
        target_params = int(self.param_ratio_target * total_params)

        # Sort layers by sensitivity (less sensitive = can compress more)
        sorted_layers = sorted(sensitivity.keys(), key=lambda x: sensitivity[x])

        # Binary search for scaling factor
        lo, hi = 0.1, 2.0

        for _ in range(20):  # Max iterations
            mid = (lo + hi) / 2

            ranks = {}
            compressed_params = 0

            for name in sorted_layers:
                m, n = layer_info[name]["m"], layer_info[name]["n"]
                # Scale rank by sensitivity
                base_rank = sensitivity[name] * mid
                k = max(self.min_rank, min(int(base_rank), min(m, n)))
                ranks[name] = k
                compressed_params += k * (m + n + 1)

            if compressed_params < target_params:
                lo = mid
            else:
                hi = mid

        return ranks

    def _allocate_by_ppl(
        self,
        layer_info: Dict,
        sensitivity: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Allocate ranks to meet PPL target.

        Uses per-layer PPL sensitivity to find minimum rank that keeps
        PPL delta below threshold.

        Args:
            layer_info: Dict mapping layer names to dimension info
            sensitivity: Dict mapping layer names to sensitivity scores

        Returns:
            Dict mapping layer names to allocated ranks
        """
        if self.ppl_target is None:
            raise ValueError("ppl_target must be set when target_mode='ppl'")

        ranks = {}

        for name in layer_info:
            m, n = layer_info[name]["m"], layer_info[name]["n"]
            sens = sensitivity.get(name, 1.0)

            # Higher sensitivity = need higher rank to preserve accuracy
            # This is a heuristic; PPL sensitivity gives actual PPL deltas
            if self.state_manager:
                ppl_sens = self.state_manager.state.get(f"svd.sensitivity.ppl.{name}")
                if ppl_sens:
                    # Find ratio where PPL delta < threshold
                    for ratio in sorted(ppl_sens.keys(), reverse=True):
                        if ppl_sens[ratio] < self.ppl_target:
                            k = max(self.min_rank, int(ratio * min(m, n)))
                            ranks[name] = k
                            break
                    else:
                        ranks[name] = min(m, n)  # Full rank
                    continue

            # Fallback to stable rank
            k = max(self.min_rank, int(0.5 * min(m, n)))
            ranks[name] = k

        return ranks

    def allocate_ranks(
        self,
        layer_dims: Dict[str, tuple],
        sensitivity_scores: Dict[str, float],
        target_ratio: float,
        min_rank: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Allocate ranks to achieve target parameter ratio.

        This is a standalone utility method that can be called without
        initializing the full plugin context.

        Args:
            layer_dims: Dict mapping layer names to (m, n) tuples (out_features, in_features)
            sensitivity_scores: Dict mapping layer names to sensitivity scores
            target_ratio: Target parameter ratio (0 < ratio <= 1)
            min_rank: Minimum rank to preserve (defaults to self.min_rank)

        Returns:
            Dict mapping layer names to allocated ranks
        """
        if min_rank is None:
            min_rank = self.min_rank

        # Build layer_info from layer_dims
        layer_info: Dict[str, Dict[str, int]] = {}
        for name, dims in layer_dims.items():
            m, n = dims
            layer_info[name] = {"m": m, "n": n, "params": m * n}

        total_params = sum(info["params"] for info in layer_info.values())
        target_params = int(target_ratio * total_params)

        # Sort layers by sensitivity (less sensitive = can compress more)
        sorted_layers = sorted(sensitivity_scores.keys(), key=lambda x: sensitivity_scores.get(x, 1.0))

        # Binary search for scaling factor
        lo, hi = 0.1, 2.0

        for _ in range(20):  # Max iterations
            mid = (lo + hi) / 2

            ranks: Dict[str, int] = {}
            compressed_params = 0

            for name in sorted_layers:
                if name not in layer_info:
                    continue
                m, n = layer_info[name]["m"], layer_info[name]["n"]
                # Scale rank by sensitivity
                sens = sensitivity_scores.get(name, 0.5 * min(m, n))
                base_rank = sens * mid
                k = max(min_rank, min(int(base_rank), min(m, n)))
                ranks[name] = k
                compressed_params += k * (m + n + 1)

            if compressed_params < target_params:
                lo = mid
            else:
                hi = mid

        return ranks
