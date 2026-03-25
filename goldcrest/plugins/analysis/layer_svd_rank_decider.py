"""
LayerSVDRankDecider

Utility to decide which linear layers to compress with SVD and assign ranks
based on activation statistics reports. Integrates with LayerMetricSelector for
block scoring and supports simple stable-rank heuristics, optional module
weighting, and optional global budget scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - allows import in environments without torch
    torch = None
    nn = None

from .layer_selector import LayerMetricSelector


Number = float


@dataclass
class LayerTarget:
    module_path: str  # bracket path e.g., model.layers[0].mlp.up_proj
    in_features: int
    out_features: int
    block_idx: int


class LayerSVDRankDecider:
    """Compute SVD ranks per linear layer from activation reports.

    Minimal scaffold implementing:
      - Block scoring via LayerMetricSelector
      - Selection (compress_all or choose bottom-k/ratio)
      - Stable-rank heuristic per block and block-dependent scaling
      - Optional module multipliers and basic budget scaling
    """

    def __init__(
        self,
        cfg: Optional[Mapping[str, Any]] = None,
        *,
        # Keyword argument aliases for convenience
        rank_strategy: Optional[str] = None,
        min_rank: Optional[int] = None,
        target_param_ratio: Optional[float] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        energy_threshold: Optional[float] = None,
        selection_metric_type: Optional[str] = None,
        group_prefix: Optional[str] = None,
        include_names: Optional[List[str]] = None,
        module_types: Optional[List[str]] = None,
        aggregator: Optional[str] = None,
        weights: Optional[str] = None,
        compress_all: Optional[bool] = None,
        num_deleted_block: Optional[int] = None,
        ratio_deleted_block: Optional[float] = None,
        min_keep: Optional[int] = None,
        module_multipliers: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        # Start with cfg dict or empty
        self.cfg = dict(cfg or {})

        # Merge keyword arguments (they take precedence)
        kwarg_map = {
            "rank_strategy": rank_strategy,
            "min_rank": min_rank,
            "target_param_ratio": target_param_ratio,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "energy_threshold": energy_threshold,
            "selection_metric_type": selection_metric_type,
            "group_prefix": group_prefix,
            "include_names": include_names,
            "module_types": module_types,
            "aggregator": aggregator,
            "weights": weights,
            "compress_all": compress_all,
            "num_deleted_block": num_deleted_block,
            "ratio_deleted_block": ratio_deleted_block,
            "min_keep": min_keep,
            "module_multipliers": module_multipliers,
        }
        for key, value in kwarg_map.items():
            if value is not None:
                self.cfg[key] = value

        # Also merge any extra kwargs
        for key, value in kwargs.items():
            if value is not None:
                self.cfg[key] = value

    # ------------------------
    # Public entry point
    # ------------------------
    def decide(self, model: Any, report: Mapping[str, Any]) -> Dict[str, Any]:
        scores = self._compute_block_scores(report)
        selected_blocks = self._select_blocks(scores)
        targets = self._enumerate_block_linears(model, selected_blocks)

        strategy = str(self.cfg.get("rank_strategy", "stable_rank")).lower()
        result: Dict[str, Any] = {"block_scores": scores}

        if strategy == "energy":
            energy_map = self._assign_energy_thresholds(report, scores, targets)
            result["energy_thresholds"] = energy_map
            # Budget scaling requires concrete k; skip for energy strategy
            summary = self._summarize_energy(energy_map, targets)
            result["summary"] = summary
            return result

        # default: stable-rank based integer ranks
        layer_ranks = self._assign_ranks(report, scores, targets)
        layer_ranks = self._apply_budget_if_needed(layer_ranks, targets)
        summary = self._summarize(layer_ranks, targets)
        result.update({
            "layer_ranks": layer_ranks,
            "summary": summary,
        })
        return result

    # ------------------------
    # Block scoring and selection
    # ------------------------
    def _compute_block_scores(self, report: Mapping[str, Any]) -> List[Tuple[int, Number]]:
        smt = str(self.cfg.get("selection_metric_type", "l2_norm.median"))
        selector = LayerMetricSelector(
            selection_metric_type=smt,
            group_prefix=str(self.cfg.get("group_prefix", "layers")),
            include_names=self.cfg.get("include_names"),
            module_types=self.cfg.get("module_types"),
            aggregator=str(self.cfg.get("aggregator", "mean")),
            weights=str(self.cfg.get("weights", "uniform")),
        )
        bs = selector.compute_scores(report)
        scores = list(bs.scores)
        if scores:
            return scores
        # Relax and fallback like pruning plugin
        relaxed = LayerMetricSelector(
            selection_metric_type=smt,
            group_prefix=str(self.cfg.get("group_prefix", "layers")),
            include_names=None,
            module_types=None,
            aggregator=str(self.cfg.get("aggregator", "mean")),
            weights=str(self.cfg.get("weights", "uniform")),
        )
        rel = relaxed.compute_scores(report)
        if rel.scores:
            return list(rel.scores)
        # Final fallback key
        fallback_metric = "l2_norm.median"
        fallback = LayerMetricSelector(
            selection_metric_type=fallback_metric,
            group_prefix=str(self.cfg.get("group_prefix", "layers")),
            include_names=None,
            module_types=None,
            aggregator=str(self.cfg.get("aggregator", "mean")),
            weights=str(self.cfg.get("weights", "uniform")),
        )
        return list(fallback.compute_scores(report).scores)

    def _select_blocks(self, scores: Sequence[Tuple[int, Number]]) -> List[int]:
        if not scores:
            return []
        if bool(self.cfg.get("compress_all", True)):
            return [b for b, _ in scores]
        return LayerMetricSelector.choose_to_prune(
            scores,
            num_deleted_block=self.cfg.get("num_deleted_block"),
            ratio_deleted_block=self.cfg.get("ratio_deleted_block"),
            min_keep=int(self.cfg.get("min_keep", 1)),
        )

    # ------------------------
    # Target enumeration
    # ------------------------
    def _enumerate_block_linears(self, model: Any, blocks: Iterable[int]) -> List[LayerTarget]:
        """Find linear-like modules under each selected block.

        Strategy: iterate named_modules, pick modules whose path contains
        ".layers.<idx>." and either looks like a Linear (has weight attribute
        that is 2D) or torch.nn.Linear instance. Filter to attention/MLP paths
        by include_names if provided; otherwise accept any 2D weight module.
        """
        if model is None:
            return []
        include_names = list(self.cfg.get("include_names") or [])
        group_prefix = str(self.cfg.get("group_prefix", "layers"))
        blocks_set = set(int(b) for b in blocks)
        out: List[LayerTarget] = []

        def to_bracket_path(name: str) -> str:
            # convert 'model.layers.3.mlp.up_proj' -> 'model.layers[3].mlp.up_proj'
            parts = name.split(".")
            res: List[str] = []
            i = 0
            while i < len(parts):
                res.append(parts[i])
                if parts[i] == group_prefix and i + 1 < len(parts):
                    # next part is index token
                    try:
                        idx = int(parts[i + 1])
                        res[-1] = f"{parts[i]}[{idx}]"
                        i += 1  # skip the numeric token we consumed
                    except Exception:
                        pass
                i += 1
            return ".".join(res)

        for name, module in getattr(model, "named_modules", lambda: [])():
            if not name:
                continue
            # Accept only modules under group_prefix.<idx>
            anchor = f".{group_prefix}."
            if anchor not in name:
                continue
            # Extract block index from dotted path
            try:
                after = name.split(anchor, 1)[1]
                tok = after.split(".", 1)[0]
                idx = int(tok)
            except Exception:
                continue
            if idx not in blocks_set:
                continue
            # include_names filter (like 'mlp', 'self_attn')
            if include_names and not any(key in name for key in include_names):
                continue

            # Identify linear-like modules
            in_f = out_f = None
            is_linearish = False
            if nn is not None and isinstance(module, nn.Linear):
                in_f, out_f = int(module.in_features), int(module.out_features)
                is_linearish = True
            else:
                w = getattr(module, "weight", None)
                if w is not None:
                    try:
                        shape = tuple(int(x) for x in getattr(w, "shape", ()) or [])
                        if len(shape) == 2:
                            out_f, in_f = shape[0], shape[1]
                            is_linearish = True
                    except Exception:
                        pass
            if not is_linearish:
                continue

            out.append(LayerTarget(module_path=to_bracket_path(name), in_features=in_f, out_features=out_f, block_idx=idx))

        return out

    # ------------------------
    # Rank assignment
    # ------------------------
    def _assign_ranks(self, report: Mapping[str, Any], scores: Sequence[Tuple[int, Number]], targets: Sequence[LayerTarget]) -> Dict[str, int]:
        if not targets:
            return {}
        min_rank = int(self.cfg.get("min_rank", 8))
        alpha_min = float(self.cfg.get("alpha_min", 0.5))
        alpha_max = float(self.cfg.get("alpha_max", 1.0))
        multipliers: Dict[str, float] = dict(self.cfg.get("module_multipliers", {}))
        score_min = scores[0][1] if scores else 0.0
        score_max = scores[-1][1] if scores else 1.0
        denom = (score_max - score_min) if (score_max > score_min) else 1.0

        # Pre-index block scores
        score_map: Dict[int, float] = {b: float(s) for b, s in scores}
        ranks: Dict[str, int] = {}
        for t in targets:
            m, n = int(t.in_features), int(t.out_features)
            k0 = self._base_rank_from_stats(report, t.block_idx, m, n, min_rank)
            s = score_map.get(t.block_idx, score_min)
            s_hat = (s - score_min) / denom
            alpha = alpha_min + (1.0 - s_hat) * (alpha_max - alpha_min)
            k = int(round(alpha * k0))
            # module-specific pattern multipliers
            for pat, mul in multipliers.items():
                if pat and pat in t.module_path:
                    k = int(round(float(mul) * k))
            k = max(min_rank, min(k, min(m, n)))
            ranks[t.module_path] = int(k)
        return ranks

    def _assign_energy_thresholds(self, report: Mapping[str, Any], scores: Sequence[Tuple[int, Number]], targets: Sequence[LayerTarget]) -> Dict[str, float]:
        """Assign per-layer energy preservation thresholds based on block scores.

        Uses a base threshold (energy_threshold) and optionally scales it within
        [energy_min, energy_max] by normalized block score (s_hat).
        """
        if not targets:
            return {}
        # Base threshold and optional bounds
        base_tau = float(self.cfg.get("energy_threshold", 0.9))
        e_min = float(self.cfg.get("energy_min", base_tau))
        e_max = float(self.cfg.get("energy_max", base_tau))

        # Normalize scores to [0,1]
        score_min = scores[0][1] if scores else 0.0
        score_max = scores[-1][1] if scores else 1.0
        denom = (score_max - score_min) if (score_max > score_min) else 1.0
        score_map: Dict[int, float] = {b: float(s) for b, s in scores}

        # Compute energy threshold per target
        energy_map: Dict[str, float] = {}
        for t in targets:
            s = score_map.get(t.block_idx, score_min)
            s_hat = (s - score_min) / denom
            # Mirror stable-rank alpha scaling: lower score → closer to e_max
            tau = e_min + (1.0 - s_hat) * (e_max - e_min)
            # Clamp to (0,1)
            tau = float(max(0.5, min(0.999, tau)))
            energy_map[t.module_path] = tau
        return energy_map

    def _base_rank_from_stats(self, report: Mapping[str, Any], block_idx: int, m: int, n: int, min_rank: int) -> int:
        # Stable rank heuristic using LDA stats aggregated per block
        base_key = "compute_lda_matrix"
        frob_key = f"{base_key}.frobenius_norm"
        spec_key = f"{base_key}.spectral_norm"
        reducer = str(self.cfg.get("reducer", "median"))  # "mean" or "median"
        fr_vals: List[float] = []
        sp_vals: List[float] = []

        per_layer = report.get("per_layer") or []
        anchor = f"layers.{block_idx}."
        for entry in per_layer:
            name = entry.get("name", "") or ""
            if anchor not in name:
                continue
            stats = entry.get("statistics", {}) or {}
            fr = self._safe_float((stats.get(frob_key) or {}).get(reducer))
            sp = self._safe_float((stats.get(spec_key) or {}).get(reducer))
            if fr is not None:
                fr_vals.append(fr)
            if sp is not None and sp > 0:
                sp_vals.append(sp)

        if fr_vals and sp_vals:
            fr_v = float(self._reduce(fr_vals, reducer))
            sp_v = float(self._reduce(sp_vals, reducer))
            try:
                r_stable = float((fr_v / sp_v) ** 2)
            except Exception:
                r_stable = float(min(m, n))
        else:
            # fallback: mid fraction of min(m, n)
            r_stable = 0.5 * float(min(m, n))

        k0 = int(round(r_stable))
        k0 = max(min_rank, min(k0, min(m, n)))
        return k0

    # ------------------------
    # Budget and summary
    # ------------------------
    def _apply_budget_if_needed(self, layer_ranks: Dict[str, int], targets: Sequence[LayerTarget]) -> Dict[str, int]:
        budget_ratio = self.cfg.get("target_param_ratio")
        if budget_ratio is None or not targets or not layer_ranks:
            return layer_ranks
        # Compute original and compressed params
        total_orig = 0
        total_comp = 0
        per_key: Dict[str, Tuple[int, int, int]] = {}  # path -> (m,n,k)
        tmap = {t.module_path: t for t in targets}
        for path, k in layer_ranks.items():
            t = tmap.get(path)
            if not t:
                continue
            m, n = int(t.out_features), int(t.in_features)
            total_orig += m * n
            total_comp += k * (m + n + 1)
            per_key[path] = (m, n, k)
        if total_comp <= 0 or total_orig <= 0:
            return layer_ranks
        # Desired compressed params
        desired = max(1, int(budget_ratio * total_orig))
        if total_comp <= desired:
            return layer_ranks
        # Scale all k by common gamma
        gamma = float(desired) / float(total_comp)
        min_rank = int(self.cfg.get("min_rank", 8))
        out: Dict[str, int] = {}
        for path, (m, n, k) in per_key.items():
            new_k = int(round(gamma * k))
            new_k = max(min_rank, min(new_k, min(m, n)))
            out[path] = new_k
        return out

    def _summarize(self, layer_ranks: Mapping[str, int], targets: Sequence[LayerTarget]) -> Dict[str, Any]:
        tmap = {t.module_path: t for t in targets}
        total_layers = len(layer_ranks)
        orig = comp = 0
        for path, k in layer_ranks.items():
            t = tmap.get(path)
            if not t:
                continue
            m, n = int(t.out_features), int(t.in_features)
            orig += m * n
            comp += k * (m + n + 1)
        ratio = (float(orig) / float(comp)) if comp > 0 else 0.0
        return {
            "layers": total_layers,
            "total_original_params": int(orig),
            "total_compressed_params": int(comp),
            "overall_compression_ratio": float(ratio),
        }

    def _summarize_energy(self, energy_thresholds: Mapping[str, float], targets: Sequence[LayerTarget]) -> Dict[str, Any]:
        """Lightweight summary for energy strategy (no exact param counts prior to SVD)."""
        total_layers = len(energy_thresholds)
        avg_energy = 0.0
        if energy_thresholds:
            avg_energy = float(sum(energy_thresholds.values()) / len(energy_thresholds))
        return {
            "layers": total_layers,
            "average_preserve_energy": avg_energy,
        }

    # ------------------------
    # Small helpers
    # ------------------------
    def _safe_float(self, v: Any) -> Optional[float]:
        try:
            if isinstance(v, (int, float)):
                return float(v)
            if hasattr(v, "item") and callable(getattr(v, "item")):
                val = v.item()
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception:
            return None
        return None

    def _reduce(self, values: Sequence[float], reducer: str) -> float:
        if not values:
            return 0.0
        if reducer == "mean":
            return float(sum(values) / len(values))
        if reducer == "min":
            return float(min(values))
        if reducer == "max":
            return float(max(values))
        # default median
        v = sorted(values)
        n = len(v)
        mid = n // 2
        if n % 2 == 1:
            return float(v[mid])
        return float(0.5 * (v[mid - 1] + v[mid]))
