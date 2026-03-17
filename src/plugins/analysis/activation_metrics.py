"""
Activation metrics analysis plugin and analyzer.

Captures activations (forward outputs) from selected layers while the model
is evaluated, computes configured metrics per activation, and aggregates
statistics per layer for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import time

import torch

from ...framework.plugins import Plugin, PluginMetadata
from ...framework.statistics import ActivationsStatisticsManager
from ...framework.model_manager import ModelManager
from ...framework.reproducibility import get_generator
from .metric_utils import BasicMetricsBackend, ExternalMetricsBackend


@dataclass
class ActivationRecord:
    module_type: str
    series: Dict[str, List[float]]
    last_shape: Optional[List[int]] = None
    sample_count: int = 0
    # Track result shapes for non-scalar metric outputs
    result_shape_counts: Dict[str, Dict[str, int]] = None


class ActivationHookHandleGroup:
    """Container for managing multiple hook handles."""

    def __init__(self, handles: Optional[List[Any]] = None) -> None:
        self._handles: List[Any] = list(handles or [])

    def add(self, handle: Any) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ActivationMetricsAnalyzer:
    """Pure-logic analyzer to register hooks and collect activation metrics."""

    def __init__(self) -> None:
        self.metrics_map: Dict[str, Callable[[torch.Tensor], Any]] = {}
        self.compute_cfg: Dict[str, Any] = {}
        self.reductions_cfg: Dict[str, Any] = {}
        self.records: Dict[str, ActivationRecord] = {}
        self.selected_modules: List[Tuple[str, str]] = []
        self.metric_success_counts: Dict[str, int] = {}
        self.metric_error_counts: Dict[str, int] = {}
        self.total_activation_events: int = 0
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def start_capture(self, metrics_map: Dict[str, Callable[[torch.Tensor], Any]], compute_cfg: Dict[str, Any], reductions_cfg: Dict[str, Any]) -> None:
        self.metrics_map = dict(metrics_map or {})
        self.compute_cfg = dict(compute_cfg or {})
        self.reductions_cfg = dict(reductions_cfg or {})
        self.records.clear()
        self.selected_modules.clear()
        self.metric_success_counts.clear()
        self.metric_error_counts.clear()
        self.total_activation_events = 0
        if self.logger:
            names = list(self.metrics_map.keys())
            preview = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
            self.logger.info(f"Activation capture started with {len(names)} metrics: {preview}")

    def _select_modules(
        self, model: torch.nn.Module, selection: Dict[str, Any]
    ) -> List[Tuple[str, torch.nn.Module]]:
        include = selection.get("include_names") or []
        exclude = selection.get("exclude_names") or []
        module_types = set(selection.get("module_types") or [])
        max_layers = selection.get("max_layers")

        selected: List[Tuple[str, torch.nn.Module]] = []
        for name, module in model.named_modules():
            if name == "":
                # skip the root module itself
                continue
            if include and not any(pat in name for pat in include):
                continue
            if exclude and any(pat in name for pat in exclude):
                continue
            if module_types and type(module).__name__ not in module_types:
                continue
            selected.append((name, module))
        if max_layers is not None:
            selected = selected[: int(max_layers)]
        return selected

    def register_hooks(
        self, model: torch.nn.Module, selection: Dict[str, Any]
    ) -> ActivationHookHandleGroup:
        handles = ActivationHookHandleGroup()
        selected = self._select_modules(model, selection)
        self.selected_modules = [(name, type(module).__name__) for name, module in selected]
        if self.logger:
            names = [n for n, _ in self.selected_modules]
            preview = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
            self.logger.info(f"Selected {len(names)} modules for capture: {preview}")
        move_to_cpu = bool(self.compute_cfg.get("move_to_cpu", True))

        def _normalize_output(output):
            # Accept tuple/list/dict outputs; pick the primary tensor
            if isinstance(output, (tuple, list)) and len(output) > 0:
                output = output[0]
            elif isinstance(output, dict) and len(output) > 0:
                # first value
                output = next(iter(output.values()))
            return output

        for full_name, module in selected:
            module_type = type(module).__name__

            def make_hook(name: str, mtype: str):
                def hook(_module, _inputs, output):
                    t = _normalize_output(output)
                    if not isinstance(t, torch.Tensor):
                        return
                    tt = t.detach()
                    if move_to_cpu and tt.device.type != "cpu":
                        tt = tt.to("cpu")
                    self.on_activation(name, mtype, tt)

                return hook

            handle = module.register_forward_hook(make_hook(full_name, module_type))
            handles.add(handle)
        return handles

    def on_activation(self, name: str, module_type: str, tensor: torch.Tensor) -> None:
        if not self.metrics_map:
            return
        if name not in self.records:
            self.records[name] = ActivationRecord(module_type=module_type, series={}, last_shape=None, sample_count=0, result_shape_counts={})
        rec = self.records[name]
        rec.last_shape = list(tensor.shape)
        rec.sample_count += 1
        self.total_activation_events += 1

        for metric_name, fn in self.metrics_map.items():
            try:
                out = fn(tensor)
                self._record_metric_output(rec, metric_name, out)
            except Exception as e:
                # skip metric on failure
                self.metric_error_counts[metric_name] = self.metric_error_counts.get(metric_name, 0) + 1
                if self.logger:
                    self.logger.debug(f"Metric '{metric_name}' failed on '{name}': {e}")
                continue

    def _to_float(self, v: Any) -> Optional[float]:
        try:
            if isinstance(v, (int, float)):
                return float(v)
            # torch scalar tensor
            if hasattr(v, "item") and callable(getattr(v, "item")):
                val = v.item()
                if isinstance(val, (int, float)):
                    return float(val)
            # numpy scalar
            if hasattr(v, "astype") and hasattr(v, "dtype") and hasattr(v, "shape") and v.shape == ():
                return float(v)
        except Exception:
            return None
        return None

    def _as_tensor_and_shape(self, v: Any) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        try:
            import numpy as _np  # optional
        except Exception:
            _np = None
        if isinstance(v, torch.Tensor):
            t = v.detach()
            if t.device.type != "cpu":
                t = t.to("cpu")
            return t, list(t.shape)
        if _np is not None and isinstance(v, _np.ndarray):
            t = torch.from_numpy(v)
            return t, list(t.shape)
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
            t = torch.tensor(v, dtype=torch.float32)
            return t, list(t.shape)
        return None, None

    def _record_shape(self, rec: ActivationRecord, key: str, shape: Optional[List[int]]) -> None:
        if rec.result_shape_counts is None:
            rec.result_shape_counts = {}
        if shape is None:
            return
        skey = str(shape)
        rec.result_shape_counts.setdefault(key, {})
        rec.result_shape_counts[key][skey] = rec.result_shape_counts[key].get(skey, 0) + 1

    def _record_reductions(self, rec: ActivationRecord, base_key: str, t: torch.Tensor) -> None:
        if t.numel() == 0:
            return
        import numpy as _np
        flat = t.detach().float().reshape(-1).cpu().numpy()
        vmin = float(_np.min(flat))
        vmax = float(_np.max(flat))
        vmean = float(_np.mean(flat))
        vvar = float(_np.var(flat))
        # histogram
        try:
            counts, edges = _np.histogram(flat, bins=32)
            probs = counts.astype(_np.float64)
            total = probs.sum()
            if total > 0:
                probs = probs / total
                entropy = float(-_np.sum(_np.where(probs > 0, probs * _np.log2(probs), 0.0)))
            else:
                entropy = 0.0
        except Exception:
            counts, edges, entropy = _np.array([]), _np.array([]), 0.0

        # norms
        frob = float(_np.linalg.norm(flat))

        # spectral norm
        spec = None
        try:
            if t.ndim >= 2:
                m = t.reshape(int(_np.prod(t.shape[:-1])), t.shape[-1]).float().cpu()
                # top singular value
                svals = torch.linalg.svdvals(m)
                spec = float(svals.max().item()) if svals.numel() > 0 else 0.0
            else:
                spec = float(frob)
        except Exception:
            spec = None

        # ranks
        cp_rank_est = None
        tucker_rank_est = None
        if t.ndim >= 3:
            ranks = []
            for mode in range(t.ndim):
                # unfold along mode
                dim = t.shape[mode]
                other = int(_np.prod([d for i, d in enumerate(t.shape) if i != mode]))
                m = t.permute(mode, *[i for i in range(t.ndim) if i != mode]).reshape(dim, other).float().cpu()
                try:
                    r = int(torch.linalg.matrix_rank(m).item())
                except Exception:
                    r = int(min(dim, other))
                ranks.append(r)
            if ranks:
                cp_rank_est = int(min(ranks))
                tucker_rank_est = tuple(int(r) for r in ranks)

        # singular values for 2D
        singular_values = None
        if t.ndim == 2:
            try:
                svals = torch.linalg.svdvals(t.float().cpu())
                k = int(min(64, svals.numel()))
                singular_values = [float(x) for x in svals[:k].tolist()]
            except Exception:
                singular_values = None

        # mutual information between first two modes (approximate)
        mi = None
        if t.ndim >= 2:
            try:
                arr = t.detach().cpu().float().numpy()
                # reduce higher dims by mean
                if arr.ndim > 2:
                    axes = tuple(range(2, arr.ndim))
                    arr2 = arr.mean(axis=axes)
                else:
                    arr2 = arr
                A = _np.abs(arr2)
                total = A.sum()
                if total > 0:
                    Pij = A / total
                    Pi = Pij.sum(axis=1, keepdims=True)
                    Pj = Pij.sum(axis=0, keepdims=True)
                    with _np.errstate(divide='ignore', invalid='ignore'):
                        term = _np.where((Pij > 0) & (Pi > 0) & (Pj > 0), Pij * _np.log2(Pij / (Pi * Pj)), 0.0)
                    mi = float(_np.nansum(term))
                else:
                    mi = 0.0
            except Exception:
                mi = None

        # record all
        rec_vals = {
            f"{base_key}.mean": vmean,
            f"{base_key}.variance": vvar,
            f"{base_key}.min": vmin,
            f"{base_key}.max": vmax,
            f"{base_key}.frobenius_norm": frob,
        }
        if spec is not None:
            rec_vals[f"{base_key}.spectral_norm"] = float(spec)
        if cp_rank_est is not None:
            rec_vals[f"{base_key}.cp_rank_est"] = float(cp_rank_est)
        if tucker_rank_est is not None:
            # store as comma-separated tuple string to keep CSV simple
            rec_vals[f"{base_key}.tucker_rank_est"] = str(tuple(int(x) for x in tucker_rank_est))
        if singular_values is not None:
            # store small summary stats and top singular values separately
            sv = _np.asarray(singular_values, dtype=float)
            rec_vals[f"{base_key}.singular_values.mean"] = float(_np.mean(sv))
            rec_vals[f"{base_key}.singular_values.max"] = float(_np.max(sv))
            rec_vals[f"{base_key}.singular_values.min"] = float(_np.min(sv))
        # histogram outputs
        if counts.size > 0 and edges.size > 0:
            rec_vals[f"{base_key}.histogram_entropy"] = float(entropy)
            # We avoid dumping full hist to series to keep size reasonable; but include basic shape
            rec_vals[f"{base_key}.histogram_bins"] = float(len(edges) - 1)
        if mi is not None:
            rec_vals[f"{base_key}.mutual_information_m0_m1"] = float(mi)

        for k, fv in rec_vals.items():
            rec.series.setdefault(k, []).append(float(fv))
            self.metric_success_counts[k] = self.metric_success_counts.get(k, 0) + 1

    def _record_metric_output(self, rec: ActivationRecord, metric_name: str, out: Any) -> None:
        # Scalar
        fv = self._to_float(out)
        if fv is not None:
            rec.series.setdefault(metric_name, []).append(fv)
            self.metric_success_counts[metric_name] = self.metric_success_counts.get(metric_name, 0) + 1
            return
        # Tensor-like
        t, shape = self._as_tensor_and_shape(out)
        if t is not None and shape is not None:
            self._record_shape(rec, metric_name, shape)
            self._record_reductions(rec, metric_name, t)
            return
        # Dict: recurse
        if isinstance(out, dict):
            for k, v in out.items():
                subkey = f"{metric_name}.{k}"
                fv2 = self._to_float(v)
                if fv2 is not None:
                    rec.series.setdefault(subkey, []).append(fv2)
                    self.metric_success_counts[subkey] = self.metric_success_counts.get(subkey, 0) + 1
                    continue
                t2, shape2 = self._as_tensor_and_shape(v)
                if t2 is not None and shape2 is not None:
                    self._record_shape(rec, subkey, shape2)
                    self._record_reductions(rec, subkey, t2)
                    continue
                # Nested dicts/lists: attempt one more level for lists
                if isinstance(v, (list, tuple)):
                    t3, shape3 = self._as_tensor_and_shape(v)
                    if t3 is not None and shape3 is not None:
                        self._record_shape(rec, subkey, shape3)
                        self._record_reductions(rec, subkey, t3)
            return
        # List/tuple: attempt tensor conversion
        if isinstance(out, (list, tuple)):
            t4, shape4 = self._as_tensor_and_shape(out)
            if t4 is not None and shape4 is not None:
                self._record_shape(rec, metric_name, shape4)
                self._record_reductions(rec, metric_name, t4)
                return

    def finalize(self) -> List[Dict[str, Any]]:
        raw: List[Dict[str, Any]] = []
        for name, rec in self.records.items():
            raw.append(
                {
                    "name": name,
                    "module_type": rec.module_type,
                    "last_shape": rec.last_shape or [],
                    "sample_count": rec.sample_count,
                    "series": {k: list(v) for k, v in rec.series.items()},
                    "result_shape_counts": rec.result_shape_counts or {},
                }
            )
        if self.logger:
            nonempty_layers = sum(1 for r in raw if any(len(v) for v in r.get("series", {}).values()))
            self.logger.info(
                f"Activation capture finalized: {self.total_activation_events} events; "
                f"{nonempty_layers}/{len(raw)} layers have values"
            )
            if self.metric_success_counts:
                top_s = sorted(self.metric_success_counts.items(), key=lambda x: -x[1])[:10]
                self.logger.info("Top success metrics: " + ", ".join(f"{k}:{v}" for k, v in top_s))
            if self.metric_error_counts:
                top_e = sorted(self.metric_error_counts.items(), key=lambda x: -x[1])[:10]
                self.logger.info("Top error metrics: " + ", ".join(f"{k}:{v}" for k, v in top_e))
        return raw


class ActivationMetricsPlugin(Plugin):
    """Plugin to compute activation metrics and persist a statistics report."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Compute metrics on activations and generate a report",
            category="analysis",
        )

    def _drive_evaluation(self, model: torch.nn.Module, tokenizer, lm_eval_cfg: Dict[str, Any]) -> None:
        """Drive forward passes to trigger hooks.

        Prefer LM Eval if available and configured; otherwise perform a minimal
        dummy forward if possible.
        """
        task = (lm_eval_cfg or {}).get("task")
        batch_size = int((lm_eval_cfg or {}).get("batch_size", 1))
        limit = int((lm_eval_cfg or {}).get("limit", 0))

        # Try LM Eval path via LMHarness
        try:
            from ..evaluation.lm_eval import LMHarness, LM_EVAL_AVAILABLE
        except Exception:
            LM_EVAL_AVAILABLE = False  # type: ignore
            UnifiedLMEval = None  # type: ignore

        if task and 'LM_EVAL_AVAILABLE' in locals() and LM_EVAL_AVAILABLE and LMHarness is not None and tokenizer is not None:
            evaluator = LMHarness(tasks=[task], batch_size=batch_size, limit=limit or None, device="auto")
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Driving evaluation via LM Eval task='{task}', batch_size={batch_size}, limit={limit or 'None'}")
            evaluator.evaluate_task(model=model, tokenizer=tokenizer, task=task)
            return

        # Fallback: attempt a minimal dummy forward for models accepting a single tensor input
        model.eval()
        with torch.no_grad():
            try:
                # Heuristic: try calling model with a small random tensor
                # Assume the first linear-like layer input features can be inferred from weight shape
                in_features = None
                for m in model.modules():
                    if isinstance(m, torch.nn.Linear):
                        in_features = m.in_features
                        break
                if in_features is None:
                    # last resort
                    in_features = 8
                bs = max(1, batch_size)
                # Use seeded generator for reproducibility
                generator = get_generator()
                x = torch.randn(bs, in_features, generator=generator)
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Driving evaluation via dummy forward with input dim={in_features}, batch_size={bs}")
                _ = model(x)
            except Exception:
                # If we cannot drive a forward, we silently return; hooks won't collect
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning("Dummy forward failed; no activations captured. Provide tokenizer + lm_eval or sample inputs.")
                return

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        start = time.time()
        config = dict(self._configuration)
        analysis_cfg = config.get("analysis", {})
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        aggregation_cfg = analysis_cfg.get("aggregation", {})
        compute_cfg = analysis_cfg.get("compute", {})
        output_cfg = analysis_cfg.get("output", {})
        lm_eval_cfg = config.get("lm_eval", {})

        # Resolve model/tokenizer
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None:
            model = self.get_model("current")
        if tokenizer is None:
            try:
                tokenizer = self.get_tokenizer("current")
            except Exception:
                tokenizer = None

        # Metrics backend(s)
        ext_import = (metrics_cfg.get("import", {}) if isinstance(metrics_cfg, dict) else {})
        ext_backend = ExternalMetricsBackend(
            module_path=ext_import.get("module"),
            file_path=ext_import.get("file"),
            name_prefix=ext_import.get("name_prefix"),
        )
        provenance = ext_backend.provenance
        basic_backend = BasicMetricsBackend()

        # Determine metrics list
        requested = metrics_cfg.get("names", []) if isinstance(metrics_cfg, dict) else []
        if requested == "all":
            # union of basic + external for better coverage
            names = set(basic_backend.list_metrics().keys())
            if provenance.get("found", False):
                names.update(ext_backend.list_metrics().keys())
            metrics_to_use = sorted(names)
        else:
            metrics_to_use = [str(n).lower() for n in (requested or list(basic_backend.list_metrics().keys()))]

        # Build metrics map according to requested names, prefer external when overlapping
        metrics_map: Dict[str, Callable[[torch.Tensor], Any]] = {}
        # start with basic
        for n, fn in basic_backend.list_metrics().items():
            if not metrics_to_use or n.lower() in metrics_to_use:
                metrics_map[n.lower()] = fn
        # then external overrides/adds
        if provenance.get("found", False):
            for n, fn in ext_backend.list_metrics().items():
                key = n.lower()
                if not metrics_to_use or key in metrics_to_use:
                    metrics_map[key] = fn

        # Prepare analyzer and hooks
        analyzer = ActivationMetricsAnalyzer()
        analyzer.set_logger(self.logger)
        analyzer.start_capture(metrics_map=metrics_map, compute_cfg=compute_cfg, reductions_cfg=aggregation_cfg)

        hooks = analyzer.register_hooks(model, selection_cfg)
        if self.logger and not analyzer.selected_modules:
            self.logger.warning("No modules selected for activation capture. Check selection.include_names/module_types.")
        try:
            # Drive forward passes
            self._drive_evaluation(model, tokenizer, lm_eval_cfg)
        finally:
            hooks.remove_all()

        raw_results = analyzer.finalize()

        # Build report
        stats_mgr = ActivationsStatisticsManager()
        model_info = self.model_manager.get_model_info(model) if self.model_manager else {}
        timings = {
            "runtime_sec": time.time() - start,
            "external_metrics": provenance,
        }
        # Log summary diagnostics for debugging empty values
        if self.logger:
            sel_count = len(analyzer.selected_modules)
            nonempty = sum(1 for r in raw_results if any(len(v) for v in r.get("series", {}).values()))
            self.logger.info(
                f"Activation analysis summary: selected_layers={sel_count}, layers_with_values={nonempty}, "
                f"metrics={len(metrics_map)}"
            )
        report = stats_mgr.build_report(model_info=model_info, raw_results=raw_results, config=config, timings=timings)

        # Persist
        report_dir = output_cfg.get("report_dir") or f"analysis/activation_metrics"
        stem = output_cfg.get("stem", "report")
        if output_cfg.get("timestamp", True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"{stem}_{ts}"
        base_path = stats_mgr.build_report_path(self.context, report_dir, stem)
        if output_cfg.get("save_json", True):
            stats_mgr.save_json(report, base_path.with_suffix(".json"))
        # Always write CSV (empty allowed) to keep parity with tests
        rows = stats_mgr.extract_csv_rows(report)
        stats_mgr.save_csv(rows, base_path.with_suffix(".csv"))

        # Store in state
        if self.state_manager:
            self.state_manager.state.set("analysis.activation_metrics", report)

        return report

    def compute_activation_l2_norm(self, activation: torch.Tensor) -> float:
        """
        Compute L2 norm of an activation tensor.

        Args:
            activation: Activation tensor

        Returns:
            L2 norm as a float
        """
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)
        return float(torch.norm(activation.float(), p=2).item())

    def compute_activation_entropy(self, activation: torch.Tensor) -> float:
        """
        Compute entropy of an activation tensor.

        Uses softmax to convert activations to a probability distribution,
        then computes Shannon entropy.

        Args:
            activation: Activation tensor

        Returns:
            Entropy value as a float
        """
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)

        # Flatten and convert to probabilities via softmax
        flat = activation.float().flatten()

        # Handle edge cases
        if flat.numel() == 0:
            return 0.0

        # Softmax to get probability distribution
        probs = torch.softmax(flat, dim=0)

        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs)

        return float(entropy.item())

    def compute_activation_sparsity(self, activation: torch.Tensor) -> float:
        """
        Compute sparsity of an activation tensor.

        Sparsity is defined as the ratio of zero (or near-zero) elements
        to total elements.

        Args:
            activation: Activation tensor

        Returns:
            Sparsity ratio as a float in [0, 1]
        """
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)

        flat = activation.float().flatten()

        if flat.numel() == 0:
            return 0.0

        # Consider values below threshold as "zero"
        threshold = 1e-6
        num_zeros = (flat.abs() < threshold).sum()

        return float(num_zeros.item() / flat.numel())
