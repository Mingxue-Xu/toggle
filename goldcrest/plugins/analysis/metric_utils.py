"""
Metrics backends for weight metrics analysis.

Provides dynamic import of external metrics (module or file) and a
basic fallback backend with a minimal metric set.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import Any, Callable, Dict, Optional

import torch


class MetricsBackend:
    def list_metrics(self) -> Dict[str, Callable[[torch.Tensor], Any]]:
        raise NotImplementedError

    def get(self, name: str) -> Callable[[torch.Tensor], Any]:
        metrics = self.list_metrics()
        key = name.lower()
        if key in metrics:
            return metrics[key]
        # try exact lookup
        if name in metrics:
            return metrics[name]
        raise KeyError(f"Metric '{name}' not found. Available: {list(metrics.keys())}")


class BasicMetricsBackend(MetricsBackend):
    """Minimal built-in metric set."""

    def __init__(self, *, include_advanced: bool = False):
        self._metrics = {
            "compute_lda_matrix": self._compute_lda_matrix,
            "l2_norm": self._l2_norm,
            "l1_norm": self._l1_norm,
            "mean": self._mean,
            "std": self._std,
            "sparsity": self._sparsity,
            "max_abs": self._max_abs,
        }
        if include_advanced:
            self._metrics.update({
                "entropy": self._entropy,
                "stable_rank": self._stable_rank,
            })

    def list_metrics(self) -> Dict[str, Callable[[torch.Tensor], Any]]:
        return self._metrics

    # Metric implementations
    def _l2_norm(self, t: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(t.float()).item())

    def _compute_lda_matrix(self, t: torch.Tensor) -> torch.Tensor:
        """Return a covariance-like matrix from activations for downstream reductions."""
        x = t.detach().float()
        if x.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.float32)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(-1, x.shape[-1])
        x = x - x.mean(dim=0, keepdim=True)
        denom = max(1, x.shape[0] - 1)
        return (x.T @ x) / denom

    def _l1_norm(self, t: torch.Tensor) -> float:
        return float(torch.linalg.norm(t.float(), ord=1).item())

    def _mean(self, t: torch.Tensor) -> float:
        return float(t.float().mean().item())

    def _std(self, t: torch.Tensor) -> float:
        return float(t.float().std(unbiased=False).item())

    def _sparsity(self, t: torch.Tensor) -> float:
        numel = t.numel()
        if numel == 0:
            return 0.0
        zeros = torch.count_nonzero(t == 0).item()
        return float(zeros / numel)

    def _max_abs(self, t: torch.Tensor) -> float:
        return float(torch.max(torch.abs(t.float())).item())

    def _as_matrix(self, t: torch.Tensor) -> torch.Tensor:
        x = t.detach().float()
        if x.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.float32)
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.reshape(1, -1)
        return x.reshape(-1, x.shape[-1])

    def _stable_rank(self, t: torch.Tensor) -> float:
        m = self._as_matrix(t)
        if m.numel() == 0:
            return 0.0
        fro_sq = float(torch.sum(m * m).item())
        if fro_sq <= 0.0:
            return 0.0
        try:
            spec = float(torch.linalg.matrix_norm(m, ord=2).item())
        except Exception:
            svals = torch.linalg.svdvals(m)
            spec = float(svals.max().item()) if svals.numel() > 0 else 0.0
        if spec <= 0.0:
            return 0.0
        return float(fro_sq / (spec * spec))

    def _entropy(self, t: torch.Tensor) -> float:
        """Normalized spectral entropy in [0, 1]."""
        m = self._as_matrix(t)
        if m.numel() == 0:
            return 0.0
        try:
            svals = torch.linalg.svdvals(m)
        except Exception:
            return 0.0
        if svals.numel() <= 1:
            return 0.0
        energy = svals.square()
        total = float(energy.sum().item())
        if total <= 0.0:
            return 0.0
        probs = energy / total
        nonzero = probs > 0
        probs = probs[nonzero]
        if probs.numel() <= 1:
            return 0.0
        entropy = float((-torch.sum(probs * torch.log(probs))).item())
        max_entropy = float(torch.log(torch.tensor(float(probs.numel()))).item())
        if max_entropy <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, entropy / max_entropy)))


class ExternalMetricsBackend(MetricsBackend):
    """Dynamic loader of metric functions from an external module/file."""

    def __init__(
        self,
        module_path: Optional[str] = None,
        file_path: Optional[str] = None,
        name_prefix: Optional[str] = None,
        env_module_var: str = "INFO_FLOW_METRICS_MODULE",
        env_file_var: str = "INFO_FLOW_METRICS_FILE",
    ):
        self.module_path = module_path
        self.file_path = file_path
        self.name_prefix = name_prefix or ""
        self.env_module_var = env_module_var
        self.env_file_var = env_file_var
        self._module: Optional[ModuleType] = None
        self._registry: Dict[str, Callable[[torch.Tensor], Any]] = {}
        self._provenance: Dict[str, Any] = {"found": False, "module_path": None, "load_method": None}

        self._load()

    @property
    def provenance(self) -> Dict[str, Any]:
        return dict(self._provenance)

    def list_metrics(self) -> Dict[str, Callable[[torch.Tensor], Any]]:
        return self._registry

    def _load(self) -> None:
        # 1) Configuration
        if self.module_path:
            mod = self._import_module(self.module_path)
            if mod:
                self._register_module(mod, method="module", path=self.module_path)
                return
        if self.file_path:
            mod = self._import_file(self.file_path)
            if mod:
                self._register_module(mod, method="file", path=self.file_path)
                return

        # 2) Environment
        env_mod = os.environ.get(self.env_module_var)
        if env_mod:
            mod = self._import_module(env_mod)
            if mod:
                self._register_module(mod, method="module", path=env_mod)
                return
        env_file = os.environ.get(self.env_file_var)
        if env_file:
            mod = self._import_file(env_file)
            if mod:
                self._register_module(mod, method="file", path=env_file)
                return

        # 3) Defaults
        defaults = [
            "information_flow.experiments.utils.metrics.metric_functions",
            "experiments.utils.metrics.metric_functions",
        ]
        for mp in defaults:
            mod = self._import_module(mp)
            if mod:
                self._register_module(mod, method="default", path=mp)
                return

        # 4) Fallback: empty registry
        self._provenance = {"found": False, "module_path": None, "load_method": None}

    def _import_module(self, module_path: str) -> Optional[ModuleType]:
        try:
            return importlib.import_module(module_path)
        except Exception:
            return None

    def _import_file(self, file_path: str) -> Optional[ModuleType]:
        try:
            abspath = os.path.abspath(file_path)
            name = os.path.splitext(os.path.basename(abspath))[0]
            loader = SourceFileLoader(name, abspath)
            spec = importlib.util.spec_from_loader(name, loader)
            if spec is None:
                return None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            # Make importable by name in this session
            sys.modules[name] = module
            return module
        except Exception:
            return None

    def _register_module(self, mod: ModuleType, method: str, path: str) -> None:
        self._module = mod
        self._registry = {}
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if not callable(obj):
                continue
            # Accept functions taking at least one positional argument
            try:
                sig = inspect.signature(obj)
            except Exception:
                continue
            params = list(sig.parameters.values())
            if not params:
                continue
            # Map name with optional prefix stripping
            export_name = name
            if self.name_prefix and export_name.startswith(self.name_prefix):
                export_name = export_name[len(self.name_prefix) :]
            self._registry[export_name.lower()] = obj  # store lower-case key
        self._provenance = {"found": True, "module_path": path, "load_method": method}
