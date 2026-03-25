"""Backend abstraction for 2D SVD operations."""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import importlib
import sys

import torch


def _candidate_cola_paths() -> Tuple[Path, ...]:
    candidates = []
    for env_name in ("TOGGLE_COLA_PATH", "COLA_PATH", "COLA_SRC"):
        raw_path = os.environ.get(env_name)
        if raw_path:
            candidates.append(Path(raw_path).expanduser())

    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            cwd / "third_party" / "cola",
            cwd / "cola",
            repo_root / "third_party" / "cola",
            repo_root.parent / "cola",
        ]
    )

    unique_candidates = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)
    return tuple(unique_candidates)


def _normalize_which(which: Optional[str]) -> str:
    normalized = (which or "LM").upper()
    if normalized not in ("LM", "SM"):
        raise ValueError(f"Unsupported which='{which}'. Expected 'LM' or 'SM'.")
    return normalized


def _normalize_k(k: Optional[int], max_rank: int) -> Optional[int]:
    if k is None:
        return None
    k_int = int(k)
    if k_int <= 0:
        raise ValueError("k must be positive when provided")
    return min(k_int, max_rank)


def _order_svd_uvh(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    k: Optional[int],
    which: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    which = _normalize_which(which)
    order = torch.argsort(S, descending=(which == "LM"))
    if k is not None:
        order = order[:k]
    return U[:, order], S[order], Vh[order, :]


def _order_svd_uv(
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    k: Optional[int],
    which: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    which = _normalize_which(which)
    order = torch.argsort(S, descending=(which == "LM"))
    if k is not None:
        order = order[:k]
    U = U[:, order]
    V = V[:, order]
    S = S[order]
    Vh = V.transpose(-2, -1).conj()
    return U, S, Vh


class SVDBackend(ABC):
    @abstractmethod
    def compute_svd(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
        full: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def svdvals(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
    ) -> torch.Tensor:
        raise NotImplementedError


class TorchSVDBackend(SVDBackend):
    def compute_svd(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
        full: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if matrix.ndim != 2:
            raise ValueError("SVD backend expects a 2D matrix")
        max_rank = min(matrix.shape)
        k = _normalize_k(k, max_rank)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=full)
        return _order_svd_uvh(U, S, Vh, k, which)

    def svdvals(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
    ) -> torch.Tensor:
        if matrix.ndim != 2:
            raise ValueError("SVD backend expects a 2D matrix")
        max_rank = min(matrix.shape)
        k = _normalize_k(k, max_rank)
        which = _normalize_which(which)
        svals = torch.linalg.svdvals(matrix)
        ordered = torch.sort(svals, descending=(which == "LM")).values
        if k is not None:
            ordered = ordered[:k]
        return ordered


class ColaSVDBackend(SVDBackend):
    def __init__(
        self,
        algorithm: str = "auto",
        which: str = "LM",
        **kwargs: Any,
    ) -> None:
        self.algorithm = (algorithm or "auto").lower()
        self.which = _normalize_which(which)
        self.algorithm_config = dict(kwargs)

    def _cola_has_svd(self) -> bool:
        try:
            importlib.import_module("cola.linalg.svd.svd")
            return True
        except Exception:
            return False

    def _purge_cola_modules(self) -> None:
        for name in list(sys.modules):
            if name == "cola" or name.startswith("cola."):
                del sys.modules[name]

    def _import_cola(self):
        last_exc: Optional[BaseException] = None
        try:
            cola = importlib.import_module("cola")  # type: ignore
            if self._cola_has_svd():
                return cola
        except ImportError as exc:
            last_exc = exc
        # Try caller-provided or nearby CoLA checkouts before failing.
        for local_path in _candidate_cola_paths():
            if not local_path.exists():
                continue
            if str(local_path) not in sys.path:
                sys.path.insert(0, str(local_path))
            self._purge_cola_modules()
            try:
                cola = importlib.import_module("cola")  # type: ignore
                if self._cola_has_svd():
                    return cola
            except ImportError as exc:
                last_exc = exc

        message = (
            "CoLA backend requested but 'cola.linalg.svd' is not importable. "
            "Install cola-ml, add a CoLA checkout to PYTHONPATH, or set "
            "TOGGLE_COLA_PATH/COLA_PATH/COLA_SRC."
        )
        if last_exc:
            raise ImportError(message) from last_exc
        raise ImportError(message)

    def _import_svd_module(self):
        self._import_cola()
        return importlib.import_module("cola.linalg.svd.svd")

    def _build_algorithm(self, svd_mod: Optional[Any] = None):
        self._import_cola()
        algo_kwargs = {
            k: v
            for k, v in self.algorithm_config.items()
            if k in {"start_vector", "max_iters", "tol", "pbar", "key"} and v is not None
        }
        if self.algorithm == "auto":
            from cola.linalg.algorithm_base import Auto

            return Auto(**algo_kwargs)
        if self.algorithm == "dense":
            svd_mod = svd_mod or self._import_svd_module()
            return svd_mod.DenseSVD()
        if self.algorithm == "lanczos":
            from cola.linalg.decompositions.decompositions import Lanczos

            return Lanczos(**algo_kwargs)
        if self.algorithm == "lobpcg":
            from cola.linalg.eig.lobpcg import LOBPCG

            lobpcg_kwargs = {k: v for k, v in algo_kwargs.items() if k == "max_iters"}
            return LOBPCG(**lobpcg_kwargs)
        raise ValueError(f"Unknown CoLA SVD algorithm '{self.algorithm}'")

    def _extract_sigma(self, sigma_op: Any) -> torch.Tensor:
        if hasattr(sigma_op, "diag"):
            sigma = sigma_op.diag
        else:
            dense = sigma_op.to_dense()
            sigma = torch.diagonal(dense)
        return sigma.reshape(-1)

    def compute_svd(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
        full: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if matrix.ndim != 2:
            raise ValueError("SVD backend expects a 2D matrix")
        max_rank = min(matrix.shape)
        k = _normalize_k(k, max_rank)
        which = _normalize_which(which or self.which)
        svd_mod = self._import_svd_module()
        Dense = svd_mod.Dense
        alg = self._build_algorithm(svd_mod)
        k_for_cola = k if k is not None else max_rank
        U_op, Sigma_op, V_op = svd_mod.svd(Dense(matrix), k_for_cola, which=which, alg=alg)
        U = U_op.to_dense()
        S = self._extract_sigma(Sigma_op)
        V = V_op.to_dense()
        return _order_svd_uv(U, S, V, k, which)

    def svdvals(
        self,
        matrix: torch.Tensor,
        k: Optional[int] = None,
        which: str = "LM",
    ) -> torch.Tensor:
        U, S, _ = self.compute_svd(matrix, k=k, which=which, full=False)
        return S


def build_svd_backend(name: Optional[str], config: Optional[Dict[str, Any]] = None) -> SVDBackend:
    backend = (name or "torch").lower()
    config = dict(config or {})
    if backend in ("torch", "pytorch"):
        return TorchSVDBackend()
    if backend == "cola":
        return ColaSVDBackend(**config)
    raise ValueError(f"Unknown SVD backend '{name}'")
