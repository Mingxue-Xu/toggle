"""
KV Cache Low-Rank Projection Calibrator

Computes projection matrices for KV cache compression using SVD on collected
activation samples, following the ASVD4LLM low-rank projection approach.

This is an offline/calibration utility: collect samples during short forwards,
then compute per-head_dim projection matrices proj_K/proj_V such that
K @ proj_K and V @ proj_V reduce head_dim→k while preserving target energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .svd_backend import build_svd_backend

@dataclass
class KVProjectionResult:
    proj_K: torch.Tensor  # [head_dim, k]
    proj_V: torch.Tensor  # [head_dim, k]
    rank: int
    energy_preserved_K: float
    energy_preserved_V: float


class KVCacheProjectionCalibrator:
    """Calibrate low-rank projection matrices for KV cache compression.

    Typical usage:
        cal = KVCacheProjectionCalibrator(target_rank=32)
        cal.collect_sample(K, V, layer_idx=0)  # repeatedly on batches
        res = cal.compute_projections()
        torch.save(res, path)
    """

    def __init__(
        self,
        target_rank: Optional[int] = None,
        preserve_energy: float = 0.9,
        svd_backend: str = "torch",
        svd_backend_config: Optional[Dict[str, object]] = None,
    ) -> None:
        if target_rank is None and not (0.0 < preserve_energy <= 1.0):
            raise ValueError("preserve_energy must be in (0,1] when target_rank is None")
        self.target_rank = target_rank
        self.preserve_energy = preserve_energy
        self._K_samples: List[torch.Tensor] = []
        self._V_samples: List[torch.Tensor] = []
        self.svd_backend_name = svd_backend
        self.svd_backend_config = dict(svd_backend_config or {})
        self.svd_backend = build_svd_backend(self.svd_backend_name, self.svd_backend_config)

    def clear_samples(self) -> None:
        self._K_samples.clear()
        self._V_samples.clear()

    def collect_sample(self, K: torch.Tensor, V: torch.Tensor, layer_idx: int, device: str = "cpu") -> None:  # layer_idx reserved for future per-layer projections
        """Collect a KV sample.

        K,V shape: [batch, heads, seq_len, head_dim]
        We flatten to [N, head_dim] and store on CPU to reduce VRAM.
        """
        if K.dim() != 4 or V.dim() != 4:
            raise ValueError("Expected K,V with shape [batch, heads, seq_len, head_dim]")
        if K.shape != V.shape:
            raise ValueError("K and V must have identical shapes")
        Kf = K.detach().to(device).reshape(-1, K.shape[-1])
        Vf = V.detach().to(device).reshape(-1, V.shape[-1])
        self._K_samples.append(Kf)
        self._V_samples.append(Vf)

    def _determine_rank(self, svals: torch.Tensor) -> int:
        if self.target_rank is not None:
            return int(self.target_rank)
        # energy-based rank: smallest k with cumulative energy >= preserve_energy
        e = svals.square()
        total = e.sum()
        if total <= 0:
            return 1
        cdf = torch.cumsum(e, dim=0) / total
        idx = torch.nonzero(cdf >= self.preserve_energy, as_tuple=False)
        return int(idx[0].item() + 1) if idx.numel() > 0 else int(svals.numel())

    def compute_projections(self) -> KVProjectionResult:
        if not self._K_samples or not self._V_samples:
            raise ValueError("No KV samples collected")
        K_all = torch.cat(self._K_samples, dim=0)  # [N, d]
        V_all = torch.cat(self._V_samples, dim=0)

        # SVD on CPU to avoid GPU memory pressure for calibration
        U_k, S_k, Vt_k = self.svd_backend.compute_svd(K_all, which="LM", full=False)
        U_v, S_v, Vt_v = self.svd_backend.compute_svd(V_all, which="LM", full=False)

        k = self._determine_rank(S_k)
        proj_K = Vt_k[:k, :].T.contiguous()  # [d, k]
        proj_V = Vt_v[:k, :].T.contiguous()

        energy_K = float(S_k[:k].square().sum() / (S_k.square().sum() + 1e-12))
        energy_V = float(S_v[:k].square().sum() / (S_v.square().sum() + 1e-12))

        return KVProjectionResult(
            proj_K=proj_K,
            proj_V=proj_V,
            rank=int(k),
            energy_preserved_K=energy_K,
            energy_preserved_V=energy_V,
        )
