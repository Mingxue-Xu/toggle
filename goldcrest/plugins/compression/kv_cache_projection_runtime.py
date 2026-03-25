"""
KV Cache Low-Rank Projection Runtime Adapter

Provides utilities to compress/decompress KV tensors at runtime using
precomputed projection matrices, plus an optional forward hook that can
transform module outputs carrying KV tuples.

Note: Hooking into HF LlamaAttention is model-specific; for generic tests we 
provide a simple forward-output tuple transform. Production integrations should
install method-specific wrappers during model load without editing HF code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class KVProjector:
    proj_K: torch.Tensor  # [head_dim, k]
    proj_V: torch.Tensor  # [head_dim, k]

    def to(self, device: torch.device | str) -> "KVProjector":
        self.proj_K = self.proj_K.to(device)
        self.proj_V = self.proj_V.to(device)
        return self

    @property
    def rank(self) -> int:
        return int(self.proj_K.shape[-1])

    def compress_kv(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress K,V along last dim using projection matrices.

        Shapes:
          K,V: [B, H, T, D] → Kc,Vc: [B, H, T, k]
        """
        if K.shape != V.shape:
            raise ValueError("K and V must have identical shapes")
        if K.shape[-1] != self.proj_K.shape[0] or V.shape[-1] != self.proj_V.shape[0]:
            raise ValueError("Last dim of K/V must match projector input dim")
        Kc = torch.matmul(K, self.proj_K)  # [..., D] @ [D, k]
        Vc = torch.matmul(V, self.proj_V)
        return Kc, Vc

    def decompress_kv(self, Kc: torch.Tensor, Vc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress K_c,V_c using transposed projections.

        Shapes:
          Kc,Vc: [B, H, T, k] → K,V: [B, H, T, D]
        """
        if Kc.shape != Vc.shape:
            raise ValueError("Kc and Vc must have identical shapes")
        K = torch.matmul(Kc, self.proj_K.T)
        V = torch.matmul(Vc, self.proj_V.T)
        return K, V


class KVForwardHook:
    """Attach a forward hook that rewrites KV tuple in module outputs.

    This is a convenience for modules that return outputs like:
      - (attn_out, (K, V)) or (attn_out, present_key_value)

    The hook replaces (K,V) with (Kc,Vc) using the given projector. It also
    stores the projector on the module for optional downstream use.
    """

    def __init__(self, projector: KVProjector, output_kv_index: int = 1) -> None:
        self.projector = projector
        self.output_kv_index = int(output_kv_index)
        self._handle: Optional[Any] = None

    def _hook(self, _module: nn.Module, _inputs: Tuple, output: Any):
        try:
            # We only transform tuple/list outputs with a KV pair at output_kv_index
            if isinstance(output, (tuple, list)) and len(output) > self.output_kv_index:
                kv = output[self.output_kv_index]
                if isinstance(kv, (tuple, list)) and len(kv) >= 2:
                    K, V = kv[0], kv[1]
                    if isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor):
                        Kc, Vc = self.projector.compress_kv(K, V)
                        new_kv = (Kc, Vc) + tuple(kv[2:]) if isinstance(kv, tuple) else [Kc, Vc, *kv[2:]]
                        if isinstance(output, tuple):
                            output = tuple(
                                new_kv if i == self.output_kv_index else output[i]
                                for i in range(len(output))
                            )
                        else:
                            output[self.output_kv_index] = new_kv
        except Exception:
            # Be conservative; on failure, leave output unchanged
            return output
        return output

    def attach(self, module: nn.Module) -> None:
        if self._handle is not None:
            return
        # Store projector on module for optional consumers
        setattr(module, "_kv_projector", self.projector)
        self._handle = module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

