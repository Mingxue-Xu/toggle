"""Tensorly backend selection helpers for compression plugins."""
from __future__ import annotations

from typing import Optional

import tensorly as tl


def set_tensorly_backend(backend: Optional[str]) -> None:
    if not backend:
        return
    backend_name = str(backend).lower()
    if backend_name == "torch":
        backend_name = "pytorch"
    if backend_name == "cola":
        # TODO: add a tensorly backend adapter for CoLA.
        raise NotImplementedError("CoLA tensorly backend is not implemented yet.")
    tl.set_backend(backend_name)
