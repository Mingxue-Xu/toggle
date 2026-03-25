"""
Reproducibility utilities for Goldcrest framework.

This module provides seed management and determinism controls to ensure
experiment reproducibility across runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global state for tracking seed
_GLOBAL_SEED: Optional[int] = None
_SEED_SET: bool = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all random sources.

    Args:
        seed: The seed value to use
        deterministic: If True, also set CUDA deterministic flags (may impact performance)
    """
    global _GLOBAL_SEED, _SEED_SET

    _GLOBAL_SEED = seed
    _SEED_SET = True

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA determinism flags
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for additional CUDA determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enable PyTorch deterministic algorithms (may raise errors for non-deterministic ops)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch versions may not support warn_only
            pass

    logger.info(f"Reproducibility seed set to {seed} (deterministic={deterministic})")


def get_seed() -> Optional[int]:
    """
    Get the currently set global seed.

    Returns:
        The seed value if set, None otherwise
    """
    return _GLOBAL_SEED


def is_seed_set() -> bool:
    """
    Check if a global seed has been set.

    Returns:
        True if set_seed() has been called
    """
    return _SEED_SET


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader to ensure reproducibility.

    Use this with DataLoader's worker_init_fn parameter:
        DataLoader(..., worker_init_fn=seed_worker)

    Args:
        worker_id: The worker ID provided by DataLoader
    """
    worker_seed = (_GLOBAL_SEED or 0) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a seeded PyTorch generator for reproducible random operations.

    Args:
        seed: Optional seed override. If None, uses global seed or 42 as fallback.

    Returns:
        A seeded torch.Generator
    """
    if seed is None:
        seed = _GLOBAL_SEED if _GLOBAL_SEED is not None else 42

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def config_hash(config: Dict[str, Any]) -> str:
    """
    Generate a deterministic hash from a configuration dictionary.

    Useful for versioning model artifacts based on their configuration.

    Args:
        config: Configuration dictionary

    Returns:
        8-character hex hash string
    """
    # Sort keys for deterministic serialization
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get current reproducibility state information.

    Returns:
        Dictionary with seed, CUDA determinism flags, and library versions
    """
    info = {
        "seed": _GLOBAL_SEED,
        "seed_set": _SEED_SET,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    return info


__all__ = [
    "set_seed",
    "get_seed",
    "is_seed_set",
    "seed_worker",
    "get_generator",
    "config_hash",
    "get_reproducibility_info",
]
