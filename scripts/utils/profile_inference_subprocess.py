#!/usr/bin/env python3
"""
CLI wrapper for isolated inference profiling.

Delegates to src.framework.inference_subprocess to keep logic in src/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path to import src.*
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framework.inference_subprocess import main, DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS

__all__ = ["main", "DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS"]


if __name__ == "__main__":
    raise SystemExit(main())
