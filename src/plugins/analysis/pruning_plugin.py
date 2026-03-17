from __future__ import annotations

import warnings

# Backward-compatibility shim: re-export from compression package
from ..compression.pruning import PruningPlugin  # noqa: F401

warnings.warn(
    "src.plugins.analysis.pruning_plugin.PruningPlugin is deprecated; "
    "use src.plugins.compression.pruning.PruningPlugin instead.",
    DeprecationWarning,
    stacklevel=2,
)
