"""
Goldcrest Framework - Phase 1 Implementation

This package provides the core framework for the Goldcrest plugin-based architecture:
- EventBus: Central event communication system
- Plugin: Base plugin class with lifecycle management  
- PluginRegistry: Plugin registration and management
- PipelineContext: Shared execution context and state management
- PipelineState: Centralized state container
- StateManager: State persistence operations
- ConfigurationLoader: Configuration loading with validation
"""

from .events import EventBus, EventMessage, PipelineEvent
from .plugins import Plugin, PluginRegistry, PluginMetadata
from .context import PipelineContext, ExecutionMetadata
from .state import PipelineState, StateManager
from .reproducibility import (
    set_seed,
    get_seed,
    is_seed_set,
    seed_worker,
    get_generator,
    config_hash,
    get_reproducibility_info,
)

__all__ = [
    # Events
    'EventBus',
    'EventMessage',
    'PipelineEvent',

    # Plugins
    'Plugin',
    'PluginRegistry',
    'PluginMetadata',

    # Context and State
    'PipelineContext',
    'ExecutionMetadata',
    'PipelineState',
    'StateManager',

    # Reproducibility
    'set_seed',
    'get_seed',
    'is_seed_set',
    'seed_worker',
    'get_generator',
    'config_hash',
    'get_reproducibility_info',
]

__version__ = "1.0.0-phase1"