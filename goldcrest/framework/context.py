"""
Pipeline Context and State Management for Goldcrest Architecture

This module provides the execution context and state management system
for plugin-based pipeline orchestration.
"""
import uuid
import logging
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .events import EventBus
from .state import PipelineState, StateManager
from .reproducibility import set_seed, get_seed, is_seed_set


@dataclass
class ExecutionMetadata:
    """Metadata about pipeline execution"""
    pipeline_id: str
    start_time: datetime
    workflow_name: Optional[str] = None


class PipelineContext:
    """
    Centralized context for plugin execution with shared state and services
    
    The PipelineContext provides:
    - Shared state management across plugins
    - Event bus for inter-plugin communication  
    - Configuration access
    - Basic resource management
    """
    
    def __init__(self,
                 event_bus: Optional[EventBus] = None,
                 config: Optional[Dict[str, Any]] = None,
                 workspace_dir: Optional[Union[str, Path]] = None,
                 pipeline_id: Optional[str] = None):
        """
        Initialize pipeline context
        
        Args:
            event_bus: Event bus for communication (creates new if None)
            config: Configuration dictionary
            workspace_dir: Working directory for pipeline execution
            pipeline_id: Unique pipeline identifier (generates if None)
        """
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.event_bus = event_bus or EventBus()
        self.config = config or {}
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd() / "workspace"
        
        # Initialize state management
        self.state = PipelineState()
        self.state_manager = StateManager(self.state)
        
        # Execution metadata
        self.metadata = ExecutionMetadata(
            pipeline_id=self.pipeline_id,
            start_time=datetime.now()
        )
        
        # Simple resource management
        self._resources: Dict[str, Any] = {}
        
        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging for context
        self.logger = logging.getLogger(f"context.{self.pipeline_id[:8]}")

        # Initialize reproducibility seed from config if provided
        self._init_seed_from_config()

        self.logger.info(f"Pipeline context initialized: {self.pipeline_id}")
    
    def _init_seed_from_config(self) -> None:
        """
        Initialize reproducibility seed from configuration.

        Looks for seed in multiple config locations:
        - config.seed
        - config.analysis.compute.seed
        - config.compute.seed
        """
        seed = None

        # Check common seed locations in config
        seed_paths = [
            "seed",
            "analysis.compute.seed",
            "compute.seed",
            "runtime.seed",
        ]

        for path in seed_paths:
            seed = self.get_config(path)
            if seed is not None:
                break

        if seed is not None:
            try:
                seed = int(seed)
                deterministic = self.get_config("reproducibility.deterministic", True)
                set_seed(seed, deterministic=deterministic)
                self.logger.info(f"Reproducibility seed initialized: {seed}")
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid seed value in config: {seed} ({e})")

    def get_seed(self) -> Optional[int]:
        """
        Get the current reproducibility seed.

        Returns:
            The seed value if set, None otherwise
        """
        return get_seed()

    def is_seed_set(self) -> bool:
        """
        Check if a reproducibility seed has been set.

        Returns:
            True if a seed has been set
        """
        return is_seed_set()

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support
        
        Args:
            key: Configuration key (supports dot notation like 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if '.' not in key:
            self.config[key] = value
            return
        
        # Handle nested keys
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_resource(self, name: str) -> Optional[Any]:
        """
        Get a shared resource by name
        
        Args:
            name: Resource name
            
        Returns:
            Resource object or None if not found
        """
        return self._resources.get(name)
    
    def set_resource(self, name: str, resource: Any) -> None:
        """
        Set a shared resource
        
        Args:
            name: Resource name
            resource: Resource object
        """
        self._resources[name] = resource
    
    def get_workspace_path(self, *path_parts: str) -> Path:
        """
        Get a path within the workspace directory
        
        Args:
            *path_parts: Path components
            
        Returns:
            Path within workspace
        """
        path = self.workspace_dir
        for part in path_parts:
            path = path / part
        return path
    
    def save_state(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Save pipeline state to disk
        
        Args:
            filepath: Path to save state (defaults to workspace)
            
        Returns:
            Path where state was saved
        """
        if filepath is None:
            filepath = self.get_workspace_path(f"pipeline_state_{self.pipeline_id}.json")
        
        return self.state_manager.save_state(filepath)
    
    def load_state(self, filepath: Union[str, Path]) -> None:
        """
        Load pipeline state from disk
        
        Args:
            filepath: Path to load state from
        """
        self.state_manager.load_state(filepath)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary with metadata and statistics
        
        Returns:
            Dictionary with execution information
        """
        summary = {
            "pipeline_id": self.pipeline_id,
            "start_time": self.metadata.start_time.isoformat(),
            "duration": (datetime.now() - self.metadata.start_time).total_seconds(),
            "workflow_name": self.metadata.workflow_name,
            "state_summary": self.state.get_summary(),
            "resource_count": len(self._resources)
        }
        
        return summary
    
    def cleanup(self) -> None:
        """Cleanup all resources and close context"""
        self.logger.info(f"Cleaning up pipeline context: {self.pipeline_id}")
        
        # Clear resources
        self._resources.clear()
        
        # Clear event bus history
        if self.event_bus:
            self.event_bus.clear_history()
        
        self.logger.info("Pipeline context cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def __str__(self) -> str:
        """String representation"""
        return f"PipelineContext(id={self.pipeline_id[:8]}, workspace={self.workspace_dir})"
