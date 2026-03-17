"""
Plugin System for Toggle Architecture

This module provides the base plugin classes and plugin registry system
for the event-driven plugin-based architecture.
"""
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass

from .events import EventBus, PipelineEvent


@dataclass
class PluginMetadata:
    """Metadata for plugin registration and discovery"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    category: str = "general"


class Plugin(ABC):
    """
    Base plugin class with event integration and lifecycle management
    
    Enhanced with shared infrastructure integration including strategy factory
    and model manager support for consistent behavior across all plugins.
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.plugin_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"plugin.{self.name}")
        self._context = None
        self._event_bus = None
        self._event_manager = None
        self._initialized = False
        self._configuration = kwargs
        
        # Simple execution stats
        self._execution_count = 0
        self._error_count = 0
        
        # Shared infrastructure components (set during initialization)
        self._strategy_factory = None
        self._model_manager = None
        self._state_manager = None
        
        # Progress tracking
        self._progress = 0.0
        self._status = "initialized"
    
    @property
    def context(self):
        """Get the plugin execution context"""
        return self._context
    
    @property
    def event_bus(self) -> Optional[EventBus]:
        """Get the event bus for emitting events"""
        return self._event_bus
    
    def initialize(self, context) -> None:
        """Initialize the plugin with context and shared infrastructure"""
        self._context = context
        self._event_bus = getattr(context, 'event_bus', None)
        
        # Initialize PluginEventManager
        if self._event_bus:
            from .events import PluginEventManager
            self._event_manager = PluginEventManager(self._event_bus, self.name)
        
        # Initialize shared infrastructure components
        try:
            from .strategy_factory import UnifiedStrategyFactory
            self._strategy_factory = UnifiedStrategyFactory()
        except ImportError:
            self.logger.warning("UnifiedStrategyFactory not available")
        
        try:
            from .model_manager import ModelManager
            self._model_manager = ModelManager()
        except ImportError:
            self.logger.warning("ModelManager not available")
        
        try:
            from .state import StateManager
            if hasattr(context, 'state') and context.state:
                self._state_manager = StateManager(context.state)
        except ImportError:
            self.logger.warning("StateManager not available")
        
        self._initialized = True
        self._status = "initialized"
        
        if self._event_bus:
            self._event_bus.emit(
                PipelineEvent.PLUGIN_LOADED,
                {
                    "plugin_name": self.name,
                    "plugin_id": self.plugin_id,
                    "category": self.get_metadata().category,
                    "has_strategy_factory": self._strategy_factory is not None,
                    "has_model_manager": self._model_manager is not None,
                    "has_state_manager": self._state_manager is not None,
                    "has_event_manager": self._event_manager is not None
                },
                source=self.name
            )
        
        self.logger.info(f"Plugin {self.name} initialized successfully")
    
    def execute(self, **kwargs) -> Any:
        """Execute with event emission and error handling"""
        if not self._initialized:
            raise RuntimeError(f"Plugin {self.name} not initialized")
        
        self._status = "running"
        self._progress = 0.0
        
        try:
            if self._event_manager:
                self._event_manager.emit_started(
                    parameters=list(kwargs.keys()),  # Only log parameter names
                    plugin_id=self.plugin_id
                )
            
            result = self.do_execute(**kwargs)
            self._status = "completed"
            self._progress = 100.0
            
            # Store result in state
            if self._state_manager:
                self._state_manager.state.set(f'plugin_results.{self.name}', result)
            
            if self._event_manager:
                self._event_manager.emit_completed(
                    results=result,
                    success=True,
                    plugin_id=self.plugin_id,
                    execution_count=self._execution_count + 1
                )
            
            self._execution_count += 1
            return result
            
        except Exception as e:
            self._error_count += 1
            self._execution_count += 1
            self._status = "failed"
            self.logger.error(f"Plugin {self.name} execution failed: {str(e)}")
            
            if self._event_manager:
                self._event_manager.emit_failed(
                    error=e,
                    plugin_id=self.plugin_id
                )
            
            raise
    
    @abstractmethod
    def do_execute(self, **kwargs) -> Any:
        """
        Implement the actual plugin logic here
        
        Args:
            **kwargs: Plugin-specific parameters
            
        Returns:
            Plugin execution results
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources when plugin execution is complete"""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description=self.__class__.__doc__ or "",
            category="general"
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get basic plugin execution statistics"""
        return {
            "executions": self._execution_count,
            "errors": self._error_count,
            "success_rate": (self._execution_count - self._error_count) / max(1, self._execution_count),
            "current_status": self._status,
            "current_progress": self._progress
        }
    
    @property
    def strategy_factory(self):
        """Get the strategy factory for creating strategies"""
        return self._strategy_factory
    
    @property
    def model_manager(self):
        """Get the model manager for model operations"""
        return self._model_manager
    
    @property
    def state_manager(self):
        """Get the state manager for state operations"""
        return self._state_manager
    
    def update_progress(self, progress: float) -> None:
        """
        Update plugin execution progress

        Args:
            progress: Progress value between 0.0 and 100.0
        """
        self._progress = max(0.0, min(100.0, progress))

        if self._event_manager:
            self._event_manager.emit_progress(
                progress=self._progress,
                plugin_id=self.plugin_id,
                status=self._status
            )

    def emit_progress(self, progress: float, message: str = "") -> None:
        """
        Emit progress event with optional message

        Args:
            progress: Progress value (0.0 to 1.0 or 0.0 to 100.0)
            message: Optional progress message
        """
        # Normalize progress to 0-100 scale
        if progress <= 1.0:
            progress = progress * 100.0
        self._progress = max(0.0, min(100.0, progress))

        if self._event_manager:
            self._event_manager.emit_progress(
                progress=self._progress,
                message=message,
                plugin_id=self.plugin_id,
                status=self._status
            )

    def get_model(self, model_type: str = "compressed"):
        """
        Convenience method to get model via model manager
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            Model instance
        """
        if not self._model_manager:
            raise RuntimeError("ModelManager not available - plugin not properly initialized")
        
        return self._model_manager.get_model(self._context, model_type)
    
    def get_tokenizer(self, model_type: str = "compressed"):
        """
        Convenience method to get tokenizer via model manager
        
        Args:
            model_type: Type of tokenizer to retrieve
            
        Returns:
            Tokenizer instance
        """
        if not self._model_manager:
            raise RuntimeError("ModelManager not available - plugin not properly initialized")
        
        return self._model_manager.get_tokenizer(self._context, model_type)
    
    def emit_event(self, event_type: str, data: Dict[str, Any], priority: int = 0) -> None:
        """
        Convenience method to emit events with plugin context
        
        Args:
            event_type: Type of event to emit
            data: Event data
            priority: Event priority (0=normal, 1=high, 2=critical)
        """
        if self._event_bus:
            enhanced_data = data.copy()
            enhanced_data.update({
                "plugin_name": self.name,
                "plugin_id": self.plugin_id
            })
            
            self._event_bus.emit(
                event_type,
                enhanced_data,
                source=self.name,
                priority=priority
            )
    
    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Log a metric for this plugin execution
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.emit_event(
            f"plugin.{self.name.lower()}.metric",
            {
                "metric": metric_name,
                "value": value,
                "execution_count": self._execution_count
            }
        )


class PluginRegistry:
    """Simplified registry for plugin registration and management"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self.logger = logging.getLogger("plugin_registry")
    
    def register(self, plugin_class: Type[Plugin], name: Optional[str] = None) -> None:
        """
        Register a plugin class
        
        Args:
            plugin_class: Plugin class to register
            name: Override plugin name (defaults to class name)
        """
        plugin_name = name or plugin_class.__name__
        
        if plugin_name in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' is already registered")
        
        # Validate plugin class
        if not issubclass(plugin_class, Plugin):
            raise TypeError(f"Plugin class must inherit from Plugin")
        
        self._plugins[plugin_name] = plugin_class
        self.logger.info(f"Registered plugin: {plugin_name}")
    
    def create_plugin(self, name: str, **kwargs) -> Plugin:
        """
        Create a plugin instance
        
        Args:
            name: Plugin name
            **kwargs: Plugin constructor arguments
            
        Returns:
            Plugin instance
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered")
        
        plugin_class = self._plugins[name]
        return plugin_class(name=name, **kwargs)
    
    def list_plugins(self) -> List[str]:
        """List available plugins"""
        return list(self._plugins.keys())