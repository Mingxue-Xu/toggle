"""
Event System for Toggle Plugin Architecture

This module implements the central event bus and event message system for 
loose coupling between components in the plugin-based architecture.
"""
import threading
import logging
from collections import defaultdict
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EventMessage:
    """Standard event message format"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=critical


class EventBus:
    """Central event bus for component communication"""
    
    def __init__(self, max_history: int = 1000):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[EventMessage] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        
    def emit(self, event_type: str, data: Dict[str, Any],
             source: Optional[str] = None,
             correlation_id: Optional[str] = None,
             priority: int = 0) -> None:
        """Emit an event to all subscribers (alias: publish)"""
        
        message = EventMessage(
            event_type=event_type,
            data=data.copy(),
            timestamp=datetime.now(),
            source=source,
            correlation_id=correlation_id,
            priority=priority
        )
        
        with self._lock:
            # Add to history
            self._event_history.append(message)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Get handlers for this event type and wildcard
            handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])
        
        # Log event emission
        self._logger.debug(f"Emitting event: {event_type} from {source}")
        
        # Call synchronous handlers
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                self._logger.error(f"Error in event handler for {event_type}: {str(e)}")
    
    def publish(self, event_type: str, data: Dict[str, Any],
                source: Optional[str] = None,
                correlation_id: Optional[str] = None,
                priority: int = 0) -> None:
        """Publish an event to all subscribers (alias for emit)"""
        self.emit(event_type, data, source, correlation_id, priority)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type"""
        with self._lock:
            self._handlers[event_type].append(handler)

        self._logger.debug(f"Subscribed handler to event: {event_type}")
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         limit: Optional[int] = None) -> List[EventMessage]:
        """Get event history, optionally filtered by type and limited"""
        with self._lock:
            history = self._event_history.copy()
        
        if event_type:
            history = [msg for msg in history if msg.event_type == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._event_history.clear()


class PluginEventManager:
    """Standardized event management for all plugins"""
    
    def __init__(self, event_bus: EventBus, plugin_name: str):
        self.event_bus = event_bus
        self.plugin_name = plugin_name
    
    def emit_started(self, **data):
        """Emit plugin started event with standardized format"""
        event_name = f"{self.plugin_name}.started"
        payload = {
            "plugin": self.plugin_name,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.event_bus.emit(event_name, payload, source=self.plugin_name)
    
    def emit_progress(self, progress: float, message: str = "", **data):
        """Emit plugin progress event with standardized format"""
        event_name = f"{self.plugin_name}.progress"
        payload = {
            "plugin": self.plugin_name,
            "progress": max(0.0, min(100.0, progress)),
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.event_bus.emit(event_name, payload, source=self.plugin_name)
    
    def emit_completed(self, results=None, **data):
        """Emit plugin completed event with standardized format"""
        event_name = f"{self.plugin_name}.completed"
        payload = {
            "plugin": self.plugin_name,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.event_bus.emit(event_name, payload, source=self.plugin_name)
    
    def emit_failed(self, error: Exception, **data):
        """Emit plugin failed event with standardized format"""
        event_name = f"{self.plugin_name}.failed"
        payload = {
            "plugin": self.plugin_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.event_bus.emit(event_name, payload, source=self.plugin_name)


class PipelineEvent:
    """Standard pipeline event types"""
    
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    PIPELINE_CANCELLED = "pipeline.cancelled"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_STEP_FAILED = "workflow.step.failed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    
    # Model events
    MODEL_LOADING_STARTED = "model.loading.started"
    MODEL_LOADED = "model.loaded"
    MODEL_LOADING_FAILED = "model.loading.failed"
    MODEL_ANALYSIS_STARTED = "model.analysis.started"
    MODEL_ANALYSIS_COMPLETED = "model.analysis.completed"
    
    # Compression events
    COMPRESSION_STARTED = "compression.started"
    COMPRESSION_PROGRESS = "compression.progress"
    COMPRESSION_COMPLETED = "compression.completed"
    COMPRESSION_FAILED = "compression.failed"
    
    # Evaluation events
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_PROGRESS = "evaluation.progress"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    
    # System events
    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_FAILED = "plugin.failed"
    CONFIG_LOADED = "config.loaded"
    CONFIG_VALIDATION_FAILED = "config.validation.failed"
    
    # Resource events
    MEMORY_WARNING = "system.memory.warning"
    DISK_SPACE_WARNING = "system.disk.warning"
    GPU_UTILIZATION = "system.gpu.utilization"

    # Calibration events (ASVD/SVD-LLM)
    CALIBRATION_STARTED = "calibration.started"
    CALIBRATION_COMPLETED = "calibration.completed"

    # ASVD events
    ACTIVATION_SCALING_STARTED = "activation_scaling.started"
    ACTIVATION_SCALING_COMPLETED = "activation_scaling.completed"
    PPL_SENSITIVITY_STARTED = "ppl_sensitivity.started"
    PPL_SENSITIVITY_COMPLETED = "ppl_sensitivity.completed"
    BINARY_SEARCH_RANK_STARTED = "binary_search_rank.started"
    BINARY_SEARCH_RANK_COMPLETED = "binary_search_rank.completed"
    FISHER_STARTED = "fisher.started"
    FISHER_COMPLETED = "fisher.completed"

    # SVD-LLM events
    DATA_WHITENING_STARTED = "data_whitening.started"
    DATA_WHITENING_COMPLETED = "data_whitening.completed"
    CLOSED_FORM_UPDATE_STARTED = "closed_form_update.started"
    CLOSED_FORM_UPDATE_COMPLETED = "closed_form_update.completed"
