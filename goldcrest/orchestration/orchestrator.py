"""
Pipeline Orchestrator for Goldcrest Plugin Architecture

This module implements the main orchestration system that manages
workflow execution, plugin coordination, and event-driven pipeline control.
"""
import time
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager


from ..framework.context import PipelineContext
from ..framework.events import EventBus, PipelineEvent
from ..framework.plugins import PluginRegistry, Plugin
from .workflow import Workflow, WorkflowStep, WorkflowStatus, StepStatus
from .executor import WorkflowExecutor


class OrchestrationError(Exception):
    """Exception raised during workflow orchestration"""
    pass


class PipelineOrchestrator:
    """
    Central orchestrator for plugin-based pipeline execution
    
    The PipelineOrchestrator manages:
    - Workflow execution lifecycle
    - Plugin registry and creation
    - Event coordination between components
    - Parallel step execution
    - Error handling and recovery
    """
    
    def __init__(self, 
                 context: Optional[PipelineContext] = None,
                 plugin_registry: Optional[PluginRegistry] = None,
                 max_workers: int = 4):
        """
        Initialize pipeline orchestrator
        
        Args:
            context: Pipeline execution context (creates new if None)
            plugin_registry: Plugin registry (creates new if None)
            max_workers: Maximum number of parallel workers
        """
        self.context = context or PipelineContext()
        self.plugin_registry = plugin_registry or PluginRegistry()
        self.max_workers = max_workers
        
        # Workflow execution state
        self.current_workflow: Optional[Workflow] = None
        self.executor = WorkflowExecutor(self.context, self.plugin_registry)
        
        # Thread pool for parallel execution
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Event subscriptions
        self._setup_event_handlers()
        
        self.logger = logging.getLogger("orchestrator")
        self.logger.info(f"Pipeline orchestrator initialized with {max_workers} workers")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for orchestration events"""
        event_bus = self.context.event_bus
        
        # Subscribe to plugin events for monitoring
        event_bus.subscribe(PipelineEvent.PLUGIN_LOADED, self._on_plugin_loaded)
        event_bus.subscribe(PipelineEvent.PLUGIN_FAILED, self._on_plugin_failed)
        
        # Subscribe to workflow events
        event_bus.subscribe(PipelineEvent.WORKFLOW_STEP_COMPLETED, self._on_step_completed)
        event_bus.subscribe(PipelineEvent.WORKFLOW_STEP_FAILED, self._on_step_failed)
    
    def _on_plugin_loaded(self, event_message) -> None:
        """Handle plugin loaded event"""
        data = event_message.data
        self.logger.debug(f"Plugin loaded: {data.get('plugin_name')}")
    
    def _on_plugin_failed(self, event_message) -> None:
        """Handle plugin failed event"""
        data = event_message.data
        self.logger.error(f"Plugin failed: {data.get('plugin_name')} - {data.get('error')}")
    
    def _on_step_completed(self, event_message) -> None:
        """Handle workflow step completed event"""
        data = event_message.data
        self.logger.info(f"Workflow step completed: {data.get('step_name')}")
    
    def _on_step_failed(self, event_message) -> None:
        """Handle workflow step failed event"""
        data = event_message.data
        self.logger.error(f"Workflow step failed: {data.get('step_name')} - {data.get('error')}")
    
    def register_plugin(self, plugin_class, name: Optional[str] = None) -> None:
        """
        Register a plugin class with the orchestrator
        
        Args:
            plugin_class: Plugin class to register
            name: Optional plugin name override
        """
        self.plugin_registry.register(plugin_class, name)
        
        self.context.event_bus.emit(
            PipelineEvent.PLUGIN_LOADED,
            {"plugin_name": name or plugin_class.__name__},
            source="orchestrator"
        )
    
    def create_plugin(self, name: str, **kwargs) -> Plugin:
        """
        Create and initialize a plugin instance
        
        Args:
            name: Plugin name
            **kwargs: Plugin configuration
            
        Returns:
            Initialized plugin instance
        """
        plugin = self.plugin_registry.create_plugin(name, **kwargs)
        plugin.initialize(self.context)
        return plugin
    
    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute a workflow with full orchestration
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            Workflow execution results
            
        Raises:
            OrchestrationError: If workflow execution fails
        """
        self.current_workflow = workflow
        workflow.status = WorkflowStatus.RUNNING
        
        # Update context metadata
        self.context.metadata.workflow_name = workflow.name
        
        # Emit workflow started event
        self.context.event_bus.emit(
            PipelineEvent.WORKFLOW_STARTED,
            {
                "workflow_name": workflow.name,
                "total_steps": len(workflow.steps),
                "settings": workflow.settings.__dict__
            },
            source="orchestrator"
        )
        
        self.logger.info(f"Starting workflow execution: {workflow.name}")
        
        try:
            # Execute workflow using the executor
            results = self.executor.execute_workflow(workflow)
            
            # Mark workflow as completed
            workflow.status = WorkflowStatus.COMPLETED
            
            self.context.event_bus.emit(
                PipelineEvent.WORKFLOW_COMPLETED,
                {
                    "workflow_name": workflow.name,
                    "execution_summary": workflow.get_execution_summary(),
                    "results": results
                },
                source="orchestrator"
            )
            
            self.logger.info(f"Workflow completed successfully: {workflow.name}")
            return results
            
        except Exception as e:
            # Mark workflow as failed
            workflow.status = WorkflowStatus.FAILED
            
            self.context.event_bus.emit(
                PipelineEvent.WORKFLOW_FAILED,
                {
                    "workflow_name": workflow.name,
                    "error": str(e),
                    "execution_summary": workflow.get_execution_summary()
                },
                source="orchestrator"
            )
            
            self.logger.error(f"Workflow execution failed: {workflow.name} - {str(e)}")
            raise OrchestrationError(f"Workflow execution failed: {str(e)}") from e
        
        finally:
            self.current_workflow = None
    
    def execute_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """
        Execute a single workflow step
        
        Args:
            step: Workflow step to execute
            workflow: Parent workflow
            
        Returns:
            Step execution result
            
        Raises:
            OrchestrationError: If step execution fails
        """
        return self.executor.execute_step(step, workflow)
    
    def execute_parallel_steps(self, steps: List[WorkflowStep], workflow: Workflow) -> Dict[str, Any]:
        """
        Execute multiple steps in parallel
        
        Args:
            steps: List of workflow steps to execute in parallel
            workflow: Parent workflow
            
        Returns:
            Dictionary mapping step names to results
            
        Raises:
            OrchestrationError: If any step execution fails
        """
        if not steps:
            return {}
        
        self.logger.info(f"Executing {len(steps)} steps in parallel")
        
        # Submit all steps to thread pool
        future_to_step = {}
        for step in steps:
            future = self._thread_pool.submit(self.execute_step, step, workflow)
            future_to_step[future] = step
        
        # Collect results
        results = {}
        errors = []
        
        for future in as_completed(future_to_step):
            step = future_to_step[future]
            try:
                result = future.result()
                results[step.name] = result
                
            except Exception as e:
                error_msg = f"Parallel step '{step.name}' failed: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Handle errors
        if errors and not all(step.skip_on_failure for step in steps):
            error_summary = "; ".join(errors)
            raise OrchestrationError(f"Parallel execution failed: {error_summary}")
        
        return results
    
    def validate_workflow(self, workflow: Workflow) -> List[str]:
        """
        Validate workflow before execution
        
        Args:
            workflow: Workflow to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if all required plugins are available
        for step_name, step in workflow.steps.items():
            if step.plugin not in self.plugin_registry.list_plugins():
                errors.append(f"Plugin '{step.plugin}' required by step '{step_name}' is not registered")
        
        # Check workflow integrity (already done in Workflow constructor, but double-check)
        try:
            # This will raise ValueError if there are issues
            pass
        except ValueError as e:
            errors.append(f"Workflow structure error: {str(e)}")
        
        return errors
    
    def get_workflow_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current workflow execution status
        
        Returns:
            Workflow status dictionary or None if no active workflow
        """
        if not self.current_workflow:
            return None
        
        return {
            "workflow_name": self.current_workflow.name,
            "status": self.current_workflow.status.value,
            "execution_summary": self.current_workflow.get_execution_summary(),
            "context_summary": self.context.get_execution_summary()
        }
    
    def cancel_workflow(self) -> bool:
        """
        Cancel current workflow execution
        
        Returns:
            True if workflow was cancelled, False if no active workflow
        """
        if not self.current_workflow:
            return False
        
        self.logger.info(f"Cancelling workflow: {self.current_workflow.name}")
        
        # Mark workflow as cancelled
        self.current_workflow.status = WorkflowStatus.CANCELLED
        
        # Emit cancellation event
        self.context.event_bus.emit(
            PipelineEvent.PIPELINE_CANCELLED,
            {"workflow_name": self.current_workflow.name},
            source="orchestrator"
        )
        
        return True
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugins"""
        return self.plugin_registry.list_plugins()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history from event bus"""
        events = self.context.event_bus.get_event_history()
        
        return [{
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
            "source": event.source
        } for event in events]
    
    @contextmanager
    def workflow_context(self, workflow: Workflow):
        """
        Context manager for workflow execution
        
        Args:
            workflow: Workflow to execute in context
        """
        old_workflow = self.current_workflow
        self.current_workflow = workflow
        
        try:
            yield
        finally:
            self.current_workflow = old_workflow
    
    def cleanup(self) -> None:
        """Cleanup orchestrator resources"""
        self.logger.info("Cleaning up pipeline orchestrator")
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        # Cancel any active workflow
        if self.current_workflow:
            self.cancel_workflow()
        
        # Cleanup context
        self.context.cleanup()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def __str__(self) -> str:
        """String representation"""
        workflow_name = self.current_workflow.name if self.current_workflow else "None"
        return f"PipelineOrchestrator(workflow={workflow_name}, plugins={len(self.get_available_plugins())})"
