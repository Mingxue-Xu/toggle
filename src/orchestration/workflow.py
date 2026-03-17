"""
Workflow System for Toggle Plugin Architecture

This module implements the workflow definition and step execution system
for orchestrating plugin-based pipelines.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum


from ..framework.events import PipelineEvent


class StepStatus(Enum):
    """Workflow step execution status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """Overall workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """
    Individual step in a workflow pipeline
    
    Represents a single plugin execution with its configuration,
    dependencies, and execution conditions.
    """
    name: str
    plugin: str
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry: int = 0
    timeout: Optional[int] = None
    parallel: bool = False
    skip_on_failure: bool = False
    description: Optional[str] = None
    
    # Runtime state
    status: StepStatus = field(default=StepStatus.PENDING, init=False)
    result: Optional[Any] = field(default=None, init=False)
    error: Optional[str] = field(default=None, init=False)
    retry_count: int = field(default=0, init=False)
    execution_time: Optional[float] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate step configuration"""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.plugin:
            raise ValueError("Step must specify a plugin")
        if self.retry < 0:
            raise ValueError("Retry count cannot be negative")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
    
    def is_ready(self, completed_steps: Set[str]) -> bool:
        """
        Check if step is ready to execute based on dependencies
        
        Args:
            completed_steps: Set of completed step names
            
        Returns:
            True if step can execute, False otherwise
        """
        if self.status != StepStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        for dependency in self.depends_on:
            if dependency not in completed_steps:
                return False
        
        return True
    
    def can_retry(self) -> bool:
        """Check if step can be retried"""
        return (self.status == StepStatus.FAILED and 
                self.retry_count < self.retry)
    
    def reset_for_retry(self) -> None:
        """Reset step state for retry attempt"""
        self.status = StepStatus.PENDING
        self.error = None
        self.result = None
        self.execution_time = None
        self.retry_count += 1
    
    def mark_ready(self) -> None:
        """Mark step as ready for execution"""
        self.status = StepStatus.READY
    
    def mark_running(self) -> None:
        """Mark step as currently running"""
        self.status = StepStatus.RUNNING
    
    def mark_completed(self, result: Any, execution_time: float) -> None:
        """Mark step as completed with result"""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.execution_time = execution_time
        self.error = None
    
    def mark_failed(self, error: str, execution_time: Optional[float] = None) -> None:
        """Mark step as failed with error"""
        self.status = StepStatus.FAILED
        self.error = error
        self.execution_time = execution_time
        self.result = None
    
    def mark_skipped(self, reason: str) -> None:
        """Mark step as skipped"""
        self.status = StepStatus.SKIPPED
        self.error = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation"""
        return {
            "name": self.name,
            "plugin": self.plugin,
            "config": self.config,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "retry": self.retry,
            "timeout": self.timeout,
            "parallel": self.parallel,
            "skip_on_failure": self.skip_on_failure,
            "description": self.description,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "execution_time": self.execution_time,
            "error": self.error
        }


@dataclass
class WorkflowSettings:
    """Global workflow settings"""
    device: str = "auto"
    precision: str = "float32"
    memory_limit: Optional[str] = None
    temp_dir: str = "/tmp/toggle_workflow"
    log_level: str = "INFO"
    max_parallel_steps: int = 4
    step_timeout_default: int = 300  # seconds


class Workflow:
    """
    Workflow definition for orchestrating plugin execution
    
    A workflow defines a sequence of steps with dependencies,
    conditions, and execution parameters.
    """
    
    def __init__(self,
                 name: str,
                 steps: List[WorkflowStep],
                 description: Optional[str] = None,
                 version: str = "1.0.0",
                 settings: Optional[WorkflowSettings] = None,
                 plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize workflow
        
        Args:
            name: Workflow name
            steps: List of workflow steps
            description: Workflow description
            version: Workflow version
            settings: Global workflow settings
            plugin_configs: Plugin-specific configurations
        """
        self.name = name
        self.description = description
        self.version = version
        self.settings = settings or WorkflowSettings()
        self.plugin_configs = plugin_configs or {}
        
        # Validate and store steps
        self._validate_steps(steps)
        self.steps = {step.name: step for step in steps}
        self._step_order = [step.name for step in steps]
        
        # Runtime state
        self.status = WorkflowStatus.CREATED
        self.current_step: Optional[str] = None
        self.completed_steps: Set[str] = set()
        self.failed_steps: Set[str] = set()
        
        # Dependency graph
        self._dependency_graph = self._build_dependency_graph()
        
        self.logger = logging.getLogger(f"workflow.{self.name}")
        self.logger.info(f"Workflow '{self.name}' initialized with {len(self.steps)} steps")
    
    def _validate_steps(self, steps: List[WorkflowStep]) -> None:
        """Validate workflow steps and dependencies"""
        if not steps:
            raise ValueError("Workflow must have at least one step")
        
        step_names = {step.name for step in steps}
        
        # Check for duplicate step names
        if len(step_names) != len(steps):
            raise ValueError("Duplicate step names found in workflow")
        
        # Check dependencies exist
        for step in steps:
            for dependency in step.depends_on:
                if dependency not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on non-existent step '{dependency}'")
        
        # Check for circular dependencies
        self._check_circular_dependencies(steps)
    
    def _check_circular_dependencies(self, steps: List[WorkflowStep]) -> None:
        """Check for circular dependencies in workflow"""
        # Build adjacency list
        graph = {step.name: step.depends_on for step in steps}
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_name in graph:
            if step_name not in visited:
                if has_cycle(step_name):
                    raise ValueError("Circular dependency detected in workflow")
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for efficient traversal"""
        graph = {}
        for step_name, step in self.steps.items():
            graph[step_name] = step.depends_on.copy()
        return graph
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """
        Get steps that are ready to execute
        
        Returns:
            List of steps ready for execution
        """
        ready_steps = []
        
        for step_name, step in self.steps.items():
            if step.is_ready(self.completed_steps):
                ready_steps.append(step)
        
        return ready_steps
    
    def get_parallel_ready_steps(self) -> List[WorkflowStep]:
        """
        Get steps that can run in parallel
        
        Returns:
            List of steps that can execute concurrently
        """
        ready_steps = self.get_ready_steps()
        return [step for step in ready_steps if step.parallel]
    
    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get step by name"""
        return self.steps.get(name)
    
    def get_step_results(self) -> Dict[str, Any]:
        """Get results from all completed steps"""
        return {
            name: step.result 
            for name, step in self.steps.items() 
            if step.status == StepStatus.COMPLETED and step.result is not None
        }
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin"""
        return self.plugin_configs.get(plugin_name, {})
    
    def mark_step_completed(self, step_name: str, result: Any, execution_time: float) -> None:
        """Mark a step as completed"""
        if step_name in self.steps:
            self.steps[step_name].mark_completed(result, execution_time)
            self.completed_steps.add(step_name)
            
            self.logger.info(f"Step '{step_name}' completed in {execution_time:.2f}s")
    
    def mark_step_failed(self, step_name: str, error: str, execution_time: Optional[float] = None) -> None:
        """Mark a step as failed"""
        if step_name in self.steps:
            self.steps[step_name].mark_failed(error, execution_time)
            self.failed_steps.add(step_name)
            
            self.logger.error(f"Step '{step_name}' failed: {error}")
    
    def mark_step_skipped(self, step_name: str, reason: str) -> None:
        """Mark a step as skipped"""
        if step_name in self.steps:
            self.steps[step_name].mark_skipped(reason)
            
            self.logger.info(f"Step '{step_name}' skipped: {reason}")
    
    def can_continue(self) -> bool:
        """Check if workflow can continue execution"""
        # If any step failed and doesn't have skip_on_failure, workflow should stop
        for step_name in self.failed_steps:
            step = self.steps[step_name]
            if not step.skip_on_failure and not step.can_retry():
                return False
        
        return True
    
    def is_complete(self) -> bool:
        """Check if workflow execution is complete"""
        total_steps = len(self.steps)
        completed_and_skipped = len(self.completed_steps) + len([
            s for s in self.steps.values() if s.status == StepStatus.SKIPPED
        ])
        
        return completed_and_skipped == total_steps
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        step_summary = {}
        for name, step in self.steps.items():
            step_summary[name] = {
                "status": step.status.value,
                "execution_time": step.execution_time,
                "retry_count": step.retry_count,
                "error": step.error
            }
        
        return {
            "workflow_name": self.name,
            "version": self.version,
            "status": self.status.value,
            "total_steps": len(self.steps),
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "step_details": step_summary
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "settings": {
                "device": self.settings.device,
                "precision": self.settings.precision,
                "memory_limit": self.settings.memory_limit,
                "temp_dir": self.settings.temp_dir,
                "log_level": self.settings.log_level,
                "max_parallel_steps": self.settings.max_parallel_steps,
                "step_timeout_default": self.settings.step_timeout_default
            },
            "steps": [step.to_dict() for step in self.steps.values()],
            "plugin_configs": self.plugin_configs,
            "execution_summary": self.get_execution_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create workflow from dictionary representation"""
        # Parse settings
        settings_data = data.get("settings", {})
        settings = WorkflowSettings(**settings_data)
        
        # Parse steps
        steps_data = data.get("steps", [])
        steps = []
        for step_data in steps_data:
            # Remove runtime fields
            step_config = {k: v for k, v in step_data.items() 
                          if k not in ["status", "result", "error", "retry_count", "execution_time"]}
            steps.append(WorkflowStep(**step_config))
        
        return cls(
            name=data["name"],
            steps=steps,
            description=data.get("description"),
            version=data.get("version", "1.0.0"),
            settings=settings,
            plugin_configs=data.get("plugin_configs", {})
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"Workflow(name='{self.name}', steps={len(self.steps)}, status={self.status.value})"
