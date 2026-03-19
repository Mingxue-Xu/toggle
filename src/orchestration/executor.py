"""
Workflow Execution Engine for Toggle Plugin Architecture

This module implements the core workflow execution logic, handling
step scheduling, dependency resolution, condition evaluation, and
retry mechanisms.
"""
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from contextlib import contextmanager


from ..framework.context import PipelineContext
from ..framework.events import PipelineEvent
from ..framework.plugins import PluginRegistry, Plugin

from .workflow import Workflow, WorkflowStep, StepStatus


class ExecutionError(Exception):
    """Exception raised during workflow step execution"""
    pass


class WorkflowExecutor:
    """Main workflow execution engine"""
    
    def __init__(self, context: PipelineContext, plugin_registry: PluginRegistry):
        self.context = context
        self.plugin_registry = plugin_registry
        self.logger = logging.getLogger("workflow_executor")
        self._execution_lock = threading.RLock()
    
    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute a complete workflow
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            Dictionary mapping step names to their results
            
        Raises:
            ExecutionError: If workflow execution fails
        """
        with self._execution_lock:
            self.logger.info(f"Starting workflow execution: {workflow.name}")
            
            workflow_results = {}
            
            while not workflow.is_complete():
                # Get ready steps
                ready_steps = workflow.get_ready_steps()
                
                if not ready_steps:
                    if not workflow.can_continue():
                        break
                    else:
                        # No ready steps but workflow can continue - check for retries
                        retry_steps = self._get_retry_steps(workflow)
                        if retry_steps:
                            ready_steps = retry_steps
                        else:
                            break
                
                # Execute steps (parallel if possible)
                if len(ready_steps) > 1 and any(step.parallel for step in ready_steps):
                    parallel_steps = [step for step in ready_steps if step.parallel]
                    serial_steps = [step for step in ready_steps if not step.parallel]
                    
                    # Execute parallel steps
                    if parallel_steps:
                        parallel_results = self._execute_parallel_steps(parallel_steps, workflow)
                        workflow_results.update(parallel_results)
                    
                    # Execute remaining serial steps
                    for step in serial_steps:
                        result = self.execute_step(step, workflow)
                        if result is not None:
                            workflow_results[step.name] = result
                else:
                    # Execute steps serially
                    for step in ready_steps:
                        result = self.execute_step(step, workflow)
                        if result is not None:
                            workflow_results[step.name] = result
            
            # Check final status
            if not workflow.is_complete() and not workflow.can_continue():
                failed_steps = [name for name, step in workflow.steps.items() 
                              if step.status == StepStatus.FAILED]
                raise ExecutionError(f"Workflow failed due to step failures: {failed_steps}")
            
            self.logger.info(f"Workflow execution completed: {workflow.name}")
            return workflow_results
    
    def execute_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """
        Execute a single workflow step with retry handling
        
        Args:
            step: Workflow step to execute
            workflow: Parent workflow for context
            
        Returns:
            Step execution result
            
        Raises:
            ExecutionError: If step execution fails
        """
        max_attempts = step.retry + 1
        
        for attempt in range(max_attempts):
            try:
                result = self._execute_step_once(step, workflow)
                
                # Mark as completed in workflow
                if step.status == StepStatus.COMPLETED:
                    workflow.mark_step_completed(step.name, result, step.execution_time)
                elif step.status == StepStatus.SKIPPED:
                    workflow.mark_step_skipped(step.name, step.error or "Condition not met")
                
                return result
                
            except ExecutionError as e:
                if attempt < max_attempts - 1:  # Not the last attempt
                    self.logger.warning(f"Step '{step.name}' failed (attempt {attempt + 1}/{max_attempts}), retrying: {str(e)}")
                    step.reset_for_retry()
                    time.sleep(1)  # Brief delay before retry
                else:
                    # Final attempt failed
                    workflow.mark_step_failed(step.name, str(e), step.execution_time)
                    
                    if not step.skip_on_failure:
                        raise
                    else:
                        self.logger.warning(f"Step '{step.name}' failed but continuing due to skip_on_failure")
                        return None
        
        return None
    
    def _execute_step_once(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """
        Execute a single workflow step once
        
        Args:
            step: Workflow step to execute
            workflow: Parent workflow for context
            
        Returns:
            Step execution result
            
        Raises:
            ExecutionError: If step execution fails
        """
        step.mark_running()
        start_time = time.time()
        
        self.context.event_bus.emit(
            PipelineEvent.WORKFLOW_STEP_STARTED,
            {
                "step_name": step.name,
                "plugin": step.plugin,
                "workflow_name": workflow.name
            },
            source="workflow_executor"
        )
        
        self.logger.info(f"Executing step '{step.name}' with plugin '{step.plugin}'")
        
        plugin = None

        try:
            # Check condition before execution
            workflow_results = workflow.get_step_results()
            if not self._evaluate_condition(step.condition, workflow_results):
                reason = f"Condition not met: {step.condition}"
                step.mark_skipped(reason)
                self.logger.info(f"Step '{step.name}' skipped: {reason}")
                return None
            
            # Create plugin instance from workflow-level constructor config only.
            plugin_config = workflow.get_plugin_config(step.plugin)
            plugin = self.plugin_registry.create_plugin(step.plugin, **plugin_config)
            plugin.initialize(self.context)
            
            # Prepare execution parameters
            execution_params = dict(step.config)
            
            # Add model and tokenizer from context if available
            if hasattr(self.context, 'state') and self.context.state:
                if hasattr(self.context.state, 'model') and self.context.state.model is not None:
                    execution_params['model'] = self.context.state.model
                if hasattr(self.context.state, 'tokenizer') and self.context.state.tokenizer is not None:
                    execution_params['tokenizer'] = self.context.state.tokenizer

            # Execute plugin with timeout, forwarding runtime step config.
            execution_params["context"] = self.context
            result = self._execute_with_timeout(plugin, step, **execution_params)
            
            # Record execution time and mark as completed
            execution_time = time.time() - start_time
            step.mark_completed(result, execution_time)
            
            # Emit completion event
            self.context.event_bus.emit(
                PipelineEvent.WORKFLOW_STEP_COMPLETED,
                {
                    "step_name": step.name,
                    "plugin": step.plugin,
                    "workflow_name": workflow.name,
                    "execution_time": execution_time
                },
                source="workflow_executor"
            )
            
            self.logger.info(f"Step '{step.name}' completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            step.mark_failed(error_msg, execution_time)
            
            self.context.event_bus.emit(
                PipelineEvent.WORKFLOW_STEP_FAILED,
                {
                    "step_name": step.name,
                    "plugin": step.plugin,
                    "workflow_name": workflow.name,
                    "error": error_msg,
                    "execution_time": execution_time
                },
                source="workflow_executor"
            )
            
            self.logger.error(f"Step '{step.name}' failed after {execution_time:.2f}s: {error_msg}")
            raise ExecutionError(f"Step '{step.name}' failed: {error_msg}") from e
        
        finally:
            # Cleanup plugin if needed
            if plugin is not None:
                try:
                    plugin.cleanup()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def _execute_with_timeout(self, plugin: Plugin, step: WorkflowStep, **execution_params) -> Any:
        """
        Execute plugin with timeout handling
        
        Args:
            plugin: Plugin to execute
            step: Workflow step configuration
            
        Returns:
            Plugin execution result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        timeout = step.timeout or 300  # Default 5 minutes
        
        # Use thread pool for timeout handling
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(plugin.execute, **execution_params)
            
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise ExecutionError(f"Step execution timed out after {timeout} seconds")
    
    def _evaluate_condition(self, condition: str, workflow_results: Dict[str, Any]) -> bool:
        """
        Evaluate a condition string
        
        Args:
            condition: Condition expression to evaluate
            workflow_results: Results from completed workflow steps
            
        Returns:
            True if condition is met, False otherwise
        """
        if not condition:
            return True
        
        try:
            # Create evaluation context
            eval_context = {
                "results": workflow_results,
                "context": self.context,
                "config": self.context.config
            }
            
            # Simple condition evaluation (in production, use a safer evaluator)
            result = eval(condition, {"__builtins__": {}}, eval_context)
            
            self.logger.debug(f"Condition '{condition}' evaluated to {result}")
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
    def _execute_parallel_steps(self, steps: List[WorkflowStep], workflow: Workflow) -> Dict[str, Any]:
        """Execute multiple steps in parallel"""
        self.logger.info(f"Executing {len(steps)} steps in parallel")
        
        with ThreadPoolExecutor(max_workers=min(len(steps), 4)) as executor:
            # Submit all steps
            future_to_step = {
                executor.submit(self.execute_step, step, workflow): step
                for step in steps
            }
            
            # Collect results
            results = {}
            for future in future_to_step:
                step = future_to_step[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[step.name] = result
                except Exception as e:
                    self.logger.error(f"Parallel step '{step.name}' failed: {str(e)}")
                    # Error already handled in execute_step
            
            return results
    
    def _get_retry_steps(self, workflow: Workflow) -> List[WorkflowStep]:
        """Get steps that can be retried"""
        retry_steps = []
        
        for step in workflow.steps.values():
            if step.can_retry():
                retry_steps.append(step)
        
        return retry_steps
