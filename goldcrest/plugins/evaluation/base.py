"""
Evaluation Strategy Base Classes for Goldcrest Architecture

This module provides base classes for model evaluation plugins within
the event-driven pipeline system.
"""
import torch
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...framework.plugins import Plugin, PluginMetadata


@dataclass
class ModelEvaluationResult:
    """
    Container for model evaluation results
    """
    task_name: str
    metrics: Dict[str, float]
    num_samples: int
    evaluation_time: float
    model_name: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


class ModelEvaluationPlugin(Plugin):
    """
    Base class for model evaluation plugins
    
    Provides standard interface for evaluating language models on various
    benchmarks and tasks like hellaswag, winogrande, etc.
    """
    
    def __init__(self, tasks: Optional[List[str]] = None, **kwargs):
        """
        Initialize evaluation strategy
        
        Args:
            tasks: List of tasks to evaluate on
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.tasks = tasks or self._get_default_tasks()
        self._evaluation_history = []
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Model evaluation plugin",
            category="evaluation"
        )
    
    def do_execute(
        self,
        context=None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        tasks: Optional[List[str]] = None,
        **params,
    ) -> Dict[str, ModelEvaluationResult]:
        """
        Execute model evaluation
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            tasks: Override tasks to evaluate (uses instance tasks if None)
            **params: Evaluation parameters
            
        Returns:
            Dictionary mapping task names to evaluation results
        """
        import time

        if model is None and self.context:
            state = getattr(self.context, "state", None)
            if state is not None:
                model = getattr(state, "model", None) or state.get("model")
        if tokenizer is None and self.context:
            state = getattr(self.context, "state", None)
            if state is not None:
                tokenizer = getattr(state, "tokenizer", None) or state.get("tokenizer")

        if model is None or tokenizer is None:
            raise ValueError(
                f"{self.__class__.__name__}.do_execute requires 'model' and 'tokenizer' "
                "(or values available in context.state)"
            )

        eval_tasks = tasks or self.tasks
        if isinstance(eval_tasks, str):
            eval_tasks = [eval_tasks]
        results = {}
        
        # Emit evaluation started event
        if self.event_bus:
            self.event_bus.emit(
                "evaluation.started",
                {
                    "plugin": self.name,
                    "tasks": eval_tasks,
                    "model_type": model.__class__.__name__
                },
                source=self.name
            )
        
        total_start_time = time.time()
        
        try:
            # Evaluate each task
            for task in eval_tasks:
                task_start_time = time.time()
                
                self.logger.info(f"Evaluating task: {task}")
                
                # Emit task started event
                if self.event_bus:
                    self.event_bus.emit(
                        "evaluation.task_started",
                        {
                            "plugin": self.name,
                            "task": task
                        },
                        source=self.name
                    )
                
                # Perform evaluation for this task
                task_result = self.evaluate_task(model, tokenizer, task, **params)
                task_result.evaluation_time = time.time() - task_start_time
                if not task_result.model_name:
                    task_result.model_name = getattr(model, 'name_or_path', model.__class__.__name__)
                
                results[task] = task_result
                
                # Emit task completed event
                if self.event_bus:
                    self.event_bus.emit(
                        "evaluation.task_completed",
                        {
                            "plugin": self.name,
                            "task": task,
                            "metrics": task_result.metrics,
                            "evaluation_time": task_result.evaluation_time
                        },
                        source=self.name
                    )
            
            # Store results in context if available
            if self.context:
                self.context.state.evaluation_results = results
            
            # Add to evaluation history
            self._evaluation_history.append({
                "timestamp": time.time(),
                "model_name": results[list(results.keys())[0]].model_name if results else "unknown",
                "tasks": eval_tasks,
                "results": results
            })
            
            total_time = time.time() - total_start_time
            
            # Emit overall completion event
            if self.event_bus:
                self.event_bus.emit(
                    "evaluation.completed",
                    {
                        "plugin": self.name,
                        "tasks": eval_tasks,
                        "total_time": total_time,
                        "task_count": len(results)
                    },
                    source=self.name
                )
            
            return results
            
        except Exception as e:
            if self.event_bus:
                self.event_bus.emit(
                    "evaluation.failed",
                    {
                        "plugin": self.name,
                        "error": str(e)
                    },
                    source=self.name
                )
            raise
    
    @abstractmethod
    def evaluate_task(self, 
                     model: PreTrainedModel, 
                     tokenizer: PreTrainedTokenizer, 
                     task: str, 
                     **params) -> ModelEvaluationResult:
        """
        Evaluate model on a specific task
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            task: Task name to evaluate
            **params: Task-specific parameters
            
        Returns:
            Evaluation result for the task
        """
        pass
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """
        Get list of supported evaluation tasks
        
        Returns:
            List of supported task names
        """
        pass
    
    def _get_default_tasks(self) -> List[str]:
        """
        Get default tasks for this evaluation strategy
        
        Returns:
            List of default task names
        """
        supported = self.get_supported_tasks()
        return supported[:1] if supported else []  # Return first task as default
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return self._evaluation_history.copy()
    
    def validate_task(self, task: str) -> bool:
        """
        Validate if task is supported
        
        Args:
            task: Task name to validate
            
        Returns:
            True if task is supported, False otherwise
        """
        return task in self.get_supported_tasks()
    
    # get_task_requirements removed (unused)


class BaselineModelEvaluationPlugin(ModelEvaluationPlugin):
    """
    Base class for baseline (uncompressed) model evaluation
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Baseline model evaluation plugin",
            category="evaluation"
        )


class CompressedModelEvaluationPlugin(ModelEvaluationPlugin):
    """
    Base class for compressed model evaluation
    """
    
    def __init__(self, compression_info: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize compressed evaluation strategy
        
        Args:
            compression_info: Information about the compression method used
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.compression_info = compression_info or {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Compressed model evaluation plugin",
            category="evaluation"
        )
    
    def do_execute(
        self,
        context=None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        tasks: Optional[List[str]] = None,
        **params,
    ) -> Dict[str, ModelEvaluationResult]:
        """
        Execute compressed model evaluation with compression metadata
        """
        results = super().do_execute(
            context=context,
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            **params,
        )
        
        # Add compression information to results
        for task_result in results.values():
            task_result.additional_info.update({
                "compression_method": self.compression_info.get("method", "unknown"),
                "compression_ratio": self.compression_info.get("compression_ratio"),
                "is_compressed": True
            })
        
        return results
