"""
Unified Strategy Factory for Toggle Plugin Architecture

This module provides a single factory for all strategy types (compression, evaluation, analysis)
eliminating redundancy across plugin types while maintaining clear separation of concerns.
"""
import logging
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class StrategyBase(ABC):
    """Base class for all strategy implementations"""

    def __init__(self, name: Optional[str] = None, **config):
        self.name = name or self.__class__.__name__
        self.config = config
        self.logger = logging.getLogger(f"strategy.{self.name}")
    
    @abstractmethod
    def execute(self, context, **params):
        """Execute the strategy"""
        pass


class LMHarnessStrategy(StrategyBase):
    """Language Model Evaluation Harness strategy"""

    def __init__(self, tasks: Optional[List[str]] = None, **config):
        super().__init__("lm_harness", **config)
        self.tasks = tasks if tasks is not None else []
    
    def execute(self, context, model=None, tokenizer=None, **params):
        """Execute LM Harness evaluation"""
        if not model:
            raise ValueError("Model is required for LM Harness evaluation")
        
        self.logger.info(f"Running LM Harness evaluation on tasks: {self.tasks}")
        
        # Use existing evaluation plugin implementation as an adapter
        try:
            from ..plugins.evaluation.lm_eval import LMHarness
        except Exception as e:
            raise ImportError(
                "LMHarness plugin not available. Install lm_eval or use LMEvaluator"
            ) from e

        # Merge strategy config with any execution-time overrides
        # Recognized LMHarness constructor options we pass through explicitly
        ctor_keys = {
            "device", "batch_size", "max_length", "limit",
            "model_type", "compat_mode", "compat_plugin_name"
        }
        merged_cfg = dict(self.config) if isinstance(self.config, dict) else {}
        for k in ctor_keys:
            if k in params:
                merged_cfg[k] = params[k]

        evaluator = LMHarness(tasks=self.tasks, **merged_cfg)
        results = {}
        # Don't leak constructor options into per-task params
        task_params = {k: v for k, v in params.items() if k not in ctor_keys}
        for t in self.tasks:
            res = evaluator.evaluate_task(model, tokenizer, t, **task_params)
            results[t] = res
        return results
    
    def create_eval_interface(self, model, tokenizer, **params):
        """
        Create ModelEvalInterface for LM Harness evaluation
        
        Args:
            model: Model to create interface for
            tokenizer: Tokenizer for the model
            **params: Additional parameters (device, batch_size, max_length)
            
        Returns:
            ModelEvalInterface configured for LM Harness tasks
        """
        from ..eval_interface import ModelEvalInterface
        
        # Pull from either params (override) or strategy config
        device = params.get('device', self.config.get('device', 'auto') if isinstance(self.config, dict) else 'auto')
        batch_size = params.get('batch_size', self.config.get('batch_size', 1) if isinstance(self.config, dict) else 1)
        max_length = params.get('max_length', self.config.get('max_length', 2048) if isinstance(self.config, dict) else 2048)

        return ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )


class PerplexityStrategy(StrategyBase):
    """Perplexity evaluation strategy"""
    
    def __init__(self, dataset_name: str = "wikitext2", **config):
        super().__init__("perplexity", **config)
        self.dataset_name = dataset_name
    
    def execute(self, context, model=None, tokenizer=None, **params):
        """Execute perplexity evaluation via existing plugin adapter"""
        if not model:
            raise ValueError("Model is required for perplexity evaluation")
        
        self.logger.info(f"Computing perplexity on dataset/task: {self.dataset_name}")
        try:
            from ..plugins.evaluation.lm_eval import LMEvaluator
        except Exception as e:
            raise ImportError(
                "LMEvaluator plugin not available to compute perplexity"
            ) from e
        evaluator = LMEvaluator(tasks=[self.dataset_name], **self.config)
        return {self.dataset_name: evaluator.evaluate_task(model, tokenizer, self.dataset_name, **params)}
    
    def create_eval_interface(self, model, tokenizer, **params):
        """
        Create ModelEvalInterface for perplexity evaluation
        
        Args:
            model: Model to create interface for
            tokenizer: Tokenizer for the model
            **params: Additional parameters (device, batch_size, max_length)
            
        Returns:
            ModelEvalInterface configured for perplexity evaluation
        """
        from ..eval_interface import ModelEvalInterface
        
        return ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=params.get('device', 'auto'),
            batch_size=params.get('batch_size', 1),
            max_length=params.get('max_length', 2048)
        )


class AccuracyStrategy(StrategyBase):
    """Accuracy evaluation strategy"""
    
    def __init__(self, task_name: str, **config):
        super().__init__("accuracy", **config)
        self.task_name = task_name
    
    def execute(self, context, model=None, tokenizer=None, **params):
        """Execute accuracy evaluation via LM harness adapter"""
        if not model:
            raise ValueError("Model is required for accuracy evaluation")
        
        self.logger.info(f"Computing accuracy on task: {self.task_name}")
        try:
            from ..plugins.evaluation.lm_eval import LMHarness
        except Exception as e:
            raise ImportError(
                "LMHarness plugin not available; cannot run accuracy task"
            ) from e
        evaluator = LMHarness(tasks=[self.task_name], **self.config)
        return {self.task_name: evaluator.evaluate_task(model, tokenizer, self.task_name, **params)}
    
    def create_eval_interface(self, model, tokenizer, **params):
        """
        Create ModelEvalInterface for accuracy evaluation
        
        Args:
            model: Model to create interface for
            tokenizer: Tokenizer for the model
            **params: Additional parameters (device, batch_size, max_length)
            
        Returns:
            ModelEvalInterface configured for accuracy evaluation
        """
        from ..eval_interface import ModelEvalInterface
        
        return ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=params.get('device', 'auto'),
            batch_size=params.get('batch_size', 1),
            max_length=params.get('max_length', 2048)
        )

class WeightMetricsAnalysisStrategy(StrategyBase):
    """Weight-only metrics analysis strategy (returns raw per-layer results)."""

    def __init__(self, **config):
        super().__init__("weight_metrics", **config)
        # Expect keys under analysis.metrics/selection/compute in plugin config

    def execute(self, context, model=None, **params):
        if model is None:
            raise ValueError("Model is required for weight metrics analysis")

        # Lazy import to avoid circulars
        from ..plugins.analysis.weight_metrics import WeightMetricsAnalyzer
        from ..plugins.analysis.metric_utils import ExternalMetricsBackend, BasicMetricsBackend

        analysis_cfg = self.config.get("analysis", {}) if isinstance(self.config, dict) else {}
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        compute_cfg = analysis_cfg.get("compute", {})

        ext_import = metrics_cfg.get("import", {}) if isinstance(metrics_cfg, dict) else {}
        backend = ExternalMetricsBackend(
            module_path=ext_import.get("module"),
            file_path=ext_import.get("file"),
            name_prefix=ext_import.get("name_prefix"),
        )
        if not backend.provenance.get("found", False):
            backend = BasicMetricsBackend()

        requested = metrics_cfg.get("names", []) if isinstance(metrics_cfg, dict) else []
        if requested == "all":
            metrics_to_use = list(backend.list_metrics().keys())
        else:
            metrics_to_use = [str(n).lower() for n in (requested or list(backend.list_metrics().keys()))]

        analyzer = WeightMetricsAnalyzer(backend)
        return analyzer.analyze_model(
            model=model,
            metrics_to_use=metrics_to_use,
            selection=selection_cfg,
            compute=compute_cfg,
        )


class ActivationMetricsAnalysisStrategy(StrategyBase):
    """Activation metrics analysis strategy returning raw per-layer results."""

    def __init__(self, **config):
        super().__init__("activation_metrics", **config)

    def execute(self, context, model=None, tokenizer=None, **params):
        if model is None:
            raise ValueError("Model is required for activation metrics analysis")

        analysis_cfg = self.config.get("analysis", {}) if isinstance(self.config, dict) else {}
        metrics_cfg = analysis_cfg.get("metrics", {})
        selection_cfg = analysis_cfg.get("selection", {})
        compute_cfg = analysis_cfg.get("compute", {})
        aggregation_cfg = analysis_cfg.get("aggregation", {})

        from ..plugins.analysis.activation_metrics import ActivationMetricsAnalyzer
        from ..plugins.analysis.metric_utils import ExternalMetricsBackend, BasicMetricsBackend

        ext_import = metrics_cfg.get("import", {}) if isinstance(metrics_cfg, dict) else {}
        backend = ExternalMetricsBackend(
            module_path=ext_import.get("module"),
            file_path=ext_import.get("file"),
            name_prefix=ext_import.get("name_prefix"),
        )
        if not backend.provenance.get("found", False):
            backend = BasicMetricsBackend()

        analyzer = ActivationMetricsAnalyzer()
        analyzer.start_capture(backend=backend, compute_cfg=compute_cfg, reductions_cfg=aggregation_cfg)
        hooks = analyzer.register_hooks(model, selection_cfg)
        # Attempt a minimal dummy forward
        import torch
        from .reproducibility import get_generator

        model.eval()
        with torch.no_grad():
            # Try a simple call with random input for linear-like models
            in_features = None
            for m in model.modules():
                if isinstance(m, torch.nn.Linear):
                    in_features = m.in_features
                    break
            if in_features is None:
                in_features = 8
            # Use seeded generator for reproducibility
            generator = get_generator()
            x = torch.randn(1, in_features, generator=generator)
            _ = model(x)
        hooks.remove_all()

        return analyzer.finalize()


class UnifiedStrategyFactory:
    """
    Single factory for all strategy types, eliminating redundancy across plugins
    
    This factory provides a unified interface for creating compression, evaluation,
    and analysis strategies while maintaining type safety and clear configuration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("unified_strategy_factory")
        
        # Strategy registries - now pointing to actual compression classes
        package_root = self._package_root()
        self._compression_strategies = {
            "tensor_train": f"{package_root}.plugins.compression.tensor_train.TensorTrain",
            "tt": f"{package_root}.plugins.compression.tensor_train.TensorTrain",  # Alias
            "svd": f"{package_root}.plugins.compression.svd.SVD",
            "tucker": f"{package_root}.plugins.compression.tucker.Tucker",
            "cp": f"{package_root}.plugins.compression.cp.CP",
            "candecomp": f"{package_root}.plugins.compression.cp.CP",  # Alias
            # ASVD and SVD-LLM plugins
            "calibration_collector": f"{package_root}.plugins.compression.calibration_collector.CalibrationCollectorPlugin",
            "activation_scaling": f"{package_root}.plugins.compression.svd_activation_scaling.ActivationScalingPlugin",
            "data_whitening": f"{package_root}.plugins.compression.svd_data_whitening.DataWhiteningPlugin",
            "closed_form_update": f"{package_root}.plugins.compression.svd_closed_form_update.ClosedFormUpdatePlugin",
            "ppl_sensitivity": f"{package_root}.plugins.compression.svd_ppl_sensitivity.PPLSensitivityPlugin",
            "binary_search_rank": f"{package_root}.plugins.compression.svd_binary_search_rank.BinarySearchRankPlugin",
        }
        
        self._evaluation_strategies = {
            "lm_harness": LMHarnessStrategy,
            "harness": LMHarnessStrategy,  # Alias
            "perplexity": PerplexityStrategy,
            "ppl": PerplexityStrategy,  # Alias
            "accuracy": AccuracyStrategy,
            "acc": AccuracyStrategy  # Alias
        }
        
        self._analysis_strategies = {
            "weights": WeightMetricsAnalysisStrategy,
            "weight_metrics": WeightMetricsAnalysisStrategy,
            "activations": ActivationMetricsAnalysisStrategy,
            "activation_metrics": ActivationMetricsAnalysisStrategy
        }

    @staticmethod
    def _package_root() -> str:
        package = __package__ or ""
        if package.endswith(".framework"):
            return package.rsplit(".", 1)[0]
        if package:
            return package
        return "toggle.src"
    
    def get_compression_strategy(self, method: str, **params):
        """
        Get compression strategy instance - now returns actual compression class
        
        Args:
            method: Compression method name
            **params: Strategy-specific parameters
            
        Returns:
            Configured compression instance (unified tensor/strategy interface)
            
        Raises:
            ValueError: If method is not supported
        """
        method = method.lower().strip()
        
        if method not in self._compression_strategies:
            available = list(self._compression_strategies.keys())
            raise ValueError(f"Unsupported compression method '{method}'. Available: {available}")
        
        # Resolve registered strategy: supports import path or direct class
        entry = self._compression_strategies[method]
        if isinstance(entry, str):
            module_name, class_name = entry.rsplit('.', 1)
            try:
                module = importlib.import_module(module_name)
                strategy_class = getattr(module, class_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {entry}: {e}")
            except AttributeError as e:
                raise AttributeError(f"Class {class_name} not found in {module_name}: {e}")
        elif isinstance(entry, type) and issubclass(entry, StrategyBase):
            strategy_class = entry
        else:
            raise TypeError("Invalid compression strategy registry entry")
        
        # Validate required parameters based on strategy type
        self._validate_compression_params(method, params)
        
        self.logger.info(f"Creating {method} compression strategy")
        return strategy_class(**params)
    
    def _validate_compression_params(self, method: str, params: Dict[str, Any]):
        """Validate required parameters for compression methods"""
        if method in ["tensor_train", "tt"]:
            if "tensor_ranks" not in params:
                raise ValueError("tensor_ranks parameter required for Tensor-Train compression")
        elif method == "svd":
            if "rank" not in params and "preserve_energy" not in params:
                raise ValueError("Either rank or preserve_energy parameter required for SVD compression")
        elif method == "tucker":
            if "tucker_ranks" not in params:
                raise ValueError("tucker_ranks parameter required for Tucker compression")
        elif method in ["cp", "candecomp"]:
            if "cp_rank" not in params:
                raise ValueError("cp_rank parameter required for CP compression")
    
    def get_evaluation_strategy(self, tasks: Union[List[str], str], **params) -> StrategyBase:
        """
        Get evaluation strategy instance with auto-detection
        
        Args:
            tasks: Task names or single task name
            **params: Strategy-specific parameters
            
        Returns:
            Configured evaluation strategy instance
        """
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks:
            raise ValueError("No suitable evaluation strategy for empty task list")
        
        # Auto-detect strategy based on task names
        if any(task.lower() in ["perplexity", "ppl"] for task in tasks):
            self.logger.info("Auto-detected perplexity evaluation strategy")
            dataset_name = params.get("dataset_name", "wikitext2")
            return PerplexityStrategy(dataset_name=dataset_name, **params)
        
        # Check for accuracy tasks
        accuracy_tasks = ["accuracy", "acc"]
        if any(task.lower() in accuracy_tasks for task in tasks):
            task_name = tasks[0]  # Use first task
            self.logger.info(f"Auto-detected accuracy evaluation strategy for task: {task_name}")
            return AccuracyStrategy(task_name=task_name, **params)
        
        # Default to LM Harness for other tasks
        self.logger.info("Using LM Harness evaluation strategy")
        return LMHarnessStrategy(tasks=tasks, **params)
    
    def get_analysis_strategy(self, analysis_type: str, **params) -> StrategyBase:
        """
        Get analysis strategy instance
        
        Args:
            analysis_type: Analysis method name
            **params: Strategy-specific parameters
            
        Returns:
            Configured analysis strategy instance
            
        Raises:
            ValueError: If analysis type is not supported
        """
        analysis_type = analysis_type.lower().strip()
        
        if analysis_type not in self._analysis_strategies:
            available = list(self._analysis_strategies.keys())
            raise ValueError(f"Unsupported analysis type '{analysis_type}'. Available: {available}")
        
        strategy_class = self._analysis_strategies[analysis_type]
        
        self.logger.info(f"Creating {analysis_type} analysis strategy")
        return strategy_class(**params)
    
    def list_compression_methods(self) -> List[str]:
        """Get list of available compression methods"""
        return list(self._compression_strategies.keys())
    
    def list_evaluation_strategies(self) -> List[str]:
        """Get list of available evaluation strategies"""
        return list(self._evaluation_strategies.keys())
    
    def list_analysis_types(self) -> List[str]:
        """Get list of available analysis types"""
        return list(self._analysis_strategies.keys())
    
    def register_compression_strategy(self, name: str, strategy):
        """Register a custom compression strategy.
        Accepts a StrategyBase subclass or an import path string.
        """
        if isinstance(strategy, str):
            self._compression_strategies[name.lower()] = strategy
        elif isinstance(strategy, type) and issubclass(strategy, StrategyBase):
            self._compression_strategies[name.lower()] = strategy
        else:
            raise TypeError("Strategy must inherit from StrategyBase or be an import path string")
        self.logger.info(f"Registered custom compression strategy: {name}")

    def get_strategy_info(self, kind: Optional[str] = None) -> Dict[str, Any]:
        """Return a summary of available strategies.
        kind: 'compression' | 'evaluation' | 'analysis' | None for all
        """
        info = {
            "compression": {"methods": self.list_compression_methods()},
            "evaluation": {"strategies": self.list_evaluation_strategies()},
            "analysis": {"types": self.list_analysis_types()},
        }
        if kind is None:
            return info
        kind = kind.lower()
        if kind not in info:
            raise ValueError("Unknown strategy type")
        return info[kind]
    
    def register_evaluation_strategy(self, name: str, strategy_class: type):
        """Register a custom evaluation strategy"""
        if not issubclass(strategy_class, StrategyBase):
            raise TypeError("Strategy class must inherit from StrategyBase")
        
        self._evaluation_strategies[name.lower()] = strategy_class
        self.logger.info(f"Registered custom evaluation strategy: {name}")
    
    def register_analysis_strategy(self, name: str, strategy_class: type):
        """Register a custom analysis strategy"""
        if not issubclass(strategy_class, StrategyBase):
            raise TypeError("Strategy class must inherit from StrategyBase")
        
        self._analysis_strategies[name.lower()] = strategy_class
        self.logger.info(f"Registered custom analysis strategy: {name}")

## Removed legacy compatibility aliases (TensorTrainStrategy, SVDStrategy, TuckerStrategy, CPStrategy)
## Factory now imports real compression classes dynamically.
