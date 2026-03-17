"""
LM Evaluation Plugins for Toggle Architecture

This module provides:
- LMHarness: Unified integration with EleutherAI LM Evaluation Harness
- LMEvaluator: Streamlined evaluator built on ModelEvalInterface for
  classification, perplexity, and generic loglikelihood tasks without
  requiring the external lm_eval package.
"""
import time
import os
import math
import torch
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import ModelEvaluationPlugin, ModelEvaluationResult
from .csv_logger import CSVLogger

from ...framework.eval_interface import ModelEvalInterface

# Default sample cap for local dataset adapters; can be overridden via env
# Set TOGGLE_EVAL_DEFAULT_SAMPLES to adjust globally (e.g., in CI)
DEFAULT_NUM_SAMPLES: int = int(os.getenv("TOGGLE_EVAL_DEFAULT_SAMPLES", "64") or 64)

# Optional dependency: EleutherAI LM Evaluation Harness
try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model, MODEL_REGISTRY
    from lm_eval.api.instance import Instance
    from lm_eval import evaluator
    LM_EVAL_AVAILABLE = True
except Exception:  # pragma: no cover - keep module importable without lm_eval
    LM_EVAL_AVAILABLE = False
    # Minimal placeholders to satisfy type references when lm_eval is absent
    class LM:  # type: ignore
        pass

    class Instance:  # type: ignore
        pass

# Safe registration wrapper to avoid double-registration conflicts
def register_model_safe(name: str):
    if 'MODEL_REGISTRY' in globals():
        if name in MODEL_REGISTRY:
            def decorator(cls):
                return cls
            return decorator
        return register_model(name)
    # Fallback no-op decorator
    def decorator(cls):
        return cls
    return decorator


class LMHarness(ModelEvaluationPlugin):
    """
    Unified LM Evaluation Harness plugin for Toggle models
    
    Provides comprehensive evaluation capabilities using the EleutherAI LM Evaluation Harness
    for both compressed and baseline models with extensive task support.
    """
    
    def __init__(self, 
                 tasks: Optional[List[str]] = None,
                 batch_size: int = 1,
                 max_length: int = 2048,
                 device: str = "auto",
                 limit: Optional[int] = None,
                 model_type: str = "auto",
                 compat_mode: Optional[str] = None,
                 compat_plugin_name: Optional[str] = None,
                 backend: str = "hf",
                 hf_model_name: Optional[str] = None,
                 hf_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize LM Evaluation plugin
        
        Args:
            tasks: List of evaluation tasks
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            device: Device to use ("auto", "cuda", "cpu")
            limit: Limit number of examples (for testing)
            model_type: Model type hint ("compressed", "baseline", "auto")
            **kwargs: Additional configuration
        """
        super().__init__(tasks=tasks, **kwargs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.limit = limit
        self.model_type = model_type
        # Back-compat flags to emulate Baseline/Compressed plugins
        self._compat_mode = compat_mode  # 'baseline' | 'compressed' | None
        self._compat_plugin_name = compat_plugin_name or "LMEval"
        # LM Harness backend selection (default to HF backend)
        self.backend = (backend or "hf").strip().lower()
        self.hf_model_name = hf_model_name
        self.hf_kwargs = dict(hf_kwargs) if isinstance(hf_kwargs, dict) else {}
        
        # Initialize CSV logger
        self.csv_logger: Optional[CSVLogger] = None
        
    def do_execute(self, **kwargs):
        """
        Execute evaluation via LM harness for the configured tasks.

        Expected kwargs:
        - model: PreTrainedModel (optional; resolved from context if missing)
        - tokenizer: PreTrainedTokenizer (optional; resolved from context if missing)
        - tasks: Optional[List[str]] to override instance tasks
        - other parameters forwarded to evaluate_task/_harness
        """
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        tasks = kwargs.get("tasks") or self.tasks

        # Attempt to resolve model/tokenizer from context if not provided
        if model is None and getattr(self, '_model_manager', None) is not None and getattr(self, '_context', None) is not None:
            state = getattr(self._context, 'state', None)
            if state is not None and getattr(state, 'model', None) is not None:
                model = state.model
        if tokenizer is None and getattr(self, '_model_manager', None) is not None and getattr(self, '_context', None) is not None:
            state = getattr(self._context, 'state', None)
            if state is not None and getattr(state, 'tokenizer', None) is not None:
                tokenizer = state.tokenizer

        if model is None or tokenizer is None:
            raise ValueError("LMHarness.do_execute requires 'model' and 'tokenizer' (or available via context)")

        extra = {k: v for k, v in kwargs.items() if k not in ("model", "tokenizer", "tasks")}
        return self._execute_impl(model, tokenizer, tasks, **extra)

    def get_supported_tasks(self) -> List[str]:
        """Get comprehensive list of supported evaluation tasks"""
        return [
            # Common Sense Reasoning
            "hellaswag",
            "winogrande", 
            "piqa",
            "siqa",
            "commonsense_qa",
            
            # Reading Comprehension
            "arc_easy",
            "arc_challenge",
            "openbookqa",
            "boolq",
            "race",
            
            # Mathematical Reasoning
            "gsm8k",
            "math_algebra",
            "math_counting_and_probability",
            
            # Knowledge and Facts
            "truthfulqa_mc1",
            "truthfulqa_mc2",
            "mmlu",
            
            # Language Understanding
            "copa",
            "wsc",
            "wsc273",
            
            # Multiple Choice
            "swag",
            "lambada_openai",
            
            # Perplexity
            "wikitext",
            "ptb_new"
        ]
    
    def evaluate_task(self, 
                     model: PreTrainedModel, 
                     tokenizer: PreTrainedTokenizer, 
                     task: str, 
                     **params) -> ModelEvaluationResult:
        """
        Evaluate model on a specific task using LM Evaluation Harness
        
        Args:
            model: Model to evaluate (baseline or compressed)
            tokenizer: Tokenizer for the model
            task: Task name to evaluate
            **params: Task-specific parameters
            
        Returns:
            Comprehensive evaluation result for the task
        """
        if not LM_EVAL_AVAILABLE:
            raise ImportError("lm_eval is not installed. Please install with: pip install lm_eval")
        
        if not self.validate_task(task):
            self.logger.warning(f"Task '{task}' may not be supported by LM Evaluation Harness")

        # Use explicit hint only; no auto-detection
        model_type = self.model_type if self.model_type != "auto" else "unknown"

        # Prepare evaluator call
        call_params = dict(params) if params else {}
        limit = call_params.pop('limit', self.limit)
        # Ensure batch_size isn't duplicated in **kwargs
        call_params.pop('batch_size', None)
        
        start_time = time.time()

        try:
            # Decide backend and build a single simple_evaluate call
            eval_call_kwargs: Dict[str, Any] = {
                "tasks": task,
                "limit": limit,
                "batch_size": self.batch_size,
                # Pass device to simple_evaluate so it becomes additional_config
                # Avoid including device in model_args to prevent duplicate keys
                "device": self.device if self.device else None,
                **call_params,
            }

            if self.backend in ("hf", "hf-causal"):
                # Resolve model name
                resolved_name = self.hf_model_name or getattr(model, 'name_or_path', None)
                if not resolved_name:
                    raise ValueError("LMHarness backend 'hf' requires 'hf_model_name' or model.name_or_path")

                model_args: Dict[str, Any] = {"pretrained": str(resolved_name)}
                # Merge any explicit hf kwargs
                if self.hf_kwargs:
                    model_args.update(self.hf_kwargs)

                eval_call_kwargs.update({
                    "model": self.backend,
                    "model_args": model_args,
                })
            else:
                # Use in-memory adapter backend
                adapter = LMLMHarnessModelAdapter(
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    model_type=model_type,
                )
                eval_call_kwargs.update({
                    "model": adapter,
                })

            results = evaluator.simple_evaluate(**eval_call_kwargs)

            # Extract comprehensive metrics from results
            task_results = results.get('results', {}).get(task, {})

            # Convert to our result format
            metrics = self._extract_metrics(task_results)

            # Count samples from results
            num_samples = results.get('n-samples', {}).get(task, 0)

            # Add model-specific metadata
            additional_info = {
                "model_type": model_type,
                "device": self.device,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "lm_eval_version": getattr(evaluator, '__version__', 'unknown')
            }
            # Back-compat flags: Baseline/Compressed wrappers expect these
            if self._compat_mode == "baseline":
                additional_info["is_baseline"] = True
            elif self._compat_mode == "compressed":
                additional_info["is_compressed"] = True

            # Add compression info if available
            if hasattr(model, 'compression_info'):
                ci = getattr(model, 'compression_info')
                if isinstance(ci, dict):
                    additional_info.update(ci)
                else:
                    additional_info["compression_info"] = str(ci)

            execution_time = time.time() - start_time

            # Log to CSV if logger available
            if self.csv_logger:
                self.csv_logger.log_evaluation_results(
                    model_type=model_type,
                    evaluation_type="lm_eval",
                    plugin_name=self._compat_plugin_name or "LMEval",
                    task_results={task: metrics},
                    evaluation_params={
                        'limit': limit,
                        'batch_size': self.batch_size,
                        'device': self.device,
                        'model_type_hint': self.model_type
                    },
                    execution_time=execution_time
                )

            return ModelEvaluationResult(
                task_name=task,
                metrics=metrics,
                num_samples=num_samples,
                evaluation_time=execution_time,
                model_name=getattr(model, 'name_or_path', model.__class__.__name__),
                additional_info=additional_info
            )

        except Exception as e:
            # Surface configuration/adapter errors to callers (tests expect exceptions)
            self.logger.error(f"LM Evaluation failed for task {task}: {str(e)}")
            raise
    
    
    def _extract_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract and normalize metrics from LM Evaluation results
        
        Args:
            task_results: Raw results from LM Evaluation Harness
            
        Returns:
            Normalized metrics dictionary
        """
        metrics = {}
        
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                metrics[metric_name] = float(metric_value)
            elif isinstance(metric_value, dict):
                # Handle nested metric structures
                if 'acc' in metric_value:
                    metrics[f"{metric_name}_acc"] = float(metric_value['acc'])
                if 'acc_norm' in metric_value:
                    metrics[f"{metric_name}_acc_norm"] = float(metric_value['acc_norm'])
                if 'f1' in metric_value:
                    metrics[f"{metric_name}_f1"] = float(metric_value['f1'])
                if 'exact_match' in metric_value:
                    metrics[f"{metric_name}_em"] = float(metric_value['exact_match'])
                # Add other common metrics
                for key, value in metric_value.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{metric_name}_{key}"] = float(value)
        
        return metrics
    
    # get_task_requirements removed (unused)


@register_model_safe("unified_toggle")
class LMLMHarnessModelAdapter(LM):
    """
    Unified LM Evaluation Harness adapter for Toggle models
    
    Works with both baseline and compressed models, providing comprehensive
    LM Evaluation Harness integration migrated from ToggleModelAdapter.
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 1,
                 max_length: int = 2048,
                 model_type: str = "baseline",
                 **kwargs):
        """
        Initialize unified model adapter
        
        Args:
            model: Model to evaluate (baseline or compressed)
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            model_type: Type of model ("baseline" or "compressed")
        """
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_type = model_type

        # Wrap with standardized evaluation interface to avoid duplication
        self._eval = ModelEvalInterface(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        self._vocab_size = len(self.tokenizer)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        return self._vocab_size
    
    @property
    def eot_token_id(self) -> int:
        """Return end-of-text token id"""
        return getattr(self.tokenizer, 'eos_token_id', None) or getattr(self.tokenizer, 'pad_token_id', 0)
    
    @property
    def max_gen_toks(self) -> int:
        """Return maximum generation tokens"""
        return self.max_length
    
    def tokenize(self, string: str) -> List[int]:
        """Tokenize a string (delegates to ModelEvalInterface)"""
        return self._eval.tokenize(string)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize a list of tokens (delegates to ModelEvalInterface)"""
        return self._eval.detokenize(tokens)
    
    def loglikelihood(self, requests: List[Instance]) -> List[tuple]:
        """
        Calculate log likelihood of target given context.
        Delegates computation to ModelEvalInterface.
        """
        if not requests:
            return []
        # Translate lm_eval Instances to (context, continuation) tuples
        pairs: List[tuple] = []
        for req in requests:
            context, continuation = req.args
            pairs.append((context, continuation))
        return self._eval.loglikelihood(pairs)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """
        Calculate rolling log likelihood for perplexity evaluation.
        Delegates computation to ModelEvalInterface and flattens results.
        """
        if not requests:
            return []
        texts: List[str] = []
        for req in requests:
            # Robust extraction: support lm_eval.Instance(.args), .arguments, or raw strings
            text = None
            if hasattr(req, 'args') and isinstance(req.args, (list, tuple)) and req.args:
                text = req.args[0]
            elif hasattr(req, 'arguments') and isinstance(req.arguments, (list, tuple)) and req.arguments:
                text = req.arguments[0]
            elif isinstance(req, str):
                text = req
            else:
                # Fallback to string conversion
                try:
                    text = str(req)
                except Exception:
                    text = ""
            texts.append(text if isinstance(text, str) else str(text))
        tuple_results = self._eval.loglikelihood_rolling(texts)
        # ModelEvalInterface returns list of (loglik,) tuples; flatten to float
        flattened: List[float] = []
        for item in tuple_results:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                try:
                    flattened.append(float(item[0]))
                except Exception:
                    flattened.append(0.0)
            elif isinstance(item, (int, float)):
                flattened.append(float(item))
            else:
                flattened.append(0.0)
        return flattened
    
    # Removed duplicate rolling implementation; delegated to ModelEvalInterface
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until stopping criteria are met
        
        Uses the same implementation as original ToggleModelAdapter.
        """
        results = []
        
        for req in requests:
            context, gen_kwargs = req.args
            generated = self._generate_single(context, gen_kwargs)
            results.append(generated)
        
        return results
    
    def _generate_single(self, context: str, gen_kwargs: Dict[str, Any]) -> str:
        """Generate text for a single request"""
        # Parse generation arguments
        max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
        until = gen_kwargs.get("until", [self.tokenizer.eos_token])
        temperature = gen_kwargs.get("temperature", 0.0)
        do_sample = temperature > 0.0
        
        # Tokenize input
        input_ids = self.tokenizer.encode(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        input_length = input_ids.size(1)
        
        # Generate
        with torch.no_grad():
            if do_sample:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen_toks,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode generated portion
        generated_ids = output_ids[0, input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stopping criteria
        for stop_token in until:
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]
                break
        
        return generated_text


class LMEvaluator(ModelEvaluationPlugin):
    """
    Evaluation plugin using ModelEvalInterface for standardized evaluation.

    Provides unified evaluation across various NLP tasks while maintaining compatibility
    with existing evaluation infrastructure and configuration systems.
    """

    def __init__(self, tasks: Optional[List[str]] = None, **kwargs):
        """
        Initialize LMEvaluator with task list and configuration.

        Args:
            tasks: List of evaluation tasks to run
            **kwargs: Additional configuration parameters
        """
        super().__init__(tasks=tasks, **kwargs)

        # Load configuration from context if available
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        eval_config = config.get('evaluation', {})

        # Set evaluation parameters
        self.device = eval_config.get('device', 'auto')
        self.batch_size = eval_config.get('batch_size', 1)
        self.max_length = eval_config.get('max_length', 2048)

        # Cache for eval interface to avoid recreation
        self._eval_interface = None
        self._cached_model_id = None

    def get_supported_tasks(self) -> List[str]:
        """
        Get list of supported LM evaluation tasks.

        Returns:
            List of supported task names

        TODO: Think over if here is indeed necessary.
        """

        return [
            # Common Sense Reasoning
            "hellaswag",
            "winogrande",
            "piqa",
            "siqa",

            # Reading Comprehension
            "arc_easy",
            "arc_challenge",
            "openbookqa",
            "boolq",

            # Language Understanding
            "copa",
            "wsc",

            # Perplexity Tasks
            "wikitext",
            "ptb_new"
        ]

    def evaluate_task(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        **params,
    ) -> ModelEvaluationResult:
        """
        Evaluate model on a specific task using ModelEvalInterface.

        Args:
            model: Model to evaluate (baseline or compressed)
            tokenizer: Tokenizer for the model
            task: Task name to evaluate
            **params: Task-specific parameters

        Returns:
            ModelEvaluationResult containing evaluation metrics
        """
        if not self.validate_task(task):
            raise ValueError(f"Task '{task}' is not supported by LMEvaluator")

        start_time = time.time()

        # Create or reuse eval interface
        eval_interface = self._create_eval_interface(model, tokenizer)

        # Determine requested sample count (explicit param overrides config/default)
        requested_num_samples = params.get("num_samples")

        # Get test data for the task from configuration or datasets
        test_data = self._get_task_test_data(task, num_samples=requested_num_samples)

        # Execute task evaluation using eval interface
        if task in [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
            "siqa",
            "openbookqa",
            "boolq",
            "copa",
            "wsc",
        ]:
            metrics = self._evaluate_classification_task(eval_interface, task, test_data, **params)
        elif task in ["wikitext", "ptb_new"]:
            metrics = self._evaluate_perplexity_task(eval_interface, task, test_data, **params)
        else:
            metrics = self._evaluate_generic_task(eval_interface, task, test_data, **params)

        # Calculate evaluation time
        evaluation_time = time.time() - start_time

        # Determine number of samples processed
        num_samples = len(test_data) if test_data else 0

        # Get model name
        model_name = getattr(model, 'name_or_path', model.__class__.__name__)

        # Additional info including model type from eval interface
        additional_info = {
            "model_type": eval_interface.model_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "evaluation_method": "ModelEvalInterface",
        }

        return ModelEvaluationResult(
            task_name=task,
            metrics=metrics,
            num_samples=num_samples,
            evaluation_time=evaluation_time,
            model_name=model_name,
            additional_info=additional_info,
        )

    def _create_eval_interface(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> ModelEvalInterface:
        """
        Create standardized evaluation interface.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model

        Returns:
            ModelEvalInterface instance
        """
        # Create unique model identifier for caching
        model_id = id(model)

        # Reuse cached interface if same model
        if self._eval_interface is not None and self._cached_model_id == model_id:
            return self._eval_interface

        # Create new interface
        self._eval_interface = ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        self._cached_model_id = model_id

        return self._eval_interface

    def _get_task_test_data(self, task: str, num_samples: Optional[int] = None) -> List[Any]:
        """
        Load real test data for a task when available; fall back to config samples.

        - For classification tasks (e.g., hellaswag, arc_easy), returns a list of
          dicts: {"context": str, "choices": List[str], "answer": int}.
        - For perplexity tasks (e.g., wikitext), returns a list of text strings.

        Args:
            task: Task name

        Returns:
            List of test cases for the task, capped by num_samples/default
        """
        # Access config for optional limits/fallbacks
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        eval_cfg = (config or {}).get('evaluation', {})
        test_settings = (config or {}).get('test_settings', {})
        # Determine limit precedence: explicit arg -> config keys -> default
        limit = (
            num_samples
            if isinstance(num_samples, int) and num_samples > 0
            else eval_cfg.get('limit')
            or eval_cfg.get('max_examples')
            or eval_cfg.get('num_samples')
            or eval_cfg.get('default_num_samples')
            or DEFAULT_NUM_SAMPLES
        )

        # Try to import datasets. If unavailable or loading fails, fall back below.
        try:
            from datasets import load_dataset  # type: ignore
        except Exception:
            load_dataset = None  # type: ignore

        def _apply_limit(items: List[Any]) -> List[Any]:
            if isinstance(limit, int) and limit is not None and limit > 0:
                return items[:limit]
            return items

        def _fallback_default() -> List[Any]:
            # Final fallback to existing config-driven samples
            if task in ["hellaswag", "arc_easy", "winogrande"]:
                return test_settings.get(
                    'classification_test_cases',
                    [
                        {
                            "context": "The capital of France is",
                            "choices": [" Paris", " London", " Berlin", " Madrid"],
                            "answer": 0,
                        }
                    ],
                )
            elif task in ["wikitext", "ptb_new"]:
                return test_settings.get(
                    'perplexity_test_cases',
                    [
                        "The quick brown fox jumps over the lazy dog.",
                        "Machine learning is fascinating and powerful.",
                        "Neural networks can compress language models effectively.",
                    ],
                )
            else:
                return test_settings.get(
                    'loglikelihood_test_cases',
                    [
                        {
                            "context": "Hello",
                            "continuation": " world",
                            "expected_loglik": -1.8,
                        }
                    ],
                )

        # If datasets lib not present, use fallback
        if load_dataset is None:
            return _apply_limit(_fallback_default())

        try:
            # Per-task real dataset loaders
            if task == "wikitext":
                # Prefer small, commonly cached split for tests
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts: List[str] = [
                    row["text"] for row in ds
                    if isinstance(row.get("text", None), str) and row["text"].strip()
                ]
                return _apply_limit(texts)

            if task == "ptb_new":
                # Use PTB text-only dataset; field is typically 'sentence'
                ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
                texts = [
                    (row.get("text") or row.get("sentence") or "")
                    for row in ds
                ]
                texts = [t for t in texts if isinstance(t, str) and t.strip()]
                return _apply_limit(texts)

            if task == "arc_easy":
                ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
                # Build classification cases
                cases: List[Dict[str, Any]] = []
                for row in ds:
                    q = row.get("question", "")
                    choices = row.get("choices") or {}
                    texts = list(choices.get("text", []))
                    labels = list(choices.get("label", []))
                    answer_key = row.get("answerKey")
                    try:
                        answer_index = labels.index(answer_key)
                    except Exception:
                        answer_index = 0
                    if texts and isinstance(q, str):
                        cases.append({"context": q, "choices": [f" {c}" for c in texts], "answer": int(answer_index)})
                return _apply_limit(cases)

            if task == "arc_challenge":
                ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
                cases = []
                for row in ds:
                    q = row.get("question", "")
                    choices = row.get("choices") or {}
                    texts = list(choices.get("text", []))
                    labels = list(choices.get("label", []))
                    answer_key = row.get("answerKey")
                    try:
                        answer_index = labels.index(answer_key)
                    except Exception:
                        answer_index = 0
                    if texts and isinstance(q, str):
                        cases.append({"context": q, "choices": [f" {c}" for c in texts], "answer": int(answer_index)})
                return _apply_limit(cases)

            if task == "hellaswag":
                ds = load_dataset("hellaswag", split="validation")
                cases = []
                for row in ds:
                    # Typical fields: 'ctx_a', 'ctx_b', 'endings', 'label'
                    ctx_a = row.get("ctx_a", "")
                    ctx_b = row.get("ctx_b", "")
                    context = (ctx_a + " " + ctx_b).strip()
                    endings = list(row.get("endings", []))
                    label = row.get("label", 0)
                    if endings and isinstance(context, str):
                        try:
                            answer_index = int(label)
                        except Exception:
                            answer_index = 0
                        cases.append({"context": context, "choices": [f" {e}" for e in endings], "answer": answer_index})
                return _apply_limit(cases)

            if task == "winogrande":
                # Use debiased validation split when available
                ds = load_dataset("winogrande", "winogrande_debiased", split="validation")
                cases = []
                for row in ds:
                    sent = row.get("sentence", "")
                    opt1 = row.get("option1", "")
                    opt2 = row.get("option2", "")
                    ans = row.get("answer", "1")
                    # Build a simple context by stripping the blank placeholder
                    # Many rows use '_' to mark the blank; remove it to form context prefix
                    context = str(sent).replace("_", "").strip()
                    choices = [f" {opt1}", f" {opt2}"]
                    try:
                        answer_index = max(0, min(1, int(ans) - 1))
                    except Exception:
                        answer_index = 0
                    if isinstance(context, str) and any(isinstance(c, str) for c in choices):
                        cases.append({"context": context, "choices": choices, "answer": answer_index})
                return _apply_limit(cases)

            if task == "piqa":
                ds = load_dataset("piqa", split="validation")
                cases = []
                for row in ds:
                    goal = row.get("goal", "")
                    sol1 = row.get("sol1", "")
                    sol2 = row.get("sol2", "")
                    label = row.get("label", 0)
                    try:
                        answer_index = int(label)
                    except Exception:
                        answer_index = 0
                    context = str(goal)
                    choices = [f" {sol1}", f" {sol2}"]
                    if isinstance(context, str) and any(isinstance(c, str) for c in choices):
                        cases.append({"context": context, "choices": choices, "answer": answer_index})
                return _apply_limit(cases)

            if task == "boolq":
                ds = load_dataset("boolq", split="validation")
                cases = []
                for row in ds:
                    passage = row.get("passage", "")
                    question = row.get("question", "")
                    answer = row.get("answer", False)
                    # Build a simple prompt-style context
                    context = f"{passage}\nQuestion: {question}\nAnswer:"
                    choices = [" True", " False"]
                    answer_index = 0 if bool(answer) else 1
                    cases.append({"context": context, "choices": choices, "answer": answer_index})
                return _apply_limit(cases)

            if task == "siqa":
                # Social IQa on HF hub is 'social_i_qa'
                ds = load_dataset("social_i_qa", split="validation")
                cases = []
                for row in ds:
                    ctx = row.get("context", "")
                    q = row.get("question", "")
                    a1 = row.get("answerA", "")
                    a2 = row.get("answerB", "")
                    a3 = row.get("answerC", "")
                    label = row.get("label", "1")
                    try:
                        answer_index = max(0, min(2, int(label) - 1))
                    except Exception:
                        answer_index = 0
                    context = f"{ctx} {q}".strip()
                    choices = [f" {a1}", f" {a2}", f" {a3}"]
                    if isinstance(context, str) and any(isinstance(c, str) for c in choices):
                        cases.append({"context": context, "choices": choices, "answer": answer_index})
                return _apply_limit(cases)

            # Other supported classification tasks could be added here similarly

        except Exception:
            # If any dataset load/parsing fails, fall back to config samples
            return _apply_limit(_fallback_default())

        # Unknown task type: fall back
        return _apply_limit(_fallback_default())

    def _evaluate_classification_task(
        self, eval_interface: ModelEvalInterface, task: str, test_data: List[Dict], **params
    ) -> Dict[str, float]:
        """
        Evaluate classification task using loglikelihood comparison.

        Args:
            eval_interface: ModelEvalInterface instance
            task: Task name
            test_data: Test data for the task
            **params: Additional parameters

        Returns:
            Dictionary of metrics
        """
        if not test_data:
            return {"accuracy": 0.0, "num_examples": 0}

        correct = 0
        total = 0

        for example in test_data:
            context = example.get("context", "")
            choices = example.get("choices", [])
            correct_answer = example.get("answer", 0)

            if not choices:
                continue

            # Calculate loglikelihood for each choice
            requests = [(context, choice) for choice in choices]
            results = eval_interface.loglikelihood(requests)

            # Normalize results: accept tuple/list/float/None
            loglikelihoods = []
            for res in (results or []):
                if res is None:
                    loglikelihoods.append(float('-inf'))
                elif isinstance(res, (int, float)):
                    loglikelihoods.append(float(res))
                elif isinstance(res, (list, tuple)) and len(res) > 0:
                    # (loglik, is_greedy) or [loglik, ...]
                    try:
                        loglikelihoods.append(float(res[0]))
                    except Exception:
                        loglikelihoods.append(float('-inf'))
                else:
                    loglikelihoods.append(float('-inf'))
            predicted_answer = loglikelihoods.index(max(loglikelihoods))

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_examples": len(test_data),
        }

    def _evaluate_perplexity_task(
        self, eval_interface: ModelEvalInterface, task: str, test_data: List[str], **params
    ) -> Dict[str, float]:
        """
        Evaluate perplexity task using rolling loglikelihood.

        Args:
            eval_interface: ModelEvalInterface instance
            task: Task name
            test_data: Test data (list of text strings)
            **params: Additional parameters

        Returns:
            Dictionary of metrics
        """
        if not test_data:
            return {"perplexity": float('inf'), "num_examples": 0}

        # Calculate rolling loglikelihood for each text
        results = eval_interface.loglikelihood_rolling(test_data)

        # Calculate perplexity (accept tuple/list/float entries)
        def _extract_ll(x: Any) -> float:
            if x is None:
                return 0.0
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, (list, tuple)) and len(x) > 0:
                try:
                    return float(x[0])
                except Exception:
                    return 0.0
            return 0.0

        total_log_prob = sum(_extract_ll(x) for x in (results or []))
        total_tokens = sum(len(eval_interface.tokenize(text)) for text in test_data)

        if total_tokens == 0:
            perplexity = float('inf')
        else:
            avg_log_prob = total_log_prob / total_tokens
            # avg_log_prob is a Python float; use math.exp for numeric stability
            perplexity = float(math.exp(-avg_log_prob))

        return {
            "perplexity": perplexity,
            "total_log_prob": total_log_prob,
            "total_tokens": total_tokens,
            "num_examples": len(test_data),
        }

    def _evaluate_generic_task(
        self, eval_interface: ModelEvalInterface, task: str, test_data: List[Dict], **params
    ) -> Dict[str, float]:
        """
        Evaluate generic task using loglikelihood.

        Args:
            eval_interface: ModelEvalInterface instance
            task: Task name
            test_data: Test data for the task
            **params: Additional parameters

        Returns:
            Dictionary of metrics
        """
        if not test_data:
            return {"avg_loglikelihood": 0.0, "num_examples": 0}

        # Prepare requests
        requests = []
        expected_logliks = []

        for example in test_data:
            context = example.get("context", "")
            continuation = example.get("continuation", "")
            expected_loglik = example.get("expected_loglik", None)

            requests.append((context, continuation))
            expected_logliks.append(expected_loglik)

        # Calculate loglikelihoods
        results = eval_interface.loglikelihood(requests)
        loglikelihoods = [result[0] for result in results]

        # Calculate metrics
        avg_loglikelihood = sum(loglikelihoods) / len(loglikelihoods) if loglikelihoods else 0.0

        metrics = {
            "avg_loglikelihood": avg_loglikelihood,
            "num_examples": len(test_data),
        }

        # Calculate accuracy against expected values if available
        if any(expected is not None for expected in expected_logliks):
            accurate_predictions = 0
            valid_expectations = 0
            tolerance = params.get("tolerance", 0.5)  # Tolerance for loglikelihood comparison

            for actual, expected in zip(loglikelihoods, expected_logliks):
                if expected is not None:
                    if abs(actual - expected) <= tolerance:
                        accurate_predictions += 1
                    valid_expectations += 1

            if valid_expectations > 0:
                metrics["accuracy"] = accurate_predictions / valid_expectations
                metrics["valid_expectations"] = valid_expectations

        return metrics
