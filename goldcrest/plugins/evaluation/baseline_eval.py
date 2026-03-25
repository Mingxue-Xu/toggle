"""
Baseline Model Evaluation Plugin for Goldcrest Architecture

This plugin provides comprehensive evaluation capabilities for uncompressed (baseline) models:
- BaselineModelEval: LM Evaluation Harness integration (hellaswag, winogrande, etc.)  
- UncompressedModelBenchmark: Performance benchmarking (speed, memory, throughput)

Migrated from goldcrest/lm_eval_adapter.py GoldcrestModelAdapter functionality.
"""
import time
import torch
import psutil
from typing import Any, Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import BaselineModelEvaluationPlugin, ModelEvaluationResult
from .csv_logger import CSVLogger

class UncompressedModelProfile(BaselineModelEvaluationPlugin):
    """
    Uncompressed model performance profiling
    
    Measures performance metrics like inference speed, memory usage, throughput.
    This is for performance profiling, not LM Evaluation Harness tasks.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load configuration from context if available
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        benchmark_config = config.get('benchmark', {})
        
        self.num_inference_samples = benchmark_config.get('num_inference_samples', 100)
        self.batch_size = benchmark_config.get('batch_size', 1)
        self.max_length = benchmark_config.get('max_length', 2048)
        device = benchmark_config.get('device', 'auto')
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CSV logger
        self.csv_logger: Optional[CSVLogger] = None
        
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported benchmarking tasks"""
        return ["inference_speed", "memory_usage", "parameter_count", "throughput"]
    
    def evaluate_task(self, 
                     model: PreTrainedModel, 
                     tokenizer: PreTrainedTokenizer, 
                     task: str, 
                     **params) -> ModelEvaluationResult:
        """Evaluate model on a specific benchmarking task"""
        if not self.validate_task(task):
            raise ValueError(f"Task '{task}' is not supported by UncompressedModelProfile")
        
        start_time = time.time()
        
        try:
            if task == "inference_speed":
                metrics = self._benchmark_inference_speed(model, tokenizer)
            elif task == "memory_usage":
                metrics = self._benchmark_memory_usage(model, tokenizer)
            elif task == "parameter_count":
                metrics = self._benchmark_parameter_count(model)
            elif task == "throughput":
                metrics = self._benchmark_throughput(model, tokenizer)
            else:
                raise ValueError(f"Unknown benchmarking task: {task}")
            
            execution_time = time.time() - start_time
            
            # Log to CSV if logger available
            if self.csv_logger:
                self.csv_logger.log_evaluation_results(
                    model_type="baseline",
                    evaluation_type="profile", 
                    plugin_name="UncompressedModelProfile",
                    task_results={task: metrics},
                    evaluation_params={
                        'num_inference_samples': self.num_inference_samples,
                        'batch_size': self.batch_size,
                        'device': self.device
                    },
                    execution_time=execution_time
                )
            
            return ModelEvaluationResult(
                task_name=task,
                metrics=metrics,
                num_samples=self.num_inference_samples,
                evaluation_time=execution_time,
                model_name=getattr(model, 'name_or_path', model.__class__.__name__),
                additional_info={
                    "is_baseline": True,
                    "evaluation_type": "performance_benchmark",
                    "device": self.device
                }
            )
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed for task {task}: {str(e)}")
            return ModelEvaluationResult(
                task_name=task, metrics={}, num_samples=0,
                evaluation_time=time.time() - start_time,
                model_name=getattr(model, 'name_or_path', model.__class__.__name__),
                additional_info={"is_baseline": True, "error": str(e)}
            )
    
    def _benchmark_inference_speed(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Benchmark inference speed metrics"""
        model.eval()
        model.to(self.device)
        
        # Get sample texts from config or use defaults
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        benchmark_config = config.get('benchmark', {})
        sample_texts = benchmark_config.get('sample_texts', [
            "The quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            "In the beginning was the Word, and the Word was with God."
        ])
        
        sample_texts = (sample_texts * (self.num_inference_samples // len(sample_texts) + 1))[:self.num_inference_samples]
        
        inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=self.max_length).to(self.device)
        
        # Warm up
        warmup_iterations = benchmark_config.get('warmup_iterations', 5)
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(**{k: v[:1] for k, v in inputs.items()})
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(sample_texts), self.batch_size):
                batch_inputs = {k: v[i:i+self.batch_size] for k, v in inputs.items()}
                _ = model(**batch_inputs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.time() - start_time
        
        return {
            "avg_inference_time_ms": (total_time / self.num_inference_samples) * 1000,
            "tokens_per_second": sum(len(tokenizer.encode(text)) for text in sample_texts) / total_time,
            "samples_per_second": self.num_inference_samples / total_time
        }
    
    def _benchmark_memory_usage(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Benchmark memory usage metrics"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        model.to(self.device)
        model.eval()
        
        model_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
        model_gpu_memory = (torch.cuda.memory_allocated() / 1024 / 1024 - initial_gpu_memory) if torch.cuda.is_available() else 0
        
        return {
            "model_memory_mb": model_memory,
            "model_gpu_memory_mb": model_gpu_memory,
            "total_memory_usage_mb": model_memory
        }
    
    def _benchmark_parameter_count(self, model: PreTrainedModel) -> Dict[str, float]:
        """Benchmark parameter count metrics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": float(total_params),
            "trainable_parameters": float(trainable_params),
            "parameters_millions": float(total_params / 1e6)
        }
    
    def _benchmark_throughput(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Benchmark throughput metrics"""
        model.eval()
        model.to(self.device)
        
        # Get throughput texts from config or use defaults
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        benchmark_config = config.get('benchmark', {})
        texts = benchmark_config.get('throughput_texts', [
            "Short text.",
            "This is a medium length text for throughput benchmarking.",
            "This is a longer text with multiple sentences for comprehensive throughput testing."
        ])
        
        test_texts = texts * (self.num_inference_samples // len(texts) + 1)
        test_texts = test_texts[:self.num_inference_samples]
        
        total_tokens = sum(len(tokenizer.encode(text)) for text in test_texts)
        
        start_time = time.time()
        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.device)
                _ = model(**inputs)
        
        total_time = time.time() - start_time
        
        return {
            "throughput_tokens_per_sec": total_tokens / total_time,
            "throughput_samples_per_sec": len(test_texts) / total_time
        }

