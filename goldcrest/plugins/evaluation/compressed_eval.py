"""
Compressed Model Evaluation Plugin for Goldcrest Architecture

This plugin provides evaluation capabilities for compressed models:
- CompressedLMEval: LM Evaluation Harness integration for compressed models
- CompressedModelProfile: Performance profiling for compressed models

Specialized for tensor-decomposed models from the Goldcrest framework.
"""
import time
import torch
import psutil
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import CompressedModelEvaluationPlugin, ModelEvaluationResult
from .lm_eval import LMHarness, LM, Instance, register_model_safe
from .csv_logger import CSVLogger

@register_model_safe("compressed_goldcrest")
class CompressedGoldcrestAdapter(LM):
    """
    LM Evaluation Harness adapter for compressed Goldcrest models
    
    Specialized adapter for tensor-decomposed models (tGPT2LMHeadModel, etc.)
    with compression-aware evaluation capabilities.
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 1,
                 max_length: int = 2048,
                 compression_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize compressed model adapter
        
        Args:
            model: Compressed model to evaluate (tGPT2LMHeadModel, etc.)
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            compression_info: Information about compression used
        """
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.compression_info = compression_info or {}
        
        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model.to(device)
        
        # Set up tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self._vocab_size = len(self.tokenizer)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        return self._vocab_size
    
    @property
    def eot_token_id(self) -> int:
        """Return end-of-text token id"""
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self) -> int:
        # Deprecated adapter: retained temporarily for backward compatibility.
        return self.max_length
    
    def tokenize(self, string: str) -> List[int]:
        """Tokenize a string"""
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize a list of tokens"""
        return self.tokenizer.decode(tokens)
    
    def loglikelihood(self, requests: List[Instance]) -> List[tuple]:
        """Calculate log likelihood of target given context for compressed model"""
        results = []
        
        # Process requests in batches with compressed model considerations
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_loglikelihood_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_loglikelihood_batch(self, batch: List[Instance]) -> List[tuple]:
        """Process a batch of loglikelihood requests for compressed model"""
        import torch.nn.functional as F
        
        # Extract context and continuation from Instance objects
        contexts = []
        continuations = []
        
        for req in batch:
            context, continuation = req.args
            contexts.append(context)
            continuations.append(continuation)
        
        # Prepare inputs
        full_texts = [ctx + cont for ctx, cont in zip(contexts, continuations)]
        context_lengths = [len(self.tokenizer.encode(ctx, add_special_tokens=False)) for ctx in contexts]
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        results = []
        
        with torch.no_grad():
            # Run inference with compressed model
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            for idx, (ctx_len, full_text) in enumerate(zip(context_lengths, full_texts)):
                # Get target token positions
                full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
                target_tokens = full_tokens[ctx_len:]
                
                if len(target_tokens) == 0:
                    results.append((0.0, True))
                    continue
                
                # Calculate log probabilities for target tokens
                sequence_logits = logits[idx]
                sequence_tokens = inputs.input_ids[idx]
                
                # Find the actual positions in the padded sequence
                non_pad_length = (sequence_tokens != self.tokenizer.pad_token_id).sum().item()
                start_idx = non_pad_length - len(full_tokens)
                target_start = start_idx + ctx_len
                target_end = start_idx + len(full_tokens)
                
                if target_start >= sequence_logits.size(0) - 1:
                    results.append((0.0, True))
                    continue
                
                # Get log probabilities
                target_logits = sequence_logits[target_start:target_end-1]
                target_token_ids = sequence_tokens[target_start+1:target_end]
                
                log_probs = F.log_softmax(target_logits, dim=-1)
                target_log_probs = log_probs.gather(1, target_token_ids.unsqueeze(-1)).squeeze(-1)
                
                total_log_prob = target_log_probs.sum().item()
                
                # Check if this was the greedy choice
                greedy_tokens = target_logits.argmax(dim=-1)
                is_greedy = torch.equal(greedy_tokens, target_token_ids)
                
                results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[tuple]:
        """Calculate rolling log likelihood for perplexity evaluation on compressed model"""
        import torch.nn.functional as F
        
        results = []
        
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_rolling_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_rolling_batch(self, batch: List[Instance]) -> List[tuple]:
        """Process a batch of rolling loglikelihood requests for compressed model"""
        import torch.nn.functional as F
        
        # Extract context strings from Instance objects
        contexts = [req.args[0] for req in batch]
        
        # Tokenize
        inputs = self.tokenizer(
            contexts,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        results = []
        
        with torch.no_grad():
            # Run inference with compressed model
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            for idx, text in enumerate(contexts):
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                
                if len(tokens) <= 1:
                    results.append((0.0,))
                    continue
                
                # Get sequence logits and tokens
                sequence_logits = logits[idx]
                sequence_tokens = inputs.input_ids[idx]
                
                # Find non-padded portion
                non_pad_mask = sequence_tokens != self.tokenizer.pad_token_id
                non_pad_length = non_pad_mask.sum().item()
                
                if non_pad_length <= 1:
                    results.append((0.0,))
                    continue
                
                # Calculate rolling log likelihood
                valid_logits = sequence_logits[:non_pad_length-1]
                valid_targets = sequence_tokens[1:non_pad_length]
                
                log_probs = F.log_softmax(valid_logits, dim=-1)
                target_log_probs = log_probs.gather(1, valid_targets.unsqueeze(-1)).squeeze(-1)
                
                total_log_prob = target_log_probs.sum().item()
                results.append((total_log_prob,))
        
        return results
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until stopping criteria are met using compressed model"""
        results = []
        
        for req in requests:
            context, gen_kwargs = req.args
            generated = self._generate_single(context, gen_kwargs)
            results.append(generated)
        
        return results
    
    def _generate_single(self, context: str, gen_kwargs: Dict[str, Any]) -> str:
        """Generate text for a single request using compressed model"""
        # Parse generation arguments
        max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
        until = gen_kwargs.get("until", [self.tokenizer.eos_token])
        temperature = gen_kwargs.get("temperature", 0.0)
        do_sample = temperature > 0.0
        
        # Tokenize input
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        input_length = input_ids.size(1)
        
        # Generate using compressed model
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
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get compression information for this adapter"""
        return self.compression_info.copy()


class CompressedModelProfile(CompressedModelEvaluationPlugin):
    """
    Compressed model performance profiling
    
    Measures performance metrics for compressed models with compression-specific analysis.
    """
    
    def __init__(self, compression_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(compression_info=compression_info, **kwargs)
        
        # Load configuration from context if available
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        profile_config = config.get('profile', {})
        
        self.num_inference_samples = profile_config.get('num_inference_samples', 100)
        self.batch_size = profile_config.get('batch_size', 1)
        self.max_length = profile_config.get('max_length', 2048)
        device = profile_config.get('device', 'auto')
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CSV logger
        self.csv_logger: Optional[CSVLogger] = None
        
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported profiling tasks"""
        return ["inference_speed", "memory_usage", "parameter_count", "compression_efficiency"]
    
    def evaluate_task(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                     task: str, **params) -> ModelEvaluationResult:
        """Evaluate compressed model on profiling task"""
        if not self.validate_task(task):
            raise ValueError(f"Task '{task}' is not supported by CompressedModelProfile")
        
        start_time = time.time()
        
        try:
            if task == "inference_speed":
                metrics = self._profile_inference_speed(model, tokenizer)
            elif task == "memory_usage":
                metrics = self._profile_memory_usage(model, tokenizer)
            elif task == "parameter_count":
                metrics = self._profile_parameter_count(model)
            elif task == "compression_efficiency":
                metrics = self._profile_compression_efficiency(model)
            else:
                raise ValueError(f"Unknown profiling task: {task}")
            
            execution_time = time.time() - start_time
            
            # Log to CSV if logger available
            if self.csv_logger:
                self.csv_logger.log_evaluation_results(
                    model_type="compressed",
                    evaluation_type="profile",
                    plugin_name="CompressedModelProfile", 
                    task_results={task: metrics},
                    evaluation_params={
                        'num_inference_samples': self.num_inference_samples,
                        'batch_size': self.batch_size,
                        'device': self.device,
                        'compression_method': self.compression_info.get("method", "unknown")
                    },
                    execution_time=execution_time
                )
            
            return ModelEvaluationResult(
                task_name=task, metrics=metrics, num_samples=self.num_inference_samples,
                evaluation_time=execution_time,
                model_name=getattr(model, 'name_or_path', model.__class__.__name__),
                additional_info={
                    "is_compressed": True, "evaluation_type": "performance_profile",
                    "device": self.device, "compression_method": self.compression_info.get("method", "unknown")
                }
            )
            
        except Exception as e:
            self.logger.error(f"Compressed profiling failed for task {task}: {str(e)}")
            return ModelEvaluationResult(
                task_name=task, metrics={}, num_samples=0,
                evaluation_time=time.time() - start_time,
                model_name=getattr(model, 'name_or_path', model.__class__.__name__),
                additional_info={"is_compressed": True, "error": str(e)}
            )
    
    def _profile_inference_speed(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Profile inference speed with compression overhead analysis"""
        model.eval().to(self.device)
        
        config = getattr(self.context, 'config', {}) if hasattr(self, 'context') and self.context else {}
        profile_config = config.get('profile', {})
        sample_texts = profile_config.get('sample_texts', [
            "Compressed model inference test.", "Tensor decomposition performance analysis."
        ])
        
        sample_texts = (sample_texts * (self.num_inference_samples // len(sample_texts) + 1))[:self.num_inference_samples]
        inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=self.max_length).to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(profile_config.get('warmup_iterations', 3)):
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
            "compressed_avg_inference_time_ms": (total_time / self.num_inference_samples) * 1000,
            "compressed_tokens_per_second": sum(len(tokenizer.encode(text)) for text in sample_texts) / total_time,
            "compressed_samples_per_second": self.num_inference_samples / total_time
        }
    
    def _profile_memory_usage(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dict[str, float]:
        """Profile memory usage with compression analysis"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        model.to(self.device).eval()
        
        compressed_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
        compressed_gpu_memory = (torch.cuda.memory_allocated() / 1024 / 1024 - initial_gpu_memory) if torch.cuda.is_available() else 0
        
        return {
            "compressed_model_memory_mb": compressed_memory,
            "compressed_model_gpu_memory_mb": compressed_gpu_memory,
            "memory_efficiency_ratio": self.compression_info.get("compression_ratio", 1.0)
        }
    
    def _profile_parameter_count(self, model: PreTrainedModel) -> Dict[str, float]:
        """Profile parameter count with compression analysis"""
        total_params = sum(p.numel() for p in model.parameters())
        original_params = self.compression_info.get("original_parameters", total_params)
        
        return {
            "compressed_total_parameters": float(total_params),
            "original_parameters": float(original_params),
            "parameter_compression_ratio": float(total_params / original_params) if original_params > 0 else 1.0,
            "parameter_reduction_mb": float((original_params - total_params) * 4 / 1024 / 1024)  # 4 bytes per float32
        }
    
    def _profile_compression_efficiency(self, model: PreTrainedModel) -> Dict[str, float]:
        """Profile overall compression efficiency metrics"""
        compression_ratio = self.compression_info.get("compression_ratio", 1.0)
        tensor_ranks = self.compression_info.get("tensor_ranks", [])
        
        # Calculate theoretical compression from tensor ranks if available
        theoretical_compression = 1.0
        if tensor_ranks and len(tensor_ranks) > 1:
            # Simple estimation for tensor-train decomposition
            rank_product = 1
            for rank in tensor_ranks[1:-1]:  # Exclude boundary ranks
                rank_product *= rank
            if rank_product > 0:
                theoretical_compression = rank_product / max(tensor_ranks)
        
        return {
            "actual_compression_ratio": float(compression_ratio),
            "theoretical_compression_ratio": float(theoretical_compression),
            "compression_efficiency_score": float(compression_ratio / theoretical_compression) if theoretical_compression > 0 else 1.0,
            "storage_savings_percent": float((1.0 - compression_ratio) * 100)
        }
