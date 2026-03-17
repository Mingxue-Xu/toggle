"""
ModelEvalInterface - Standardized Evaluation Interface for Toggle Framework

This module provides a unified evaluation interface that standardizes loglikelihood
calculations across both baseline and compressed models, serving as a bridge between
existing evaluation infrastructure and the streamlined architecture requirements.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Any
from transformers import PreTrainedModel, PreTrainedTokenizer


class ModelEvalInterface:
    """
    Standardized evaluation interface for both baseline and compressed models.
    
    Provides unified loglikelihood methods that work with any PyTorch model,
    automatically detecting model types and handling device management.
    """

    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer,
                 device: str = 'auto',
                 batch_size: int = 1,
                 max_length: int = 2048):
        """
        Initialize ModelEvalInterface with model and tokenizer.
        
        Args:
            model: PyTorch model (baseline or compressed)
            tokenizer: HuggingFace tokenizer
            device: Device for inference ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Ensure model is in eval mode and on correct device (be tolerant of stubs)
        try:
            if hasattr(self.model, 'eval') and callable(getattr(self.model, 'eval')):
                self.model.eval()
        except Exception:
            pass
        try:
            if hasattr(self.model, 'to') and callable(getattr(self.model, 'to')):
                self.model.to(self.device)
        except Exception:
            pass
        
        # Set up tokenizer (robust fallbacks)
        if getattr(self.tokenizer, 'pad_token', None) is None and getattr(self.tokenizer, 'eos_token', None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, 'pad_token', None) is None and getattr(self.tokenizer, 'unk_token', None) is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if getattr(self.tokenizer, 'pad_token_id', None) is None:
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass
        
        # Cache tokenizer properties
        self.vocab_size = len(self.tokenizer)
        self.eot_token_id = getattr(self.tokenizer, 'eos_token_id', None) or getattr(self.tokenizer, 'pad_token_id', 0)
        
        # Backwards-compat metadata only (no auto-detection)
        # Use explicit hints elsewhere; default to "unknown" here.
        self.model_type = "unknown"
    
    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """
        Calculate log likelihood of target given context.
        
        Args:
            requests: List of (context, continuation) pairs
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        if not requests:
            return []
        
        results = []
        
        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_loglikelihood_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def loglikelihood_rolling(self, requests: List[str]) -> List[Tuple[float]]:
        """
        Calculate rolling log likelihood for perplexity evaluation.
        
        Args:
            requests: List of text strings
            
        Returns:
            List of (log_likelihood,) tuples
        """
        if not requests:
            return []
        
        results = []
        
        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_rolling_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_loglikelihood_batch(self, batch: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """
        Process a batch of loglikelihood requests.
        
        Args:
            batch: List of (context, continuation) pairs
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        # Extract context and continuation from batch
        contexts = []
        continuations = []
        
        for context, continuation in batch:
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
                
                # Handle empty context case
                if ctx_len == 0:
                    # For empty context, we can't calculate loglikelihood since there's no previous token
                    # Return 0.0 as a special case
                    results.append((0.0, True))
                    continue
                
                # For single sequences, positions are straightforward
                # target_start is the position just before the first target token
                target_start = ctx_len - 1  # -1 because logits are shifted by one position
                target_end = len(full_tokens) - 1  # -1 because we don't predict beyond sequence
                
                if target_start >= sequence_logits.size(0) - 1:
                    results.append((0.0, True))
                    continue
                
                # Get log probabilities for target tokens
                # We need logits at positions ctx_len-1 to len(full_tokens)-2 (inclusive)
                # And tokens at positions ctx_len to len(full_tokens)-1 (inclusive)
                target_logits = sequence_logits[ctx_len-1:len(full_tokens)-1]
                target_token_ids = sequence_tokens[ctx_len:len(full_tokens)]
                
                log_probs = F.log_softmax(target_logits, dim=-1)
                target_log_probs = log_probs.gather(1, target_token_ids.unsqueeze(-1)).squeeze(-1)
                
                total_log_prob = target_log_probs.sum().item()
                
                # Check if this was the greedy choice
                greedy_tokens = target_logits.argmax(dim=-1)
                is_greedy = torch.equal(greedy_tokens, target_token_ids)
                
                results.append((total_log_prob, bool(is_greedy)))
        
        return results
    
    def _process_rolling_batch(self, batch: List[str]) -> List[Tuple[float]]:
        """
        Process a batch of rolling loglikelihood requests.
        
        Args:
            batch: List of text strings
            
        Returns:
            List of (log_likelihood,) tuples
        """
        # Tokenize
        inputs = self.tokenizer(
            batch,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        results = []
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            for idx, text in enumerate(batch):
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
    
    
    
    @property
    def max_gen_toks(self) -> int:
        """Return maximum generation tokens (for compatibility)"""
        return 256
    
    def tokenize(self, string: str) -> List[int]:
        """Tokenize a string (for compatibility)"""
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize a list of tokens (for compatibility)"""
        return self.tokenizer.decode(tokens)
