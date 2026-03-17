"""
Model Loading Plugin Base Class for Toggle Architecture

This module provides the ModelLoader base class for loading and managing
language models within the plugin-based pipeline system.
"""
import torch
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

try:
    from framework.plugins import Plugin, PluginMetadata
    from framework.context import PipelineContext
except ModuleNotFoundError:
    from src.framework.plugins import Plugin, PluginMetadata
    from src.framework.context import PipelineContext


class ModelLoader(Plugin):
    """
    Base class for model loading plugins
    
    Provides standard interface for loading language models with tokenizers
    and integrating them into the pipeline execution context.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: str = "auto", **kwargs):
        """
        Initialize ModelLoader
        
        Args:
            model_name: Name/path of model to load
            device: Device to load model on ("auto", "cpu", "cuda", etc.)
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device if device != "auto" else self._get_default_device()
        self._model = None
        self._tokenizer = None
        
    def _get_default_device(self) -> str:
        """Get default device (cuda if available, else cpu)"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name=self.name,
            description="Model loading plugin for language models",
            category="model"
        )
    
    def _execute_impl(self, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute model loading
        
        Args:
            model_name: Override model name
            **kwargs: Additional loading parameters
            
        Returns:
            Dictionary containing loaded model and tokenizer
        """
        target_model = model_name or self.model_name
        if not target_model:
            raise ValueError("No model name provided")
        
        self.logger.info(f"Loading model: {target_model} on device: {self.device}")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model(target_model, **kwargs)
        
        # Store in context state
        if self.context:
            self.context.state.model = model
            self.context.state.tokenizer = tokenizer
            self.context.state.model_name = target_model
            
            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    "model.loaded",
                    {
                        "model_name": target_model,
                        "device": self.device,
                        "model_size": self.get_model_size(model)
                    },
                    source=self.name
                )
        
        self._model = model
        self._tokenizer = tokenizer
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": target_model,
            "device": self.device
        }
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer
        
        Args:
            model_name: Name/path of model to load
            **kwargs: Additional loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    def analyze_model(self, model: Optional[PreTrainedModel] = None) -> Dict[str, Any]:
        """
        Analyze model properties
        
        Args:
            model: Model to analyze (uses loaded model if None)
            
        Returns:
            Dictionary with model analysis results
        """
        target_model = model or self._model
        if target_model is None:
            raise ValueError("No model available for analysis")
        
        return {
            "num_parameters": self.get_model_size(target_model),
            "model_type": target_model.__class__.__name__,
            "device": next(target_model.parameters()).device,
            "dtype": next(target_model.parameters()).dtype,
            "config": target_model.config.to_dict() if hasattr(target_model, 'config') else {}
        }
    
    def get_model_size(self, model: PreTrainedModel) -> int:
        """
        Get number of parameters in model
        
        Args:
            model: Model to analyze
            
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in model.parameters())
    
    @property
    def model(self) -> Optional[PreTrainedModel]:
        """Get loaded model"""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Get loaded tokenizer"""
        return self._tokenizer


class HuggingFaceModelLoader(ModelLoader):
    """
    Concrete implementation for loading HuggingFace transformers models
    """
    
    def __init__(self, trust_remote_code: bool = False, **kwargs):
        """
        Initialize HuggingFace model loader
        
        Args:
            trust_remote_code: Whether to trust remote code in models
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.trust_remote_code = trust_remote_code
    
    def load_model(self, model_name: str, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load HuggingFace model and tokenizer
        
        Args:
            model_name: HuggingFace model name or path
            **kwargs: Additional loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Map common auth/cache kwargs from config
            auth_token = kwargs.pop("hf_token", None) or kwargs.get("use_auth_token") or kwargs.get("token")
            cache_dir = kwargs.get("cache_dir")
            common = {}
            if auth_token:
                # Support both legacy and current parameter names
                common["use_auth_token"] = auth_token
                common["token"] = auth_token
            if cache_dir:
                common["cache_dir"] = cache_dir
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code,
                **common,
                **kwargs
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32,  # Default dtype
                **common,
                **kwargs
            )
            
            # Move to device
            model = model.to(self.device)
            
            self.logger.info(f"Successfully loaded {model_name} with {self.get_model_size(model):,} parameters")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise


class LocalModelLoader(ModelLoader):
    """
    Model loader for locally saved models
    """
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model from local path
        
        Args:
            model_path: Path to local model directory
            **kwargs: Additional loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Load from local path
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            model = AutoModel.from_pretrained(model_path, **kwargs)
            
            # Move to device
            model = model.to(self.device)
            
            self.logger.info(f"Successfully loaded local model from {model_path}")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load local model from {model_path}: {str(e)}")
            raise
