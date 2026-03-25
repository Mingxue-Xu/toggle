"""
Model Manager for Goldcrest Plugin Architecture

This module provides unified model selection, validation, and management logic
for all plugins, eliminating model handling redundancy across the system.
"""
import logging
from typing import Any, Optional, Union
from pathlib import Path

from .context import PipelineContext


class ModelValidationError(Exception):
    """Exception raised when model validation fails"""
    pass


class ModelManager:
    """
    Unified model selection and validation for all plugins
    
    Provides centralized model access with validation and type checking
    to eliminate redundant model handling code across plugins.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("model_manager")
    
    def get_model(self, context: PipelineContext, model_type: str = "compressed") -> Any:
        """
        Get model by type with validation
        
        Args:
            context: Pipeline execution context
            model_type: Type of model to retrieve ("baseline", "compressed", "original")
            
        Returns:
            The requested model instance
            
        Raises:
            ValueError: If model type is unknown or model not available
        """
        if not isinstance(context, PipelineContext):
            raise ValueError("Invalid context type - must be PipelineContext")
        
        model_type = model_type.lower().strip()
        
        if model_type in ["baseline", "original"]:
            if not hasattr(context.state, 'original_model') or context.state.original_model is None:
                raise ValueError("Original/baseline model not available in context")
            
            model = context.state.original_model
            self.logger.debug(f"Retrieved {model_type} model")
            return model
            
        elif model_type in ["compressed", "current"]:
            if not hasattr(context.state, 'model') or context.state.model is None:
                raise ValueError("Compressed/current model not available in context")
            
            model = context.state.model
            self.logger.debug(f"Retrieved {model_type} model")
            return model
            
        else:
            available_types = ["baseline", "original", "compressed", "current"]
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_types}")
    
    def get_tokenizer(self, context: PipelineContext, model_type: str = "compressed") -> Any:
        """
        Get tokenizer by type with validation
        
        Args:
            context: Pipeline execution context
            model_type: Type of model tokenizer to retrieve
            
        Returns:
            The requested tokenizer instance
            
        Raises:
            ValueError: If tokenizer not available
        """
        if not isinstance(context, PipelineContext):
            raise ValueError("Invalid context type - must be PipelineContext")
        
        model_type = model_type.lower().strip()
        
        if model_type in ["baseline", "original"]:
            if not hasattr(context.state, 'original_tokenizer') or context.state.original_tokenizer is None:
                # Fall back to regular tokenizer if original not available
                if hasattr(context.state, 'tokenizer') and context.state.tokenizer is not None:
                    self.logger.debug("Original tokenizer not available, using current tokenizer")
                    return context.state.tokenizer
                else:
                    raise ValueError("Original tokenizer not available in context")
            
            tokenizer = context.state.original_tokenizer
            self.logger.debug(f"Retrieved {model_type} tokenizer")
            return tokenizer
            
        elif model_type in ["compressed", "current"]:
            if not hasattr(context.state, 'tokenizer') or context.state.tokenizer is None:
                raise ValueError("Tokenizer not available in context")
            
            tokenizer = context.state.tokenizer
            self.logger.debug(f"Retrieved {model_type} tokenizer")
            return tokenizer
            
        else:
            available_types = ["baseline", "original", "compressed", "current"]
            raise ValueError(f"Unknown tokenizer type: {model_type}. Available: {available_types}")
    
    def validate_model(self, model: Any, expected_type: Optional[str] = None) -> bool:
        """
        Validate model structure and type
        
        Args:
            model: Model instance to validate
            expected_type: Expected model type for specific validation
            
        Returns:
            True if validation passes
            
        Raises:
            ModelValidationError: If validation fails
        """
        if model is None:
            raise ModelValidationError("Model cannot be None")
        
        # Check basic model structure
        if not hasattr(model, '__call__'):
            raise ModelValidationError("Model must be callable")
        
        # Type-specific validation
        if expected_type == "compressed":
            return self._validate_compressed_model(model)
        elif expected_type == "baseline":
            return self._validate_baseline_model(model)
        
        # General validation passed
        self.logger.debug("Model validation passed")
        return True
    
    def _validate_compressed_model(self, model: Any) -> bool:
        """Validate that model contains compressed layers"""
        try:
            # Try to import factor layer types
            from ..plugins.compression.tensorizer import FactorEmbedding, FactorLinear
            
            # Check if model has compressed layers
            has_compressed_layers = False
            
            if hasattr(model, 'modules'):
                for module in model.modules():
                    if isinstance(module, (FactorEmbedding, FactorLinear)):
                        has_compressed_layers = True
                        break
            
            if not has_compressed_layers:
                # Check for other compression indicators
                if hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
                    # Look for factor layer parameters
                    factor_keys = [k for k in state_dict.keys() if 'factor' in k.lower()]
                    has_compressed_layers = len(factor_keys) > 0
            
            if not has_compressed_layers:
                raise ModelValidationError("Model does not appear to be compressed - no factor layers found")
            
            self.logger.debug("Compressed model validation passed")
            return True
            
        except ImportError:
            # If compression modules aren't available, skip specific validation
            self.logger.warning("Cannot validate compressed model - compression modules not available")
            return True
    
    def _validate_baseline_model(self, model: Any) -> bool:
        """Validate baseline model structure"""
        # Basic structure checks for baseline models
        required_attrs = ['forward', 'parameters']
        
        for attr in required_attrs:
            if not hasattr(model, attr):
                raise ModelValidationError(f"Baseline model missing required attribute: {attr}")
        
        # Check that model has parameters
        try:
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                raise ModelValidationError("Baseline model has no parameters")
        except Exception as e:
            raise ModelValidationError(f"Could not count model parameters: {str(e)}")
        
        self.logger.debug("Baseline model validation passed")
        return True
    
    def get_model_info(self, model: Any) -> dict:
        """
        Get basic information about a model
        
        Args:
            model: Model instance to analyze
            
        Returns:
            Dictionary with model information
        """
        info = {
            "type": type(model).__name__,
            "has_parameters": hasattr(model, 'parameters'),
            "parameter_count": 0,
            "is_compressed": False,
            "device": "unknown"
        }
        
        # Get parameter count
        if hasattr(model, 'parameters'):
            info["parameter_count"] = sum(p.numel() for p in model.parameters())
        
        # Check if compressed
        try:
            from ..plugins.compression.tensorizer import FactorEmbedding, FactorLinear
            
            if hasattr(model, 'modules'):
                for module in model.modules():
                    if isinstance(module, (FactorEmbedding, FactorLinear)):
                        info["is_compressed"] = True
                        break
        except ImportError:
            pass
        
        # Get device information
        if hasattr(model, 'parameters'):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                info["device"] = str(first_param.device)
        
        return info

    def set_model(self, context: PipelineContext, model: Any, model_type: str = "compressed") -> None:
        """
        Set model in context with validation
        
        Args:
            context: Pipeline execution context
            model: Model instance to set
            model_type: Type designation for the model
        """
        if not isinstance(context, PipelineContext):
            raise ValueError("Invalid context type - must be PipelineContext")
        
        # Validate model first
        self.validate_model(model)
        
        model_type = model_type.lower().strip()
        
        if model_type in ["baseline", "original"]:
            context.state.set("original_model", model)
            self.logger.info(f"Set {model_type} model in context")
            
        elif model_type in ["compressed", "current"]:
            context.state.set("model", model)
            self.logger.info(f"Set {model_type} model in context")
            
        else:
            raise ValueError(f"Unknown model type for setting: {model_type}")
    
    def set_tokenizer(self, context: PipelineContext, tokenizer: Any, model_type: str = "compressed") -> None:
        """
        Set tokenizer in context
        
        Args:
            context: Pipeline execution context
            tokenizer: Tokenizer instance to set
            model_type: Type designation for the tokenizer
        """
        if not isinstance(context, PipelineContext):
            raise ValueError("Invalid context type - must be PipelineContext")
        
        model_type = model_type.lower().strip()
        
        if model_type in ["baseline", "original"]:
            context.state.set("original_tokenizer", tokenizer)
            self.logger.info(f"Set {model_type} tokenizer in context")
            
        elif model_type in ["compressed", "current"]:
            context.state.set("tokenizer", tokenizer)
            self.logger.info(f"Set {model_type} tokenizer in context")
            
        else:
            raise ValueError(f"Unknown tokenizer type for setting: {model_type}")
