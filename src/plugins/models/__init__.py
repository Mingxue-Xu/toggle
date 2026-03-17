"""
Model plugins for Toggle architecture

This package contains plugins for loading, managing, and analyzing language models
within the Toggle compression pipeline system.
"""

from .loader import ModelLoader, HuggingFaceModelLoader, LocalModelLoader

__all__ = [
    'ModelLoader',
    'HuggingFaceModelLoader', 
    'LocalModelLoader'
]