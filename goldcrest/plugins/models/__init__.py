"""
Model plugins for Goldcrest architecture

This package contains plugins for loading, managing, and analyzing language models
within the Goldcrest compression pipeline system.
"""

from .loader import ModelLoader, HuggingFaceModelLoader, LocalModelLoader

__all__ = [
    'ModelLoader',
    'HuggingFaceModelLoader', 
    'LocalModelLoader'
]