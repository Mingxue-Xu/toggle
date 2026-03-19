"""
Toggle Plugin System

This package provides the plugin infrastructure for the Toggle compression
framework, including model loading, compression strategies, evaluation methods,
and analysis tools.
"""

# Avoid importing subpackages eagerly to prevent heavy side effects during test discovery
# (e.g., LM Harness registry conflicts). Submodules can be imported directly.


from ..framework.plugins import Plugin, PluginRegistry, PluginMetadata

__all__ = ['Plugin', 'PluginRegistry', 'PluginMetadata']

# Plugin category information
PLUGIN_CATEGORIES = {
    'models': 'Model loading and management plugins',
    'compression': 'Tensor and model compression plugins', 
    'evaluation': 'Model evaluation and benchmarking plugins',
    'analysis': 'Model analysis and diagnostic plugins'
}
