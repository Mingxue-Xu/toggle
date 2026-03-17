"""
State Management for Toggle Pipeline Architecture

This module provides centralized state management for pipeline execution,
including model state, artifacts, results, and basic persistence.
"""
import json
import threading
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path


class PipelineState:
    """
    Centralized state container for pipeline execution
    
    Manages:
    - Model instances and metadata
    - Compression artifacts and parameters  
    - Evaluation results and metrics
    - Execution metadata and progress
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Initialize core state sections
        self._initialize_state_sections()
    
    def _initialize_state_sections(self):
        """Initialize standard state sections"""
        with self._lock:
            self._data.update({
                # Model state
                'model': None,
                'model_metadata': {},
                'original_model': None,
                'compressed_model': None,
                
                # Compression state
                'compression_artifacts': {},
                'compression_parameters': {},
                'compression_stats': {},
                
                # Evaluation state
                'evaluation_results': {},
                'baseline_results': {},
                
                # Execution state
                'workflow_status': 'initialized',
                'current_step': None,
                'completed_steps': [],
                
                # Analysis state
                'spectral_analysis': {},
                'performance_metrics': {},
                
                # Custom data
                'custom_data': {}
            })
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value with dot notation support
        
        Args:
            key: State key (supports dot notation like 'model.metadata')
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        with self._lock:
            if '.' not in key:
                return self._data.get(key, default)
            
            # Handle nested keys
            keys = key.split('.')
            value = self._data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set state value with dot notation support

        Args:
            key: State key (supports dot notation)
            value: Value to set
        """
        with self._lock:
            if '.' not in key:
                self._data[key] = value
            else:
                # Handle nested keys
                keys = key.split('.')
                current = self._data

                for k in keys[:-1]:
                    # Create dict if key doesn't exist or value is not a dict
                    if k not in current or not isinstance(current.get(k), dict):
                        current[k] = {}
                    current = current[k]

                current[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple state values
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        with self._lock:
            for key, value in updates.items():
                self.set(key, value)
    
    def exists(self, key: str) -> bool:
        """
        Check if state key exists

        Args:
            key: State key to check

        Returns:
            True if key exists
        """
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """
        Delete a state key with dot notation support

        Args:
            key: State key to delete (supports dot notation)

        Returns:
            True if key was deleted, False if key didn't exist
        """
        with self._lock:
            if '.' not in key:
                if key in self._data:
                    del self._data[key]
                    return True
                return False
            else:
                # Handle nested keys
                keys = key.split('.')
                current = self._data

                # Navigate to parent of target key
                for k in keys[:-1]:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        return False

                # Delete the target key
                if isinstance(current, dict) and keys[-1] in current:
                    del current[keys[-1]]
                    return True
                return False
    
    def __getattr__(self, name: str) -> Any:
        """
        Enable direct attribute access to state data
        
        Args:
            name: Attribute name
            
        Returns:
            State value or None
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self.get(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Enable direct attribute setting to state data
        
        Args:
            name: Attribute name
            value: Value to set
        """
        if name.startswith('_') or name in ['get', 'set', 'update', 'exists']:
            # Allow setting private attributes and methods normally
            super().__setattr__(name, value)
        else:
            # Route public attributes to state data
            if hasattr(self, '_data'):
                self.set(name, value)
            else:
                # During initialization, set normally
                super().__setattr__(name, value)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get state summary with key statistics
        
        Returns:
            State summary dictionary
        """
        with self._lock:
            summary = {}
            
            for section in ['model', 'compression_artifacts', 'evaluation_results', 
                           'workflow_status', 'spectral_analysis']:
                section_data = self.get(section)
                
                if section_data is None:
                    summary[section] = 'empty'
                elif isinstance(section_data, dict):
                    summary[section] = f'{len(section_data)} items'
                else:
                    summary[section] = f'{type(section_data).__name__}'
            
            return summary


class StateManager:
    """
    Manager for pipeline state persistence operations
    """

    def __init__(self, state: PipelineState):
        self.state = state
        self.logger = logging.getLogger(__name__)

    def set_plugin_results(self, context: Any, plugin_name: str, results: Any) -> None:
        """
        Store plugin execution results in state

        Args:
            context: Pipeline context (for consistency with interface, may be unused)
            plugin_name: Name of the plugin storing results
            results: Results data to store
        """
        key = f"plugin_results.{plugin_name}"
        self.state.set(key, results)

    def save_state(self, filepath: Path) -> Path:
        """
        Save pipeline state to disk
        
        Args:
            filepath: Path to save state
            
        Returns:
            Path where state was saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare state data for serialization
        state_data = {}
        
        with self.state._lock:
            for key, value in self.state._data.items():
                try:
                    # Skip non-serializable objects
                    if hasattr(value, '__dict__') and not isinstance(value, (dict, list, tuple, str, int, float, bool)):
                        state_data[key] = f"<{type(value).__name__}>"
                    else:
                        state_data[key] = value
                except:
                    state_data[key] = f"<non-serializable>"
            
            # Add metadata
            full_data = {
                'state': state_data,
                'metadata': {
                    'save_time': datetime.now().isoformat(),
                    'state_summary': self.state.get_summary()
                }
            }
        
        # Save to disk
        with open(filepath, 'w') as f:
            json.dump(full_data, f, indent=2, default=str)
        
        self.logger.info(f"State saved to {filepath}")
        return filepath
    
    def load_state(self, filepath: Path) -> None:
        """
        Load pipeline state from disk
        
        Args:
            filepath: Path to load state from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'r') as f:
            full_data = json.load(f)
        
        # Extract state data
        state_data = full_data.get('state', {})
        
        # Load state
        with self.state._lock:
            self.state._data.clear()
            self.state._initialize_state_sections()
            
            for key, value in state_data.items():
                # Skip placeholder objects that were non-serializable
                if isinstance(value, str) and value.startswith('<') and value.endswith('>'):
                    continue
                
                self.state.set(key, value)
        
        self.logger.info(f"State loaded from {filepath}")


# Make logging import available at module level
import logging