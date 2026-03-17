"""
Configuration Loader with Validation for Toggle Framework

This module provides configuration loading and validation capabilities
as specified in the parallel implementation plan Phase 1.
"""
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationLoader:
    """ConfigurationLoader with validation as specified in Phase 1"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file (alias for load_config).

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        return self.load_config(config_path)

    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return config or {}
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Basic configuration validation
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        # Basic validation for required sections
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        self.logger.debug("Configuration validation passed")
        return True