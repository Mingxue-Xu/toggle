from __future__ import annotations
"""
ModelConsolidator Plugin for Toggle Architecture

This plugin implements model consolidation and direct compression strategies,
including the DirectConsolidated approach mentioned in the migration strategy.
"""
import time
from typing import List, Dict, Any, Optional, Union
import torch

from ...framework.plugins import Plugin
from .tensorly_backend import set_tensorly_backend

"""
TODO: 
    1. The standard I/O information should be unified in another class.
"""


SUPPORTED_COMPRESSION_STRATEGIES = {'low_rank':['tensor_train','tucker','cp','svd']}

class ModelConsolidator(Plugin):
    """
    Model consolidation plugin for comprehensive model compression
    
    Implements DirectConsolidated approach that applies compression strategies
    directly to model components based on the original consolidate_model_direct
    functionality from factorization.py.
    """
    
    def __init__(self, 
                 compression_method: str = "tensor_train",
                 target_modules: Optional[List[str]] = None,
                 device: str = 'cpu', 
                 backend: str = 'pytorch',
                 **kwargs):
        """
        Initialize ModelConsolidator plugin with shared infrastructure
        
        Args:
            compression_method: Compression method ('tensor_train', 'tucker', 'cp')
            target_modules: List of module names to compress
            device: Computation device
            backend: Tensorly backend
            **kwargs: Additional configuration including compression parameters
        """
        # Initialize as Plugin instead of ModelCompressionPlugin
        super().__init__(name="compression", **kwargs)
        
        self.compression_method = compression_method
        self.device = device
        self.backend = backend
        self.compression_params = kwargs
        # Optional per-module overrides: list of {pattern, func_name, ...}
        self.method_overrides: Optional[List[Dict[str, Any]]] = kwargs.get('method_overrides') or kwargs.get('overrides')
        self.target_modules = target_modules or self._get_default_target_modules()
        
        # Defer compression strategy initialization until used
        self.compression_strategy = None
        self.tensorizer = None
        
        # Defer tensorly backend selection until compression is used

    def _resolve_svd_backend_settings(self) -> tuple[Optional[str], Dict[str, Any]]:
        backend = self.compression_params.get("svd_backend")
        backend_config = self.compression_params.get("svd_backend_config")

        svd_cfg: Dict[str, Any] = {}
        compression_cfg = self.compression_params.get("compression")
        if isinstance(compression_cfg, dict):
            svd_cfg = dict(compression_cfg.get("svd") or {})
        if not svd_cfg and isinstance(self.compression_params.get("svd"), dict):
            svd_cfg = dict(self.compression_params.get("svd") or {})

        if backend is None:
            backend = svd_cfg.get("backend")
        if backend_config is None:
            backend_config = svd_cfg.get("backend_config") or svd_cfg.get("cola")
        if backend is None and backend_config:
            backend = "cola"
        return backend, dict(backend_config or {})

    # ----------------------
    # Path resolution helpers
    # ----------------------
    def _get_default_target_modules(self) -> List[str]:
        """
        Provide a conservative default set of target module names for compression.

        Chosen to align with common LM architectures and existing tests:
        - transformer.wte: word/token embeddings
        - transformer.wpe: positional embeddings
        - lm_head: language modeling head
        - score / classifier: heads used in seq2seq or classification models
        """
        return [
            "transformer.wte",
            "transformer.wpe",
            "lm_head",
            "score",
            "classifier",
        ]

    def _split_attr_and_index(self, token: str) -> tuple:
        """
        Split a path token that may include bracket index, e.g., 'layers[3]' -> ('layers', 3)
        Supports wildcard index 'layers[*]' -> ('layers', '*')
        Returns (attr_name, index_or_None)
        """
        if '[' in token and token.endswith(']'):
            name, idx = token.split('[', 1)
            idx = idx[:-1]  # drop ']'
            if idx == '*':
                return name, '*'
            if ':' in idx:
                # slice syntax start:stop (no step support for simplicity)
                start_s, stop_s, *rest = (idx.split(':') + [None, None])[:2]
                start = int(start_s) if start_s not in (None, '') else None
                stop = int(stop_s) if stop_s not in (None, '') else None
                return name, slice(start, stop, None)
            try:
                return name, int(idx)
            except ValueError:
                # Fallback: treat whole token as attr
                return token, None
        return token, None

    def _expand_target_modules(self, model: torch.nn.Module, patterns: Optional[List[str]] = None) -> List[str]:
        """
        Expand wildcard list-style patterns (e.g., 'model.layers[*].self_attn.q_proj')
        into concrete dot/bracket paths (e.g., 'model.layers[0].self_attn.q_proj', ...).
        """
        patterns = patterns or self.target_modules
        expanded: List[str] = []

        for pat in patterns:
            parts = pat.split('.') if isinstance(pat, str) else []
            # BFS over path segments to expand wildcards
            nodes = [(model, None, [])]  # (obj, unused, built_parts)
            for part in parts:
                new_nodes = []
                for obj, _unused, built in nodes:
                    attr, idx = self._split_attr_and_index(part)
                    if hasattr(obj, attr):
                        child = getattr(obj, attr)
                        if idx == '*':
                            # Expand all items if ModuleList or list/tuple
                            if isinstance(child, (torch.nn.ModuleList, list, tuple)):
                                for i, sub in enumerate(child):
                                    new_nodes.append((sub, None, built + [f"{attr}[{i}]"]))
                            else:
                                # Not indexable; nothing to expand
                                continue
                        elif isinstance(idx, slice):
                            # Expand slice range
                            if isinstance(child, (torch.nn.ModuleList, list, tuple)):
                                length = len(child)
                                start = idx.start if idx.start is not None else 0
                                stop = idx.stop if idx.stop is not None else length
                                # handle negative indices
                                if isinstance(start, int) and start < 0:
                                    start = length + start
                                if isinstance(stop, int) and stop < 0:
                                    stop = length + stop
                                # clamp
                                start = max(0, min(length, start))
                                stop = max(0, min(length, stop))
                                for i in range(start, stop):
                                    new_nodes.append((child[i], None, built + [f"{attr}[{i}]"]))
                            else:
                                continue
                        elif isinstance(idx, int):
                            # Index into ModuleList/list
                            if isinstance(child, (torch.nn.ModuleList, list, tuple)) and 0 <= idx < len(child):
                                new_nodes.append((child[idx], None, built + [f"{attr}[{idx}]"]))
                            else:
                                continue
                        else:
                            new_nodes.append((child, None, built + [attr]))
                    else:
                        # If part itself is like '0' (rare) or cannot be resolved, skip branch
                        continue
                nodes = new_nodes
                if not nodes:
                    break

            # For terminal nodes, we now have full paths in 'built'
            for _obj, _u, built in nodes:
                if built:
                    expanded.append('.'.join(built))

        # Deduplicate while preserving order
        seen = set()
        result = []
        for p in expanded:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result
    
    def do_execute(self, context: 'PipelineContext', **params):
        """
        Plugin-specific execution logic using shared infrastructure

        This method implements the Plugin interface and uses shared infrastructure
        components like model_manager, state_manager, and event_manager.

        Args:
            context: Pipeline context with state and event bus
            **params: Compression parameters (compression_method, target_modules, etc.)

        Returns:
            TensorCompressionResult with compression and surgery metadata
        """
        # Get model from params first, then fall back to model manager
        model = params.pop('model', None)
        if model is None:
            model = self.model_manager.get_model(context, "compressed")

        # Validate model exists
        self.model_manager.validate_model(model)

        # Update parameters from context if provided
        compression_method = params.get('compression_method', self.compression_method)
        target_modules = params.get('target_modules', self.target_modules)

        # Update compression method if changed
        if compression_method != self.compression_method:
            self.set_compression_method(compression_method, **params)

        # Update target modules if provided
        if target_modules != self.target_modules:
            self.target_modules = target_modules

        # Execute compression with progress tracking through shared infrastructure
        self.emit_progress(0.1, "Initializing compression")

        # Perform model surgery - compress and replace layers
        results = self.compress_model_with_surgery(model, **params)

        # Update state with results using shared state manager
        self.state_manager.set_plugin_results(context, "compression", results.parameters)

        self.emit_progress(1.0, "Compression completed")
        
        return results
    
    def _init_compression_strategy_from_factory(self):
        """Initialize compression strategy using UnifiedStrategyFactory"""
        # Use shared infrastructure strategy factory
        if not self.strategy_factory:
            from ...framework.strategy_factory import UnifiedStrategyFactory
            self._strategy_factory = UnifiedStrategyFactory()
        
        # Prepare parameters for the factory based on compression method
        strategy_params: Dict[str, Any] = {}
        
        # Add method-specific parameters
        if self.compression_method == "tensor_train":
            strategy_params['device'] = self.device
            strategy_params['backend'] = self.backend
            strategy_params['tensor_ranks'] = self.compression_params.get("tensor_ranks", [1, 4, 4, 1])
        elif self.compression_method == "tucker":
            strategy_params['device'] = self.device
            strategy_params['backend'] = self.backend
            strategy_params['tucker_ranks'] = self.compression_params.get("tucker_ranks", [4, 4])
        elif self.compression_method == "cp":
            strategy_params['device'] = self.device
            strategy_params['backend'] = self.backend
            strategy_params['cp_rank'] = self.compression_params.get("cp_rank", 4)
        elif self.compression_method == "svd":
            # Support both 'rank' and legacy 'svd_rank' from config
            rank = self.compression_params.get("rank", self.compression_params.get("svd_rank", None))
            preserve_energy = self.compression_params.get("preserve_energy", None)
            if rank is None and preserve_energy is None:
                preserve_energy = 0.9
            if rank is not None:
                strategy_params['rank'] = rank
            if preserve_energy is not None:
                strategy_params['preserve_energy'] = preserve_energy
            svd_backend, svd_backend_config = self._resolve_svd_backend_settings()
            if svd_backend:
                strategy_params["svd_backend"] = svd_backend
            if svd_backend_config:
                strategy_params["svd_backend_config"] = svd_backend_config
        
        # Select tensorly backend lazily (may import torch if backend='pytorch')
        set_tensorly_backend(self.backend)

        # Get strategy from factory
        self.compression_strategy = self.strategy_factory.get_compression_strategy(
            self.compression_method, 
            **strategy_params
        )
        
        # Initialize tensorizer for preprocessing
        try:
            from .tensorizer import Tensorizer  # Lazy import; may require torch
            self.tensorizer = Tensorizer(device=self.device, backend=self.backend)
        except Exception:
            self.tensorizer = None
    
    
    def compress_model(self, model: 'torch.nn.Module', **params):
        """
        Compress the entire model using DirectConsolidated approach
        
        Args:
            model: Model to compress
            **params: Additional compression parameters
            
        Returns:
            TensorCompressionResult with compressed model information
        """
        # Local imports to avoid heavy deps at module import time
        from .base import TensorCompressionResult
        import torch

        # Ensure strategy/tensorizer are initialized
        if self.compression_strategy is None or self.tensorizer is None:
            self._init_compression_strategy_from_factory()

        compressed_tensors = {}
        total_original_size = 0
        total_compressed_size = 0
        compression_stats = {}
        
        # Print available modules in the model
        print(f"\n=== MODEL ANALYSIS ===")
        print(f"Model type: {type(model).__name__}")
        print(f"Target modules for compression: {self.target_modules}")
        
        print(f"\n=== ORIGINAL MODEL STRUCTURE ===")
        # print(model)

        available_modules = []
        for name, module in model.named_modules():          # NOTE: The current implementation is compressed all the modules with the same structure
            if hasattr(module, 'weight') and module.weight is not None:
                available_modules.append((name, type(module).__name__, module.weight.shape, module.weight.numel()))

        # Extract compression parameters from params or use defaults
        target_tensor_size = params.get("tensor_size")
        granularity = params.get("granularity", "matrix")
        
        # print(f"\n=== COMPRESSION PROCESS ===")
        # print(f"Compression method: {self.compression_method}")
        # print(f"Target tensor size: {target_tensor_size}")
        # print(f"Granularity: {granularity}")
        
        # Expand wildcard targets (e.g., layers[*], layers[0:4]) into concrete paths
        concrete_targets = self._expand_target_modules(model, self.target_modules)
        found_modules = []
        
        # Prepare override map if provided
        overrides = params.get('method_overrides') or params.get('overrides') or self.method_overrides
        expanded_override_map: Dict[str, Dict[str, Any]] = {}
        if overrides:
            for override in overrides:
                pattern = override.get('pattern') or override.get('module') or override.get('name')
                if not pattern:
                    continue
                for expanded in self._expand_target_modules(model, [pattern]):
                    expanded_override_map[expanded] = override

        def pick_override(module_path: str) -> Optional[Dict[str, Any]]:
            # Find the longest matching override prefix
            best = None
            for prefix, cfg in expanded_override_map.items():
                if module_path.startswith(prefix):
                    if best is None or len(prefix) > len(best[0]):
                        best = (prefix, cfg)
            return best[1] if best else None

        # Process each target module
        for module_name in concrete_targets:

            module_results = self._compress_module(
                model=model,
                module_name=module_name,
                tensor_size=target_tensor_size,
                granularity=granularity,
                params=params,
            )
            
            if module_results:
                found_modules.append(module_name)
                compressed_tensors.update(module_results["compressed_tensors"])
                total_original_size += module_results["original_size"]
                total_compressed_size += module_results["compressed_size"]
                compression_stats[module_name] = module_results["stats"]
            else:
                print(f"WARNING: Module '{module_name}' not found or has no weight parameters")

        return TensorCompressionResult(
            compressed_tensors=compressed_tensors,
            total_compression_ratio=0.0,  # Deprecated: not computed
            compression_time=0.0,  # Will be set by parent class
            method=f"model_consolidator_{self.compression_method}",
            parameters={
                "compression_method": self.compression_method,
                "target_modules": concrete_targets,
                "granularity": granularity,
                "compression_stats": compression_stats,
                "total_original_size": total_original_size,
                "total_compressed_size": total_compressed_size
            }
        )
    
    def _compress_module(self, 
                        model,
                        module_name: str, 
                        tensor_size: Optional[List],
                        granularity: str,
                        params: Dict) -> Optional[Dict[str, Any]]:
        """
        Compress a specific module in the model
        
        Args:
            model: The model containing the module
            module_name: Name of the module to compress
            tensor_size: Target tensor size for tensorization
            granularity: Compression granularity ('vector' or 'matrix')
            **params: Additional parameters
            
        Returns:
            Dictionary with compression results or None if module not found
        """
        # Find the target module
        target_module = self._get_module_by_name(model, module_name)
        if target_module is None:
            self.logger.warning(f"Module '{module_name}' not found in model")
            return None
        
        # Get the parameter tensor
        if hasattr(target_module, 'weight') and target_module.weight is not None:
            parameter_tensor = target_module.weight.data
            
            # # Print detailed layer information BEFORE compression
            # print(f"\n=== BEFORE COMPRESSION: Module '{module_name}' ===")
            # print(f"Module type: {type(target_module).__name__}")
            # print(f"Original tensor shape: {parameter_tensor.shape}")
            # print(f"Original tensor size: {parameter_tensor.numel():,} parameters")
            # print(f"Data type: {parameter_tensor.dtype}")
            # print(f"Device: {parameter_tensor.device}")
            # if len(parameter_tensor.shape) >= 2:
            #     print(f"Matrix dimensions: {parameter_tensor.shape[0]} x {parameter_tensor.shape[1]}")
            # print(f"Memory usage: {parameter_tensor.numel() * 4 / 1024 / 1024:.2f} MB (assuming float32)")
            
        else:
            self.logger.warning(f"Module '{module_name}' has no weight parameter")
            return None
        
        # Determine per-module method override if any
        overrides = params.get('method_overrides') or params.get('overrides') or self.method_overrides
        use_granularity = granularity
        use_tensor_size = tensor_size
        # Save current method/params to restore after this module
        prev_method = self.compression_method
        prev_params = dict(self.compression_params)
        override_applied = False
        if overrides:
            # Build expanded map lazily
            expanded_override_map: Dict[str, Dict[str, Any]] = {}
            for override in overrides:
                pattern = override.get('pattern') or override.get('module') or override.get('name')
                if not pattern:
                    continue
                for expanded in self._expand_target_modules(model, [pattern]):
                    expanded_override_map[expanded] = override

            # find best match
            best_prefix = None
            best_cfg = None
            for prefix, cfg in expanded_override_map.items():
                if module_name.startswith(prefix):
                    if best_prefix is None or len(prefix) > len(best_prefix):
                        best_prefix, best_cfg = prefix, cfg
            if best_cfg:
                method = best_cfg.get('func_name') or best_cfg.get('method')
                if method:
                    if method != self.compression_method:
                        # Switch strategy and params for this module
                        self.set_compression_method(method, **best_cfg)
                        override_applied = True
                    else:
                        # Same method but different params (e.g., different SVD rank)
                        self.compression_params.update(best_cfg)
                        self._init_compression_strategy_from_factory()
                        override_applied = True
                # module-specific params
                if 'granularity' in best_cfg:
                    use_granularity = best_cfg.get('granularity', use_granularity)
                if 'tensor_size' in best_cfg:
                    use_tensor_size = best_cfg.get('tensor_size', use_tensor_size)
                # Support alternate naming
                if 'tensor size' in best_cfg and not use_tensor_size:
                    use_tensor_size = best_cfg.get('tensor size')
                if 'tensor rank' in best_cfg and best_cfg.get('func_name') in ('tensor_train','tt'):
                    # ranks handled via set_compression_method through kwargs
                    pass

        # Apply compression based on chosen granularity
        compressed_tensors = {}
        original_size = parameter_tensor.numel()
        compressed_size = 0
        
        if use_granularity == "vector":
            # Compress each vector (row) separately
            compressed_tensors, compressed_size = self._compress_vectorwise(
                parameter_tensor, use_tensor_size, module_name
            )
        else:  # granularity == "matrix"
            # Compress the entire matrix
            compressed_tensors, compressed_size = self._compress_matrixwise(
                parameter_tensor, use_tensor_size, module_name
            )
        
        if override_applied:
            self.compression_method = prev_method
            self.compression_params = dict(prev_params)
            self._init_compression_strategy_from_factory()

        return {
            "compressed_tensors": compressed_tensors,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "stats": {
                "granularity": use_granularity,
                "tensor_shape": parameter_tensor.shape
            }
        }
    
    def _compress_vectorwise(self, 
                           parameter_tensor: torch.Tensor, 
                           tensor_size: Optional[List],
                           module_name: str) -> tuple:
        """
        Compress tensor vector by vector (similar to original vector granularity)
        
        Args:
            parameter_tensor: Parameter tensor to compress
            tensor_size: Target tensor size
            module_name: Name of the module
            
        Returns:
            Tuple of (compressed_tensors_dict, total_compressed_size)
        """
        compressed_tensors = {}
        total_compressed_size = 0
        
        # Process each vector (row) of the parameter tensor
        for i in range(parameter_tensor.shape[0]):
            vector = parameter_tensor[i]
            
            # Tensorize if tensor_size is specified
            if tensor_size:
                vector_tensorized = self.tensorizer.tensorize(tensor_size, vector)
            else:
                vector_tensorized = vector
            
            # Compress the (possibly tensorized) vector using strategy directly
            compressed_vector = self.compression_strategy.compress(vector_tensorized)
            
            # Store compressed vector
            key = f"{module_name}_vector_{i}"
            compressed_tensors[key] = compressed_vector
            
            # Calculate compressed size
            if hasattr(compressed_vector.factors, '__len__') and not isinstance(compressed_vector.factors, torch.Tensor):
                # Multiple factors
                vector_compressed_size = sum(
                    factor.numel() for factor in compressed_vector.factors
                )
            else:
                # Single tensor
                vector_compressed_size = compressed_vector.factors.numel()
            
            total_compressed_size += vector_compressed_size
        
        return compressed_tensors, total_compressed_size
    
    def _compress_matrixwise(self, 
                           parameter_tensor: torch.Tensor, 
                           tensor_size: Optional[List],
                           module_name: str) -> tuple:
        """
        Compress entire tensor as a matrix
        
        Args:
            parameter_tensor: Parameter tensor to compress
            tensor_size: Target tensor size
            module_name: Name of the module
            
        Returns:
            Tuple of (compressed_tensors_dict, total_compressed_size)
        """
        # Tensorize if tensor_size is specified
        if tensor_size:
            tensor_tensorized = self.tensorizer.tensorize(tensor_size, parameter_tensor)
        else:
            tensor_tensorized = parameter_tensor
        
        # Compress the (possibly tensorized) tensor using strategy directly
        compressed_tensor = self.compression_strategy.compress(tensor_tensorized)
        
        # Calculate compressed size - handle different compression result types
        if hasattr(compressed_tensor, 'factors'):
            # Tensor Train and Tucker decomposition results
            if hasattr(compressed_tensor.factors, '__len__') and not isinstance(compressed_tensor.factors, torch.Tensor):
                # Multiple factors
                compressed_size = sum(
                    factor.numel() for factor in compressed_tensor.factors
                )
            else:
                # Single tensor
                compressed_size = compressed_tensor.factors.numel()
        elif hasattr(compressed_tensor, 'size'):
            # SVD and other methods that have a size() method
            compressed_size = compressed_tensor.size()
        else:
            # Fallback: try to sum all tensor attributes
            compressed_size = 0
            for attr_name in dir(compressed_tensor):
                attr = getattr(compressed_tensor, attr_name)
                if isinstance(attr, torch.Tensor):
                    compressed_size += attr.numel()
        
        return {f"{module_name}_matrix": compressed_tensor}, compressed_size
    
    def _get_module_by_name(self, model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
        """
        Get a module from model by name

        Args:
            model: The model
            module_name: Name of the module

        Returns:
            The module or None if not found
        """
        try:
            parts = module_name.split('.')
            current = model
            for part in parts:
                attr, idx = self._split_attr_and_index(part)
                if not hasattr(current, attr):
                    return None
                child = getattr(current, attr)
                if isinstance(idx, int):
                    if isinstance(child, (torch.nn.ModuleList, list, tuple)) and 0 <= idx < len(child):
                        current = child[idx]
                    else:
                        return None
                else:
                    current = child
            return current
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            self.logger.debug(f"Module path '{module_name}' not found: {e}")
            return None
    
    def set_compression_method(self, method: str, **params):
        """
        Change compression method dynamically using strategy factory
        
        Args:
            method: New compression method
            **params: Method-specific parameters
        """
        self.compression_method = method
        self.compression_params.update(params)
        self._init_compression_strategy_from_factory()
    
    def _get_layer_by_name(self, model: torch.nn.Module, layer_name: str):
        """
        Get a layer from model by its name path (e.g., 'transformer.wte')
        
        Args:
            model: The model to search in
            layer_name: Dot-separated path to the layer
            
        Returns:
            The target layer module
            
        Raises:
            ValueError: If layer path not found in model
        """
        parts = layer_name.split('.')
        current = model
        for part in parts:
            attr, idx = self._split_attr_and_index(part)
            if not hasattr(current, attr):
                raise ValueError(f"Layer path '{layer_name}' not found in model (missing attr '{attr}')")
            child = getattr(current, attr)
            if isinstance(idx, int):
                if isinstance(child, (torch.nn.ModuleList, list, tuple)) and 0 <= idx < len(child):
                    current = child[idx]
                else:
                    raise ValueError(f"Layer path '{layer_name}' index out of range for '{attr}'")
            else:
                current = child
        return current
    
    def _replace_layer_in_model(self, model: torch.nn.Module, layer_name: str, new_layer):
        """
        Replace a layer in the model with a new compressed layer
        
        Args:
            model: The model containing the layer
            layer_name: Dot-separated path to the layer
            new_layer: New layer to replace with (FactorEmbedding or FactorLinear)
        """
        parts = layer_name.split('.')
        # Navigate to parent of target layer
        parent = model
        for part in parts[:-1]:
            attr, idx = self._split_attr_and_index(part)
            child = getattr(parent, attr)
            if isinstance(idx, int):
                parent = child[idx]
            else:
                parent = child
        # Replace the final layer
        final_attr, final_idx = self._split_attr_and_index(parts[-1])
        if isinstance(final_idx, int):
            # Index replacement in ModuleList
            container = getattr(parent, final_attr)
            container[final_idx] = new_layer
        else:
            setattr(parent, final_attr, new_layer)
        
        self.logger.info(f"Replaced layer '{layer_name}' with {type(new_layer).__name__}")
    
    def _create_compressed_layer(self, original_layer, compressed_data: CompressedTensor):
        """
        Create FactorEmbedding or FactorLinear layer from compressed data
        
        Args:
            original_layer: Original layer from the model
            compressed_data: CompressedTensor containing factorized data
            
        Returns:
            FactorEmbedding or FactorLinear layer based on original layer type
            
        Raises:
            ValueError: If layer type is not supported
        """
        from ...framework.layers import FactorEmbedding, FactorLinear, FactorLayer, Factor

        # Specialized path for SVD compressed data (detect by attributes)
        if hasattr(compressed_data, 'u') and hasattr(compressed_data, 's'):
            layer = self.compression_strategy.create_layer(
                compressed_data, original_layer.weight.shape
            )
            if isinstance(layer, FactorLinear):
                if original_layer.bias is not None:
                    layer.bias = original_layer.bias
                else:
                    layer.bias = None
            return layer


        # Check layer type and create appropriate compressed layer
        if hasattr(original_layer, 'num_embeddings') and hasattr(original_layer, 'embedding_dim'):
            # Create FactorEmbedding for embedding layers
            factor_layer = self._create_factor_layer_from_compressed_data(compressed_data)
            # Create FactorEmbedding with the factor layer as the weight
            return FactorEmbedding(
                _num_embeddings=original_layer.num_embeddings,
                _embedding_dim=original_layer.embedding_dim,
                _weight=[factor_layer]  # Pass as ModuleList
            )
            
        elif hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            # Create FactorLinear for linear layers  
            factor_layer = self._create_factor_layer_from_compressed_data(compressed_data)
            factor_linear = FactorLinear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                _weight=factor_layer,
                bias=original_layer.bias is not None
            )
            # Ensure func_name on the linear layer reflects the underlying factorization method
            if hasattr(factor_layer, 'func_name'):
                factor_linear.func_name = getattr(factor_layer, 'func_name', factor_linear.func_name)
            return factor_linear
        else:
            raise ValueError(f"Unsupported layer type for compression: {type(original_layer)}")
    
    def _create_factor_layer_from_compressed_data(self, compressed_data: CompressedTensor) -> 'FactorLayer':
        """
        Create FactorLayer from CompressedTensor data
        
        Args:
            compressed_data: CompressedTensor containing factorized tensor data
            
        Returns:
            FactorLayer with factors from compressed data
            
        Raises:
            ValueError: If compression method is not supported
        """
        from ...framework.layers import FactorLayer, Factor
        
        method_name = getattr(compressed_data, "method", None) or self.compression_method
        # Extract factors from compressed data based on compression method
        if method_name in SUPPORTED_COMPRESSION_STRATEGIES["low_rank"]:
            # Special-case SVD: some strategies return a structured object without `.factors`
            if method_name == 'svd' and not hasattr(compressed_data, 'factors'):
                try:
                    # Delegate to SVD strategy's layer creator for correctness
                    from .svd import SVD as _SVD
                    # Original shape is needed by SVD.create_layer; best-effort from compressed_data
                    orig_shape = getattr(compressed_data, 'original_shape', None)
                    if orig_shape is None:
                        raise ValueError("Missing original tensor shape for SVD layer creation")
                    svd_strategy = _SVD(rank=None, preserve_energy=None)  # params unused for create_layer
                    layer = svd_strategy.create_layer(compressed_data, orig_shape)
                    return layer
                except Exception:
                    # Fallback: try constructing FactorLayer manually if u/s/vt present
                    if all(hasattr(compressed_data, attr) for attr in ("u", "s", "vt")):
                        factors = [
                            Factor(_weight=compressed_data.u.clone()),
                            Factor(_weight=(torch.diag(compressed_data.s).clone() if hasattr(compressed_data.s, 'dim') and compressed_data.s.dim() == 1 else compressed_data.s.clone())),
                            Factor(_weight=compressed_data.vt.clone()),
                        ]
                        factor_layer = FactorLayer(_factors=factors)
                        factor_layer.func_name = 'svd'
                        return factor_layer
                    raise

            # Generic low-rank path expects a list of factor tensors on the result
            factors_data = compressed_data.factors  # List of factor tensors
            factors = []
            for factor_tensor in factors_data:
                factor = Factor(_weight=factor_tensor.clone())
                factors.append(factor)
            factor_layer = FactorLayer(_factors=factors)
            factor_layer.func_name = method_name  # Set correct function name
            return factor_layer

        else:
            raise ValueError(f"Unsupported low rank compression method: {self.compression_method}. Please check the compression method.")

    # ----------------------
    # Pruning helpers
    # ----------------------

    def remove_transformer_blocks(self, model: 'torch.nn.Module', container_path: str, indices: List[int]) -> Dict[str, Any]:
        from .pruning_utils import remove_transformer_blocks as _remove
        stats = _remove(model, container_path, indices)
        self.logger.info(
            "Pruned %d/%d blocks from %s; remaining=%d",
            stats["removed_count"], stats.get("original_count", 0), container_path, stats["remaining_count"],
        )
        return stats
    
    def compress_model_with_surgery(self, model: 'torch.nn.Module', **params):
        """
        Compress model and actually replace layers with compressed versions
        
        This is the main method that combines compression with model surgery,
        enabling actual inference benefits from compression.
        
        Args:
            model: Model to compress
            **params: Compression parameters
            
        Returns:
            TensorCompressionResult with model surgery metadata
        """
        # Expand wildcard targets for logging and execution
        concrete_targets = self._expand_target_modules(model, self.target_modules)
        
        # First, compress tensors using existing logic
        compression_result = self.compress_model(model, **params)
        
        # Now perform model surgery - replace layers with compressed versions
        surgery_stats = {}
        layers_replaced = []
        
        for module_name in concrete_targets:
            try:
                # Get the original layer
                original_layer = self._get_layer_by_name(model, module_name)
                
                # Attempt to locate compressed data for this module
                module_keys = [k for k in compression_result.compressed_tensors.keys()
                               if k == module_name or k.startswith(f"{module_name}_")]
                
                if not module_keys:
                    # Nothing to replace for this module
                    continue
                
                # Vector granularity aggregation path for embeddings
                vector_keys = [k for k in module_keys if k.startswith(f"{module_name}_vector_")]
                matrix_key = next((k for k in module_keys if k.endswith("_matrix")), None)
                
                if vector_keys and hasattr(original_layer, 'num_embeddings'):
                    compressed_layer = self._build_embedding_from_vector_compressions(
                        module_name, original_layer, compression_result.compressed_tensors
                    )
                    if compressed_layer is None:
                        # Could not construct embedding; skip replacement
                        continue
                    # Replace layer in model
                    self._replace_layer_in_model(model, module_name, compressed_layer)
                else:
                    # Matrix/whole-tensor path (or fallback to first matching key)
                    chosen_key = matrix_key or module_keys[0]
                    compressed_tensor_data = compression_result.compressed_tensors[chosen_key]
                    compressed_layer = self._create_compressed_layer(original_layer, compressed_tensor_data)
                    self._replace_layer_in_model(model, module_name, compressed_layer)
                
                # Track surgery statistics
                original_params = sum(p.numel() for p in [original_layer.weight])
                if hasattr(original_layer, 'bias') and original_layer.bias is not None:
                    original_params += original_layer.bias.numel()
                
                compressed_params = sum(p.numel() for p in compressed_layer.parameters())

                surgery_stats[module_name] = {
                    "original_params": original_params,
                    "compressed_params": compressed_params,
                    "layer_type": type(compressed_layer).__name__
                }

                layers_replaced.append(module_name)
                # print(f"{module_name} replaced succesfully!")

            except (ValueError, RuntimeError, TypeError) as e:
                self.logger.error(f"Model surgery failed for {module_name}: {e}")
                raise RuntimeError(f"Failed to replace layer '{module_name}': {e}") from e
        
        # Update compression result with surgery metadata
        compression_result.parameters.update({
            "model_surgery_performed": True,
            "layers_replaced": layers_replaced,
            "surgery_stats": surgery_stats,
            "total_layers_targeted": len(concrete_targets),
            "total_layers_replaced": len(layers_replaced)
        })

        # Tag the model as compressed for downstream consumers
        if len(layers_replaced) > 0:
            try:
                info = {
                    "tag": "layer_replace",
                    "method": self.compression_method,
                    "layers_replaced_count": len(layers_replaced),
                }
                setattr(model, "compression_info", info)
                self.logger.info(
                    "Tagged model with compression_info (layers_replaced=%d)", len(layers_replaced)
                )
            except (AttributeError, TypeError) as e:
                # Fallback to simple string tag if attribute setting fails
                self.logger.warning(f"Could not set compression_info dict, using string: {e}")
                setattr(model, "compression_info", "layer_replace")

        self.logger.info(f"Layers replaced: {len(layers_replaced)}/{len(self.target_modules)}")
        self.logger.debug(f"Replaced layers: {layers_replaced}")
        
        return compression_result

    def _build_embedding_from_vector_compressions(self, module_name: str, original_layer, compressed_tensors: Dict[str, CompressedTensor]):
        """
        Aggregate per-vector compressed tensors into a FactorEmbedding using ModuleList[FactorLayer].
        Returns a new FactorEmbedding or None if aggregation is not possible.

        Raises:
            ValueError: If compressed data format is unsupported
            RuntimeError: If factor layer construction fails
        """
        from ...framework.layers import FactorLayer, FactorEmbedding, Factor
        import re

        # Collect and index vector entries for this module
        pattern = re.compile(rf"^{re.escape(module_name)}_vector_(\d+)$")
        indexed = {}
        for key, value in compressed_tensors.items():
            m = pattern.match(key)
            if m:
                idx = int(m.group(1))
                indexed[idx] = value
        if not indexed:
            return None

        # Determine expected count from original embedding
        expected = getattr(original_layer, 'num_embeddings', None)
        if expected is None and hasattr(original_layer, 'weight'):
            expected = original_layer.weight.shape[0]
        if expected is None:
            return None

        # Ensure we have all vectors
        missing = [i for i in range(expected) if i not in indexed]
        if missing:
            self.logger.warning(f"Aggregation for {module_name} incomplete; missing {len(missing)} vectors")
            return None

        # Build FactorLayer per index in order
        factors_layers = []
        for i in range(expected):
            compressed_data = indexed[i]
            # SVD per-vector path: CompressedSVDTensor has attributes u/s/vt
            if hasattr(compressed_data, 'u') and hasattr(compressed_data, 's') and hasattr(compressed_data, 'vt'):
                u, s, vt = compressed_data.u, compressed_data.s, compressed_data.vt
                s_mat = torch.diag(s) if hasattr(s, 'dim') and s.dim() == 1 else s
                factors = [Factor(_weight=u.clone()), Factor(_weight=s_mat.clone()), Factor(_weight=vt.clone())]
                factor_layer = FactorLayer(_factors=factors)
                factor_layer.func_name = 'svd'

            # Generic low-rank path with explicit factors
            elif hasattr(compressed_data, 'factors'):
                factors = []
                for factor_tensor in compressed_data.factors:
                    factor = Factor(_weight=factor_tensor.clone())
                    factors.append(factor)
                factor_layer = FactorLayer(_factors=factors)
                # Prefer method from compressed_data if present; otherwise infer by factor shapes
                method_name = getattr(compressed_data, 'method', None)
                if not method_name:
                    # Heuristic inference
                    if len(factors) == 3:
                        s_w = getattr(factors[1], 'weight', None)
                        if s_w is not None and hasattr(s_w, 'dim') and s_w.dim() in (1, 2):
                            method_name = 'svd'
                    if not method_name:
                        # If any factor is 3D, assume tensor-train; else default to tucker
                        method_name = 'tensor_train' if any(getattr(f, 'weight', torch.empty(0)).dim() == 3 for f in factors) else 'tucker'
                # Guard against mismatched labeling: if S looks 3D, don't set SVD
                if method_name == 'svd':
                    s_w = getattr(factors[1], 'weight', None)
                    if s_w is not None and hasattr(s_w, 'dim') and s_w.dim() > 2:
                        method_name = 'tensor_train'
                factor_layer.func_name = method_name
            else:
                # Not supported for aggregation - raise explicit error
                raise ValueError(
                    f"Unsupported compressed data format for vector {i} in module '{module_name}': "
                    f"expected 'u/s/vt' or 'factors' attributes, got {type(compressed_data).__name__}"
                )
            factors_layers.append(factor_layer)

        # Assemble FactorEmbedding with ModuleList of FactorLayer
        embedding = FactorEmbedding.from_pretrained(factors_layers, freeze=True)
        # Set function name based on first factor layer
        if len(factors_layers) > 0:
            embedding.func_name = getattr(factors_layers[0], 'func_name', None) or self.compression_method
        return embedding
