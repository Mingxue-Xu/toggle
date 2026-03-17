"""
Neural Network Layer Components for Toggle

Core tensor decomposition layer implementations adapted from the original Toggle codebase.
Provides FactorLayer, FactorEmbedding, FactorLinear, and Factor classes for tensor-train,
Tucker, and CP decompositions.
"""

import numpy as np
import torch
import math
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module, ModuleDict, Embedding, init, Linear, ModuleList
from typing import Dict, Union, List, Tuple, Optional, OrderedDict, Callable, AnyStr
from collections.abc import Mapping
import tensorly as tl
from tensorly._factorized_tensor import FactorizedTensor
from tensorly.tt_tensor import TTTensor
from tensorly.cp_tensor import CPTensor
from tensorly.tucker_tensor import TuckerTensor
from tensorly.decomposition import tucker, tensor_train
from ..plugins.compression.svd import CompressedSVDTensor



FACTOR_NAME = {
    'tensor_train': ['factors'],
    'tucker': ['core', 'factors'],
    'cp': ['weights', 'factors'],
    'svd':['u','s','vt']
}


class Factor(Module):
    """Individual factor component for tensor decomposition"""
    __constants__ = ['factor_size']

    def __init__(
            self,
            _weight: Optional[Tensor] = None,
            _size: Optional[Union[List, Dict, torch.Size]] = None,
            _freeze: bool = False,
            # Compatibility aliases
            weight: Optional[Tensor] = None,
            name: Optional[str] = None,
            size: Optional[Union[List, Dict, torch.Size]] = None,
            freeze: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Handle compatibility aliases
        if weight is not None and _weight is None:
            _weight = weight
        if size is not None and _size is None:
            _size = size
        if freeze is not None:
            _freeze = freeze

        # Store optional name for identification
        self.name = name

        if _weight is not None:
            if isinstance(_weight, Tensor):
                # If weight is provided, use it directly
                self.weight = Parameter(data=_weight.detach(), requires_grad=not _freeze)
                self.factor_size = self.weight.size()
            elif _weight is not None:
                print(f"weight type: {type(_weight)}")
                self.weight = _weight
                self.factor_size = self.weight.size()
            else:
                raise NotImplementedError("Neither weight nor factor size is given!")
                
        elif _size is not None:
            # Convert size to tuple for torch.empty
            if isinstance(_size, (list, tuple)):
                size_tuple = tuple(_size)
            elif isinstance(_size, torch.Size):
                size_tuple = tuple(_size)
            else:
                raise ValueError(f"Unsupported size type: {type(_size)}")
            
            # Create empty tensor with proper initialization
            self.weight = Parameter(torch.empty(size_tuple), requires_grad=not _freeze)
            self.factor_size = size_tuple
            
            # Initialize the weight properly
            self._init_weight()
        else:
            raise NotImplementedError("Neither weight nor factor size is given!")

    def _init_weight(self):
        """Initialize the weight parameter properly"""
        if self.weight is not None:
            # Use Xavier/Glorot initialization for better training
            if len(self.weight.shape) == 2:
                # For 2D tensors (matrices)
                torch.nn.init.xavier_uniform_(self.weight)
            elif len(self.weight.shape) == 3:
                # For 3D tensors, initialize each slice
                for i in range(self.weight.shape[0]):
                    torch.nn.init.xavier_uniform_(self.weight[i])
            else:
                # For higher dimensions, use normal initialization
                torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def extra_repr(self) -> str:
        return f'size={self.weight.size()}, requires_grad={self.weight.requires_grad}'


class FactorLayer(Module):
    """Core factor layer for tensor decomposition operations"""
    
    __constants__ = ['num_factors', 'factor_sizes', 'func_name']

    num_factors: int
    factor_sizes: Optional[List] = None
    func_name: Optional[str] = None

    def __init__(self,
                 _factors: Optional[List] = None,
                 _freeze: bool = False,
                 _factor_sizes: Optional[List] = None) -> None:
        super().__init__()
        self.factor_sizes = None
        self.num_factors = 0
        self.func_name = None

        if _factors is not None:
            if isinstance(_factors, FactorizedTensor):
                if isinstance(_factors, TTTensor):
                    self.factors = ModuleList([Factor(_weight=_factors.factors[k]) for k in range(len(_factors.factors))])
                    self.func_name = 'tensor_train'
                    self.num_factors = len(_factors.factors)
                elif isinstance(_factors, TuckerTensor):
                    core, factors = _factors
                    self.factors = ModuleList([Factor(_weight=core)])
                    for i in range(len(factors)):
                        self.factors.append(Factor(_weight=factors[i]))
                    self.func_name = 'tucker'
                    self.num_factors = len(self.factors)
                elif isinstance(_factors, CPTensor):
                    weights, factors = _factors
                    self.factors = ModuleList([Factor(_weight=weights)])
                    for i in range(len(factors)):
                        self.factors.append(Factor(_weight=factors[i]))
                    self.func_name = 'cp'
                    self.num_factors = len(self.factors)
                elif isinstance(_factors, CompressedSVDTensor):
                    self.factors = ModuleList([Factor(_weight=_factors.u),
                                               Factor(_weight=_factors.s),
                                               Factor(_weight=_factors.vt)])
                    self.func_name = 'svd'
                    self.num_factors = 3
                else:
                    raise NotImplementedError(f"Unsupported FactorizedTensor type: {type(_factors)}")
            else:
                # _factors is a regular list of factors
                self.factors = ModuleList(_factors)
                self.num_factors = len(_factors)
                self.func_name = 'tensor_train'  # Default assumption

        elif _factor_sizes is not None:
            # Create factors and register them properly
            self.factors = ModuleList([Factor(_size=size_t, _freeze=_freeze) for size_t in _factor_sizes])
            self.factor_sizes = _factor_sizes
            self.num_factors = len(_factor_sizes)
            self.func_name = 'tensor_train'  # Default to tensor train
        else:
            raise NotImplementedError("Neither of factors nor factor_sizes was given!")
        
        # Call post_init to ensure proper initialization
        self.post_init()

        # Create a pointer self.weight, pointing to self.factors (not a deep copy, just a reference)
        self.weight = self.factors

    def get_parameter(self, target: str) -> Dict:
        """
        Override from torch.nn.Module.get_parameter() 
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L669
        """
        module_path, _, param_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: Dict = getattr(mod, param_name)

        return param
    
    def get_parameter_count(self) -> int:
        return sum(factor.weight.numel() for factor in self.factors)

    def contract(self, func_name: Optional[str] = None) -> Tensor:
        """Contract factors according to decomposition type"""
        if func_name is None:
            assert self.func_name is not None, f"{self.__class__.__name__}: No decomposition approach clarified!"
            func_name = self.func_name

        if func_name == "tensor_train":
            # For tensor train, contract all factors
            if self.factors is not None:
                # Get all factor weights
                return self._contract_tensor_train(self.factors)
            else:
                raise ValueError("Invalid factors for tensor_train contraction")
        elif func_name == "tucker":
            # For Tucker decomposition
            if self.factors is not None and len(self.factors) >= 2:
                return self._contract_tucker(self.factors)
            else:
                raise ValueError("Invalid factors for tucker contraction")
        elif func_name == "cp":
            # For CP decomposition
            if self.factors is not None and len(self.factors) >= 2:
                return self._contract_cp(self.factors)
            else:
                raise ValueError("Invalid factors for cp contraction")
        elif func_name == "svd":
            # For SVD decomposition
            if self.factors is not None and len(self.factors) == 3:
                return self._contract_svd(self.factors)
            else:
                raise ValueError("Invalid factors for svd contraction")
        else:
            raise NotImplementedError(f"Contraction for {func_name} hasn't been implemented")

    def _contract_tensor_train(self, factors: ModuleList) -> Tensor:
        """Contract tensor train factors"""

        factor_weights = [factor.weight for factor in self.factors]
        if len(factor_weights) == 0:
            raise ValueError("No factors to contract")
        
        result = torch.flatten(tl.tt_to_tensor(factor_weights))

        return result
    
    def _contract_tucker(self, factors: ModuleList) -> Tensor:
        """Contract Tucker decomposition factors"""
        if len(factors) < 2:
            raise ValueError("Tucker decomposition needs at least core and one factor")
        
        # First factor is the core tensor, rest are factors
        core = factors[0].weight
        factor_matrices = [factors[i].weight for i in range(1, len(factors))]
        
        # Reconstruct tensor using Tucker format
        result = torch.flatten(tl.tucker_to_tensor((core, factor_matrices)))
        
        return result
    
    def _contract_cp(self, factors: ModuleList) -> Tensor:
        """Contract CP decomposition factors"""
        if len(factors) < 2:
            raise ValueError("CP decomposition needs at least weights and one factor")
        
        # First factor contains weights, rest are factor matrices
        weights = factors[0].weight
        factor_matrices = [factors[i].weight for i in range(1, len(factors))]
        
        # Reconstruct tensor using CP format
        result = torch.flatten(tl.cp_to_tensor((weights, factor_matrices)))
        
        return result
    
    def _contract_svd(self, factors: ModuleList) -> Tensor:
        """Contract SVD decomposition factors"""
        if len(factors) != 3:
            raise ValueError("SVD decomposition needs exactly 3 factors: U, S, Vt")
        
        u = factors[0].weight
        s = factors[1].weight
        vt = factors[2].weight
        
        # Reconstruct matrix: A = U * S * Vt
        if s.dim() == 1:
            S = torch.diag(s)
        elif s.dim() == 2:
            S = s
        else:
            raise ValueError(f"Invalid S shape for SVD contraction: {tuple(s.shape)}")
        reconstructed = u @ S @ vt
        
        return torch.flatten(reconstructed)

    def post_init(self):
        """
        Post-initialization method called after weight assignment.
        Ensures all factors are properly initialized and on the correct device.
        """
        if hasattr(self, 'factors') and self.factors is not None:
            for factor in self.factors:
                if hasattr(factor, 'post_init'):
                    factor.post_init()
                
                # Ensure factor weights are properly initialized
                if hasattr(factor, 'weight') and factor.weight is not None:
                    # Check if weight needs initialization
                    try:
                        if hasattr(factor, 'weight') and factor.weight.numel() > 0:
                            # Weight is already set, just ensure it's on the right device
                            pass
                        else:
                            # Weight needs initialization
                            if hasattr(factor, '_init_weight'):
                                factor._init_weight()
                    except Exception:
                        # If there's any issue with individual factor, try to initialize all factors
                        for f in self.factors:
                            if hasattr(f, 'weight') and hasattr(f.weight, 'numel'):
                                if f.weight.numel() > 0:
                                    pass
                                else:
                                    if hasattr(f, '_init_weight'):
                                        f._init_weight()

    def extra_repr(self) -> str:
        param_count = self.get_parameter_count()
        return f'num_factors={self.num_factors}, func_name={self.func_name}, parameters={param_count:,}'


class FactorEmbedding(Module):
    """Factorized embedding layer using tensor decomposition"""
    
    __constants__ = ['num_embeddings']
    num_embeddings: int
    func_name: str

    def __init__(self, _num_embeddings: int,
                 _func_name: str = 'tensor_train',
                 _weight: Optional[Union[List, ModuleList]] = None,
                 _factor_sizes: Optional[Union[List, Tuple]] = None,
                 device=None,
                 dtype=None,
                 _freeze: bool = True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert _weight is not None or _factor_sizes is not None, f"Either weights or factor sizes should be given."

        super().__init__()
        
        # Set tensorly backend to pytorch once for all operations
        tl.set_backend('pytorch')

        self.num_embeddings = _num_embeddings
        self.func_name = _func_name

        if _weight is not None:
            self.weight = _weight if isinstance(_weight, ModuleList) else ModuleList(_weight)
            for k in range(len(self.weight)):
                # Check if the weight is already a FactorLayer
                if isinstance(self.weight[k], FactorLayer):
                    # Already a FactorLayer, just update freeze status if needed
                    if _freeze:
                        for factor in self.weight[k].factors:
                            if hasattr(factor, 'weight'):
                                factor.weight.requires_grad_(False)
                elif isinstance(self.weight[k], (ModuleList, list)):
                    # Create FactorLayer from list of factors
                    self.weight[k] = FactorLayer(_factors=list(self.weight[k]), _freeze=_freeze)
                else:
                    # Single factor, wrap in FactorLayer
                    self.weight[k] = FactorLayer(_factors=[self.weight[k]], _freeze=_freeze)
        else:
            # Initialize zero weights for each embedding
            assert _factor_sizes is not None, "_factor_sizes must be provided if _weight is None"
            self.weight = ModuleList()
            for _ in range(_num_embeddings):
                # For each embedding, create a FactorLayer with zero-initialized factors
                factor_layers = []
                for size in _factor_sizes:
                    # Create a zero tensor of the given size
                    zero_tensor = torch.zeros(size, device=device, dtype=dtype)
                    factor = Factor(_weight=zero_tensor, _freeze=_freeze)
                    factor_layers.append(factor)
                factor_layer = FactorLayer(_factors=factor_layers, _freeze=_freeze)
                self.weight.append(factor_layer)

        self.post_init()

    def get_parameter_count(self) -> int:
        """Get the total number of parameters in this FactorEmbedding"""
        total_params = 0
        for factor_layer in self.weight:
            if hasattr(factor_layer, 'get_parameter_count'):
                total_params += factor_layer.get_parameter_count()
            else:
                total_params += sum(p.numel() for p in factor_layer.parameters())
        return total_params

    def forward(self, input: Tensor):
        output = []
        for input_t in input:
            output_t = []
            if input_t.ndim == 0:
                idx_t = [str(input_t.item())]
            else:
                idx_t = [str(x.item()) for x in input_t]

            for k in idx_t:
                if k in self.weight._modules:
                    factor_layer = self.weight._modules[k]
                    params_t = factor_layer.contract()
                    output_t.append(params_t)
                else:
                    raise ValueError(f"Index {k} not found in embeddings")
            
            output.append(torch.stack(output_t))
        res = torch.stack(output)
        return res

    @staticmethod
    def tt_factors_to_embeds_vec(tt_factors: List):
        """
        Convert tensor train factors to embeddings using proper tensor train contraction.
        
        Args:
            tt_factors: List of lists, where each inner list contains factors for one token
                       Each factor should be a tensor
        """
        batch_embeds = []
        batch_size = len(tt_factors)
        tokens_num = len(tt_factors[0])

        for i in range(batch_size):
            full_tensor = []
            for j in range(tokens_num):
                factors = tt_factors[i][j]
                if isinstance(factors, (list, tuple)):
                    # Use tensorly for proper tensor train contraction
                    if len(factors) == 0:
                        raise ValueError("Empty factors list")
                    
                    # Use tensorly's tt_to_tensor for proper contraction
                    tt = torch.flatten(tl.tt_to_tensor(factors))
                else:
                    # If it's already a tensor, just flatten it
                    tt = torch.flatten(factors)
                
                full_tensor.append(tt)
            batch_embeds.append(torch.stack(full_tensor))

        return torch.stack(batch_embeds)

    @staticmethod
    def tucker_factors_to_embeds_vec(tucker_factors: List):
        """
        Convert Tucker decomposition factors to embeddings using proper Tucker contraction.
        
        Args:
            tucker_factors: List of lists, where each inner list contains [core, factor_matrices] for one token
                           Core tensor and factor matrices should be tensors
        """
        batch_embeds = []
        batch_size = len(tucker_factors)
        tokens_num = len(tucker_factors[0])

        for i in range(batch_size):
            full_tensor = []
            for j in range(tokens_num):
                factors = tucker_factors[i][j]
                if isinstance(factors, (list, tuple)) and len(factors) >= 2:
                    # First element is core, rest are factor matrices
                    core = factors[0]
                    factor_matrices = factors[1:]
                    
                    # Use tensorly's tucker_to_tensor for proper contraction
                    tucker_tensor = torch.flatten(tl.tucker_to_tensor((core, factor_matrices)))
                else:
                    # If it's already a tensor, just flatten it
                    tucker_tensor = torch.flatten(factors)
                
                full_tensor.append(tucker_tensor)
            batch_embeds.append(torch.stack(full_tensor))

        return torch.stack(batch_embeds)

    @staticmethod
    def cp_factors_to_embeds_vec(cp_factors: List):
        """
        Convert CP decomposition factors to embeddings using proper CP contraction.
        
        Args:
            cp_factors: List of lists, where each inner list contains [weights, factor_matrices] for one token
                       Weights and factor matrices should be tensors
        """
        batch_embeds = []
        batch_size = len(cp_factors)
        tokens_num = len(cp_factors[0])

        for i in range(batch_size):
            full_tensor = []
            for j in range(tokens_num):
                factors = cp_factors[i][j]
                if isinstance(factors, (list, tuple)) and len(factors) >= 2:
                    # First element is weights, rest are factor matrices
                    weights = factors[0]
                    factor_matrices = factors[1:]
                    
                    # Use tensorly's cp_to_tensor for proper contraction
                    cp_tensor = torch.flatten(tl.cp_to_tensor((weights, factor_matrices)))
                else:
                    # If it's already a tensor, just flatten it
                    cp_tensor = torch.flatten(factors)
                
                full_tensor.append(cp_tensor)
            batch_embeds.append(torch.stack(full_tensor))

        return torch.stack(batch_embeds)

    @staticmethod
    def svd_factors_to_embeds(svd_factors: List):
        """
        Convert SVD decomposition factors to embeddings using proper SVD reconstruction.
        
        Args:
            svd_factors: List of lists, where each inner list contains [U, S, Vt] for one token
                        U, S, Vt should be tensors
        """
        batch_embeds = []
        batch_size = len(svd_factors)
        tokens_num = len(svd_factors[0])

        for i in range(batch_size):
            full_tensor = []
            for j in range(tokens_num):
                factors = svd_factors[i][j]
                if isinstance(factors, (list, tuple)) and len(factors) == 3:
                    # SVD factors: U, S, Vt
                    u, s, vt = factors
                    
                    # Reconstruct matrix: A = U * S * Vt
                    reconstructed = u @ torch.diag(s.flatten()) @ vt
                    svd_tensor = torch.flatten(reconstructed)
                else:
                    # If it's already a tensor, just flatten it
                    svd_tensor = torch.flatten(factors)
                
                full_tensor.append(svd_tensor)
            batch_embeds.append(torch.stack(full_tensor))

        return torch.stack(batch_embeds)

    def contract(self, factors: List):
        if self.func_name == "tensor_train":
            return self.tt_factors_to_embeds_vec(factors)
        elif self.func_name == "tucker":
            return self.tucker_factors_to_embeds_vec(factors)
        elif self.func_name == "cp":
            return self.cp_factors_to_embeds_vec(factors)
        elif self.func_name == "svd":
            return self.svd_factors_to_embeds(factors)
        else:
            raise NotImplementedError(f"Contract method for {self.func_name} not implemented")

    def set_weight(self, weight: Union[List, ModuleList]):
        # The assertion process is omitted to speed up.
        if isinstance(weight, ModuleList):
            self.weight = weight
        elif isinstance(weight, List):
            self.weight = ModuleList(weight)
        else:
            raise NotImplementedError

    @classmethod
    def from_pretrained(cls, embeddings: Union[List, ModuleList], freeze: bool = True, device=None, dtype=None):
        embedding_module = cls(
            _num_embeddings=len(embeddings),
            _weight=embeddings,
            device=device,
            dtype=dtype,
            _freeze=freeze
        )
        return embedding_module

    def extra_repr(self) -> str:
        param_count = self.get_parameter_count()
        return f'num_embeddings={self.num_embeddings}, func_name={self.func_name}, parameters={param_count:,}'

    def post_init(self):
        if hasattr(self, 'weight') and self.weight is not None:
            for factor_layer in self.weight:
                # First call post_init on the factor layer
                if hasattr(factor_layer, 'post_init'):
                    factor_layer.post_init()


class FactorLinear(Module):
    """Factorized linear layer using tensor decomposition"""
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    func_name: str

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 _func_name: str = 'tensor_train',
                 _weight: Optional[Union[List, FactorLayer]] = None,
                 _factor_sizes: Optional[Union[List, Tuple]] = None,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 _freeze: bool = True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.func_name = _func_name

        if _weight is not None:
            if isinstance(_weight, FactorLayer):
                self.weight = _weight
            elif isinstance(_weight, (list, ModuleList)):
                self.weight = FactorLayer(_factors=list(_weight), _freeze=_freeze)
            else:
                raise ValueError(f"Unsupported weight type: {type(_weight)}")
        elif _factor_sizes is not None:
            # Create FactorLayer with specified factor sizes
            self.weight = FactorLayer(_factor_sizes=_factor_sizes, _freeze=_freeze)
        else:
            raise NotImplementedError("Either weight or factor_sizes must be provided")

        # Optional bias
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self._init_bias()
        else:
            self.register_parameter('bias', None)

        self.post_init()

    def _init_bias(self):
        """Initialize bias parameter"""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            init.uniform_(self.bias, -bound, bound)

    def get_parameter_count(self) -> int:
        """Get the total number of parameters in this FactorLinear"""
        total_params = 0
        if hasattr(self.weight, 'get_parameter_count'):
            total_params += self.weight.get_parameter_count()
        else:
            total_params += sum(p.numel() for p in self.weight.parameters())
        
        if self.bias is not None:
            total_params += self.bias.numel()
        
        return total_params

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through factorized linear layer"""
        # Contract the factorized weight into a full weight matrix
        weight_matrix = self.weight.contract()
        
        # Reshape weight to proper linear layer dimensions
        weight_matrix = weight_matrix.view(self.out_features, self.in_features)
        
        # Apply linear transformation
        output = F.linear(input, weight_matrix, self.bias)
        
        return output

    def post_init(self):
        """Post-initialization for weight factors"""
        if hasattr(self.weight, 'post_init'):
            self.weight.post_init()

    def extra_repr(self) -> str:
        param_count = self.get_parameter_count()
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'func_name={self.func_name}, parameters={param_count:,}, bias={self.bias is not None}'

    @classmethod
    def from_pretrained(cls, linear_layer: Linear,
                       decomposition_method: str = 'tensor_train',
                       factor_sizes: Optional[List] = None,
                       freeze: bool = True) -> 'FactorLinear':
        """
        Create a FactorLinear from a pretrained Linear layer

        Args:
            linear_layer: The original Linear layer
            decomposition_method: Type of tensor decomposition to use
            factor_sizes: Factor sizes for decomposition
            freeze: Whether to freeze the factorized weights
        """
        if factor_sizes is None:
            raise ValueError("factor_sizes must be provided for decomposition")

        # Create the factorized linear layer
        factor_linear = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            _func_name=decomposition_method,
            _factor_sizes=factor_sizes,
            bias=linear_layer.bias is not None,
            _freeze=freeze
        )

        # Copy bias if it exists
        if linear_layer.bias is not None:
            factor_linear.bias.data.copy_(linear_layer.bias.data)

        return factor_linear

    @classmethod
    def from_linear(cls, original: Linear,
                    rank: Optional[int] = None,
                    method: str = 'svd',
                    freeze: bool = True) -> 'FactorLinear':
        """
        Create a FactorLinear from an existing Linear layer using SVD or other decomposition.

        This is an alias/convenience method that wraps from_pretrained with simpler parameters.

        Args:
            original: The original Linear layer to decompose.
            rank: The rank for low-rank decomposition. If None, uses min(in, out) // 2.
            method: Decomposition method ('svd', 'tensor_train', etc.).
            freeze: Whether to freeze the factorized weights.

        Returns:
            FactorLinear instance with decomposed weights.
        """
        in_features = original.in_features
        out_features = original.out_features

        if rank is None:
            rank = min(in_features, out_features) // 2
        rank = max(1, min(rank, min(in_features, out_features)))

        # Determine factor sizes based on method
        if method == 'svd':
            # SVD: W = U @ S @ Vt, where U is (out, rank), S is (rank,), Vt is (rank, in)
            factor_sizes = [
                (out_features, rank),  # U
                (rank,),               # S (diagonal)
                (rank, in_features),   # Vt
            ]
            decomposition_method = 'svd'
        else:
            # For tensor_train or other methods, use simple 2-factor decomposition
            factor_sizes = [
                (out_features, rank),
                (rank, in_features),
            ]
            decomposition_method = method

        return cls.from_pretrained(
            linear_layer=original,
            decomposition_method=decomposition_method,
            factor_sizes=factor_sizes,
            freeze=freeze,
        )
