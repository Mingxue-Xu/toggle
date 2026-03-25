"""
Compressed model save/load helpers using safetensors with reconstruction.

This module reconstructs a baseline HF model by replacing targeted modules with
FactorLinear/FactorEmbedding stubs sized from saved safetensors, then loads weights.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import torch

from .layers import FactorLayer, FactorLinear, FactorEmbedding
from .reproducibility import config_hash, get_seed, get_reproducibility_info


@dataclass
class Manifest:
    base_model: str
    modules_replaced: List[str]
    module_types: Optional[Dict[str, str]] = None
    func_names: Optional[Dict[str, str]] = None
    # Optional explicit factor sizes for FactorLinear reconstruction
    # Map: module_path -> list of factor shapes (each shape is a list of ints)
    factor_sizes: Optional[Dict[str, List[List[int]]]] = None

    @staticmethod
    def load(path: Union[str, Path]) -> "Manifest":
        p = Path(path)
        data = json.loads(p.read_text())
        return Manifest(
            base_model=data.get("base_model") or data.get("model") or "",
            modules_replaced=data.get("modules_replaced", []),
            module_types=data.get("module_types"),
            func_names=data.get("func_names"),
            factor_sizes=data.get("factor_sizes"),
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Manifest":
        """
        Create a Manifest from a dictionary.

        Args:
            data: Dictionary with manifest fields. Supports aliases:
                  - 'model_name' or 'base_model' for the base model
                  - 'compression_method' stored in module_types
                  - 'module_paths' or 'modules_replaced' for module list

        Returns:
            Manifest instance.
        """
        base_model = data.get("base_model") or data.get("model_name") or data.get("model") or ""
        modules_replaced = data.get("modules_replaced") or data.get("module_paths") or []
        module_types = data.get("module_types")
        # Handle compression_method shorthand
        if "compression_method" in data and not module_types:
            module_types = {m: data["compression_method"] for m in modules_replaced}
        return Manifest(
            base_model=base_model,
            modules_replaced=modules_replaced,
            module_types=module_types,
            func_names=data.get("func_names"),
            factor_sizes=data.get("factor_sizes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Manifest to a dictionary.

        Returns:
            Dictionary representation of the manifest.
        """
        result: Dict[str, Any] = {
            "base_model": self.base_model,
            "modules_replaced": self.modules_replaced,
        }
        if self.module_types:
            result["module_types"] = self.module_types
        if self.func_names:
            result["func_names"] = self.func_names
        if self.factor_sizes:
            result["factor_sizes"] = self.factor_sizes
        return result


def _split_attr_and_index(token: str):
    if "[" in token and token.endswith("]"):
        name, idx = token.split("[", 1)
        idx = idx[:-1]
        if ":" in idx:
            start_s, stop_s, *_ = (idx.split(":") + [None, None])[:2]
            start = int(start_s) if start_s not in (None, "") else None
            stop = int(stop_s) if stop_s not in (None, "") else None
            return name, slice(start, stop, None)
        try:
            return name, int(idx)
        except ValueError:
            return token, None
    return token, None


def _get_module_by_path(model: torch.nn.Module, module_path: str) -> Optional[torch.nn.Module]:
    current = model
    for part in module_path.split("."):
        attr, idx = _split_attr_and_index(part)
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


def resolve_module_path(model: torch.nn.Module, module_path: str) -> Optional[torch.nn.Module]:
    """
    Resolve a dotted module path to the corresponding module in a model.

    This is the public API for nested module path resolution. Supports both
    dotted notation (e.g., "encoder.layers.0.self_attn") and bracket notation
    (e.g., "encoder.layers[0].self_attn").

    Args:
        model: The PyTorch model to traverse.
        module_path: Dotted path string to the target module.

    Returns:
        The module at the specified path, or None if not found.

    Example:
        >>> module = resolve_module_path(model, "encoder.layers.0.self_attn")
        >>> module = resolve_module_path(model, "model.layers[0].mlp.down_proj")
    """
    return _get_module_by_path(model, module_path)


def _set_module_by_path(model: torch.nn.Module, module_path: str, new_module: torch.nn.Module) -> bool:
    parts = module_path.split(".")
    parent_path = ".".join(parts[:-1])
    last = parts[-1]
    parent = model if not parent_path else _get_module_by_path(model, parent_path)
    if parent is None:
        return False
    attr, idx = _split_attr_and_index(last)
    if not hasattr(parent, attr):
        return False
    container = getattr(parent, attr)
    if isinstance(idx, int):
        if isinstance(container, (torch.nn.ModuleList, list)) and 0 <= idx < len(container):
            container[idx] = new_module  # type: ignore[index]
            return True
        return False
    else:
        setattr(parent, attr, new_module)
        return True


def _normalize_path_for_state_keys(module_path: str) -> str:
    """Convert bracket-indexed module path to state_dict key form.

    Example: "model.layers[0].self_attn.q_proj" -> "model.layers.0.self_attn.q_proj"
    """
    return re.sub(r"\[(\d+)\]", r".\\1", module_path)


def _group_factor_shapes_from_weights(weights: Dict[str, torch.Tensor], module_path: str) -> Tuple[str, Dict[int, List[torch.Size]]]:
    """Infer module type and factor shapes from safetensor keys for a given module path.

    Returns (module_type, shapes_map). For FactorLinear, shapes_map has a single entry -1.
    For FactorEmbedding, shapes_map maps token index -> list of factor shapes.
    """
    shapes: Dict[int, List[torch.Size]] = {}
    # FactorLinear keys: support both bracket and dotted index forms
    prefix_linear_a = f"{module_path}.weight.factors."
    prefix_linear_b = f"{_normalize_path_for_state_keys(module_path)}.weight.factors."
    has_linear = any((k.startswith(prefix_linear_a) or k.startswith(prefix_linear_b)) and k.endswith(".weight") for k in weights.keys())
    if has_linear:
        factors: Dict[int, torch.Size] = {}
        for k, t in weights.items():
            if (k.startswith(prefix_linear_a) or k.startswith(prefix_linear_b)) and k.endswith(".weight"):
                try:
                    if k.startswith(prefix_linear_a):
                        idx_part = k.split(prefix_linear_a, 1)[1]
                    else:
                        idx_part = k.split(prefix_linear_b, 1)[1]
                    idx = int(idx_part.split(".")[0])
                except Exception:
                    continue
                factors[idx] = t.shape
        if factors:
            shapes[-1] = [factors[i] for i in sorted(factors.keys())]
            return "FactorLinear", shapes

    # FactorEmbedding keys: support both path forms: f"{path}.weight.<i>.factors.<k>.weight"
    prefix_emb_a = f"{module_path}.weight."
    prefix_emb_b = f"{_normalize_path_for_state_keys(module_path)}.weight."
    emb_seen = False
    pattern_a = re.compile(re.escape(prefix_emb_a) + r"(\d+)\.factors\.(\d+)\.weight$")
    pattern_b = re.compile(re.escape(prefix_emb_b) + r"(\d+)\.factors\.(\d+)\.weight$")
    for k, t in weights.items():
        m = pattern_a.match(k) or pattern_b.match(k)
        if not m:
            continue
        emb_seen = True
        tok_idx = int(m.group(1))
        factor_idx = int(m.group(2))
        shapes.setdefault(tok_idx, [])
        # Ensure list long enough
        while len(shapes[tok_idx]) <= factor_idx:
            shapes[tok_idx].append(torch.Size([]))
        shapes[tok_idx][factor_idx] = t.shape
    if emb_seen:
        # Normalize to dense 0..max index
        max_idx = max(shapes.keys()) if shapes else -1
        for i in range(max_idx + 1):
            if i not in shapes:
                shapes[i] = []
        return "FactorEmbedding", shapes

    return "unknown", {}


def _detect_consolidated_embedding(weights: Dict[str, torch.Tensor], module_path: str) -> bool:
    dotted = _normalize_path_for_state_keys(module_path)
    key = f"{dotted}._consolidated_factors"
    return any(k == key for k in weights.keys())


def _reconstruct_embedding_from_consolidated(
    model: torch.nn.Module,
    module_path: str,
    weights: Dict[str, torch.Tensor],
    func_name: Optional[str],
    debug: bool = False,
) -> bool:
    """Reconstruct a FactorEmbedding at module_path using consolidated entries in weights.

    Returns True on success.
    """
    logger = logging.getLogger(__name__)
    dotted = _normalize_path_for_state_keys(module_path)
    key_base = dotted
    try:
        consolidated = weights[f"{key_base}._consolidated_factors"]
        shapes_dims = weights[f"{key_base}._shapes_dims"].tolist()
        shapes_values = weights[f"{key_base}._shapes_values"].tolist()  # list[list[int]] with padding -1
        offsets = weights[f"{key_base}._offsets"].tolist()
        num_embeddings = int(weights[f"{key_base}._num_embeddings"].item())
        factors_per_emb = int(weights[f"{key_base}._factors_per_embedding"].item())
    except KeyError:
        return False

    # Build per-token factor sizes list
    sizes_per_factor: List[List[int]] = []
    for i, dims in enumerate(shapes_dims):
        vals = [int(x) for x in shapes_values[i][:dims]]
        sizes_per_factor.append(vals)
    # Group by token
    sizes_per_token: List[List[Tuple[int, ...]]] = []
    idx = 0
    for t in range(num_embeddings):
        token_sizes: List[Tuple[int, ...]] = []
        for f in range(factors_per_emb):
            token_sizes.append(tuple(sizes_per_factor[idx]))
            idx += 1
        sizes_per_token.append(token_sizes)

    # Create FactorEmbedding with specified factor sizes
    try:
        layers: List[FactorLayer] = []
        for token_idx in range(num_embeddings):
            fl = FactorLayer(_factor_sizes=sizes_per_token[token_idx], _freeze=True)
            if func_name:
                fl.func_name = func_name
            layers.append(fl)
        emb = FactorEmbedding.from_pretrained(layers, freeze=True)
        if func_name:
            emb.func_name = func_name
        # Fill weights from consolidated tensor via offsets
        flat = consolidated.detach()
        if flat.device.type != "cpu":
            flat = flat.cpu()
        cursor = 0
        # offsets are absolute; use offsets list to slice
        idx = 0
        for token_idx in range(num_embeddings):
            fl = emb.weight[token_idx]
            for f_idx in range(factors_per_emb):
                shape = sizes_per_token[token_idx][f_idx]
                off = int(offsets[idx])
                count = int(torch.Size(shape).numel())
                slice_t = flat[off:off + count].view(shape).to(fl.factors[f_idx].weight.dtype)
                with torch.no_grad():
                    fl.factors[f_idx].weight.copy_(slice_t)
                idx += 1
        # Replace in model
        _set_module_by_path(model, module_path, emb)
        if debug:
            logger.info("[compressed_io] Reconstructed consolidated embedding at %s: tokens=%d factors/token=%d", module_path, num_embeddings, factors_per_emb)
        return True
    except Exception:
        if debug:
            logger.exception("[compressed_io] Failed to reconstruct consolidated embedding for %s", module_path)
        return False


def load_compressed_from_safetensors(
    base_model_name: str,
    save_dir: Union[str, Path],
    device: str = "cpu",
    debug: bool = False,
) -> torch.nn.Module:
    """
    Reconstruct a compressed model by:
    1) Loading the base model
    2) Replacing modules listed in manifest with Factor* stubs sized from safetensors
    3) Loading weights using safetensors
    """
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file

    save_dir = Path(save_dir)
    manifest = Manifest.load(save_dir / "manifest.json")
    weights = load_file(str(save_dir / "model.safetensors"))
    logger = logging.getLogger(__name__)
    if debug:
        logger.info("[compressed_io] Loaded manifest for base_model=%s; modules=%d", manifest.base_model, len(manifest.modules_replaced))

    # 1) Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.to(device)
    model.eval()

    # 2) Replace modules based on shapes and (optional) manifest hints
    replaced_paths: List[str] = []
    for path in manifest.modules_replaced:
        mtype_hint = (manifest.module_types or {}).get(path)
        func_hint = (manifest.func_names or {}).get(path)
        mtype, shapes_map = _group_factor_shapes_from_weights(weights, path)
        # Prefer explicit factor sizes from manifest for FactorLinear
        if (manifest.factor_sizes or {}).get(path):
            try:
                fs = manifest.factor_sizes[path]
                shapes_map = {-1: [torch.Size(list(s)) for s in fs]}
                if not mtype:
                    mtype = "FactorLinear"
                if debug:
                    logger.info("[compressed_io] Using manifest factor_sizes for %s: %s", path, fs)
            except Exception:
                if debug:
                    logger.exception("[compressed_io] Failed to parse manifest factor_sizes for %s", path)
        if mtype == "unknown" and mtype_hint:
            mtype = mtype_hint

        original = _get_module_by_path(model, path)
        if original is None:
            # Skip if path can't be resolved on baseline (model variant drift)
            continue

        if mtype == "FactorLinear":
            # Expect a single entry at key -1 with factor shapes
            factor_sizes = shapes_map.get(-1, [])
            if not factor_sizes:
                # Fallback: can't reconstruct
                if debug:
                    logger.warning("[compressed_io] Skip %s: FactorLinear shapes not found", path)
                continue
            # Debug: basic SVD sanity check
            if debug:
                try:
                    out_f = getattr(original, 'out_features', None) or original.weight.shape[0]
                    in_f = getattr(original, 'in_features', None) or original.weight.shape[1]
                    info = {
                        "in_features": int(in_f),
                        "out_features": int(out_f),
                        "num_factors": len(factor_sizes),
                        "factor_shapes": [tuple(s) for s in factor_sizes],
                    }
                    logger.info("[compressed_io] FactorLinear sanity %s: %s", path, info)
                except Exception:
                    logger.exception("[compressed_io] Failed to log FactorLinear sanity for %s", path)
            try:
                new_mod = FactorLinear(
                    in_features=getattr(original, 'in_features', None) or original.weight.shape[1],
                    out_features=getattr(original, 'out_features', None) or original.weight.shape[0],
                    _func_name=func_hint or ("svd" if len(factor_sizes) == 3 else "tensor_train"),
                    _factor_sizes=factor_sizes,
                    bias=getattr(original, 'bias', None) is not None,
                    _freeze=True,
                )
                # Also propagate func name to FactorLayer for correct contraction
                if hasattr(new_mod, 'weight') and hasattr(new_mod.weight, 'func_name'):
                    new_mod.weight.func_name = new_mod.func_name
                _set_module_by_path(model, path, new_mod)
                replaced_paths.append(path)
                if debug:
                    logger.info("[compressed_io] Replaced %s as FactorLinear: factors=%s func=%s", path, [tuple(s) for s in factor_sizes], new_mod.func_name)
            except Exception:
                if debug:
                    logger.exception("[compressed_io] Failed to reconstruct FactorLinear for %s", path)
                continue
        elif mtype == "FactorEmbedding" or _detect_consolidated_embedding(weights, path):
            # Prefer consolidated reconstruction if available
            if _detect_consolidated_embedding(weights, path):
                ok = _reconstruct_embedding_from_consolidated(model, path, weights, func_hint, debug)
                if ok:
                    replaced_paths.append(path)
                    continue
                # Fall through to non-consolidated path if failed
            try:
                num_embeddings = getattr(original, 'num_embeddings', None)
                if num_embeddings is None and hasattr(original, 'weight'):
                    num_embeddings = int(original.weight.shape[0])
                layers: List[FactorLayer] = []
                for i in range(num_embeddings):
                    sizes = shapes_map.get(i, [])
                    if not sizes:
                        sizes = [torch.Size([1])]
                    fl = FactorLayer(_factor_sizes=[tuple(s) for s in sizes], _freeze=True)
                    fl.func_name = func_hint or "tensor_train"
                    layers.append(fl)
                emb = FactorEmbedding.from_pretrained(layers, freeze=True)
                emb.func_name = func_hint or emb.func_name
                _set_module_by_path(model, path, emb)
                replaced_paths.append(path)
                if debug:
                    shapes0 = [tuple(s) for s in (shapes_map.get(0) or [])]
                    logger.info("[compressed_io] Replaced %s as FactorEmbedding (non-consolidated): token0_factors=%s func=%s", path, shapes0, emb.func_name)
            except Exception:
                if debug:
                    logger.exception("[compressed_io] Failed to reconstruct FactorEmbedding for %s", path)
                continue
        else:
            # Unknown type; skip replacement
            if debug:
                logger.warning("[compressed_io] Unknown type for %s; skip replacement", path)
            continue

    # 3) Load weights
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if debug:
        logger.info("[compressed_io] load_state_dict: replaced=%d missing=%d unexpected=%d", len(replaced_paths), len(missing), len(unexpected))
    # Require that at least some keys matched to avoid silent no-op
    if len(missing) >= len(model.state_dict()):
        raise RuntimeError("Failed to load any parameters from safetensors; reconstruction mismatch.")

    model.to(device)
    model.eval()
    return model


def save_compressed_to_safetensors(
    model: torch.nn.Module,
    manifest: Dict[str, Any],
    save_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Save a compressed model's state using safetensors with a manifest.

    - Ensures tensors are CPU + contiguous
    - Filters duplicate alias keys from FactorLayer (e.g., '.weight.weight.')
    - Breaks shared storage ties by cloning subsequent occurrences (e.g., tied embeddings)

    Returns (manifest_path, weights_path).
    """
    try:
        from safetensors.torch import save_file as _save_file
    except Exception as e:
        raise RuntimeError("safetensors is required to save compressed models. pip install safetensors") from e

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    weights_path = save_dir / "model.safetensors"

    # Add versioning and reproducibility info to manifest
    manifest_to_save = dict(manifest)
    manifest_to_save["_versioning"] = {
        "config_hash": config_hash(manifest),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "seed": get_seed(),
        "reproducibility": get_reproducibility_info(),
    }

    # Write manifest
    import json as _json
    manifest_path.write_text(_json.dumps(manifest_to_save, indent=2))

    # Identify FactorEmbedding modules for consolidation
    fe_paths: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, FactorEmbedding):
            fe_paths.append(name)

    fe_dotted = fe_paths  # named_modules already uses dotted form

    # Build safe state dict with consolidation for embeddings
    state: Dict[str, torch.Tensor] = {}
    seen_ptrs: Dict[int, str] = {}
    for k, v in model.state_dict().items():
        # Drop duplicate alias created by FactorLayer's weight=factors: keep 'weight.factors', drop 'weight.weight'
        if ".weight.weight." in k:
            continue
        # Skip per-factor keys for consolidated embedding modules
        if any(k.startswith(f"{p}.weight.") for p in fe_dotted):
            continue
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu()
            if not t.is_contiguous():
                t = t.contiguous()
            # Break shared pointers by cloning subsequent occurrences
            try:
                ptr = t.untyped_storage().data_ptr() if hasattr(t, 'untyped_storage') else t.data_ptr()
            except Exception:
                ptr = id(t)
            if ptr in seen_ptrs:
                t = t.clone().contiguous()
                ptr = t.untyped_storage().data_ptr() if hasattr(t, 'untyped_storage') else t.data_ptr()
            else:
                seen_ptrs[ptr] = k
            state[k] = t
        else:
            state[k] = v

    # Add consolidated entries for each FactorEmbedding
    for p in fe_dotted:
        # Resolve module by path
        mod = _get_module_by_path(model, p)
        if not isinstance(mod, FactorEmbedding):
            continue
        try:
            # Build consolidated tensor, shapes and offsets in token-major, factor-minor order
            flat_list: List[torch.Tensor] = []
            shapes_list: List[List[int]] = []
            dims_list: List[int] = []
            offsets_list: List[int] = []
            cur = 0
            num_embeddings = len(mod.weight)
            factors_per_emb = len(mod.weight[0].factors) if num_embeddings > 0 else 0
            for token_idx in range(num_embeddings):
                fl = mod.weight[token_idx]
                for f in fl.factors:
                    w = f.weight.detach().cpu().contiguous()
                    flat_list.append(w.reshape(-1))
                    shape_vals = list(w.shape)
                    shapes_list.append(shape_vals)
                    dims_list.append(len(shape_vals))
                    offsets_list.append(cur)
                    cur += w.numel()
            consolidated = torch.cat(flat_list, dim=0)
            max_dim = max(dims_list) if dims_list else 0
            # Build shapes_values padded with -1
            if max_dim > 0:
                shapes_values = torch.full((len(shapes_list), max_dim), -1, dtype=torch.long)
                for i, shp in enumerate(shapes_list):
                    shapes_values[i, : len(shp)] = torch.tensor(shp, dtype=torch.long)
                shapes_dims = torch.tensor(dims_list, dtype=torch.long)
            else:
                shapes_values = torch.empty((0, 0), dtype=torch.long)
                shapes_dims = torch.empty((0,), dtype=torch.long)
            offsets = torch.tensor(offsets_list, dtype=torch.long)
            state[f"{p}._consolidated_factors"] = consolidated
            state[f"{p}._shapes_values"] = shapes_values
            state[f"{p}._shapes_dims"] = shapes_dims
            state[f"{p}._offsets"] = offsets
            state[f"{p}._num_embeddings"] = torch.tensor([num_embeddings], dtype=torch.long)
            state[f"{p}._factors_per_embedding"] = torch.tensor([factors_per_emb], dtype=torch.long)
        except Exception as e:
            # If consolidation fails, fall back to regular per-factor save (already included above)
            # Log via print to avoid introducing logging dependency here
            print(f"[compressed_io] Warning: failed to consolidate embedding at {p}: {e}")

    _save_file(state, str(weights_path))
    return manifest_path, weights_path
