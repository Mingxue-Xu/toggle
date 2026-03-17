from __future__ import annotations

from typing import Any, Dict, List, Optional


def _split_attr_and_index(token: str) -> tuple:
    if '[' in token and token.endswith(']'):
        name, idx = token.split('[', 1)
        idx = idx[:-1]  # drop ']'
        if idx == '*':
            return name, '*'
        if ':' in idx:
            start_s, stop_s, *rest = (idx.split(':') + [None, None])[:2]
            start = int(start_s) if start_s not in (None, '') else None
            stop = int(stop_s) if stop_s not in (None, '') else None
            return name, slice(start, stop, None)
        try:
            return name, int(idx)
        except ValueError:
            return token, None
    return token, None


def _resolve_parent_and_name(model: Any, path: str):
    parts = path.split('.')
    parent = model
    for part in parts[:-1]:
        attr, idx = _split_attr_and_index(part)
        child = getattr(parent, attr)
        if isinstance(idx, int):
            parent = child[idx]
        else:
            parent = child
    final_attr, final_idx = _split_attr_and_index(parts[-1])
    return parent, final_attr, final_idx


def remove_transformer_blocks(model: Any, container_path: str, indices: List[int]) -> Dict[str, Any]:
    parent, attr, idx = _resolve_parent_and_name(model, container_path)
    if idx is not None:
        raise ValueError("container_path should reference a container attribute, not an indexed element")

    container = getattr(parent, attr)

    try:
        from torch.nn import ModuleList as TorchModuleList  # type: ignore
        is_modulelist = isinstance(container, TorchModuleList)
    except Exception:
        is_modulelist = False

    if not (is_modulelist or isinstance(container, (list, tuple))):
        raise ValueError(f"Container at '{container_path}' is not a list-like of blocks: {type(container).__name__}")

    length = len(container)
    to_remove = sorted(set(int(i) for i in indices if 0 <= int(i) < length))
    keep = [i for i in range(length) if i not in to_remove]

    if is_modulelist:
        from torch.nn import ModuleList as TorchModuleList  # type: ignore
        new_container = TorchModuleList([container[i] for i in keep])
    else:
        new_container = [container[i] for i in keep]

    setattr(parent, attr, new_container)

    return {
        "container_path": container_path,
        "removed_indices": to_remove,
        "removed_count": len(to_remove),
        "remaining_count": len(keep),
        "original_count": length,
    }

