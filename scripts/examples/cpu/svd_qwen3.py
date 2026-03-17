#!/usr/bin/env python3
"""
Unified example: Activation-driven SVD on Qwen3-0.6B with optional CoLA backend.

Usage:
  HF_TOKEN=... python scripts/examples/svd_qwen3.py [--cola yes|no] [--config path] [--workspace path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM as AutoModel, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.framework.memory_profiler import MemoryProfiler
from src.framework.layers import FactorLinear
from src.plugins.compression.consolidator import ModelConsolidator


def add_project_root_to_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def random_ids_inputs(model, batch_size: int, seq_len: int, device: str):
    vocab = getattr(getattr(model, "config", None), "vocab_size", 32000)
    input_ids = torch.randint(low=0, high=int(vocab), size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def mp_to_dict(mp: MemoryProfiler) -> dict:
    return {name: profile.to_dict() for name, profile in (mp.profiles or {}).items()}


def _configure_linalg_backend():
    preferred = os.environ.get("TORCH_LINALG_PREFERRED")
    if preferred:
        torch.backends.cuda.preferred_linalg_library(preferred)
        print(f"[linalg] TORCH_LINALG_PREFERRED={preferred} -> set preferred backend")
    else:
        current = torch.backends.cuda.preferred_linalg_library()
        print(f"[linalg] using default backend: {current}")
    return torch.backends.cuda.preferred_linalg_library()


def _run_with_linalg_retry(fn, *, fallback_env="TORCH_LINALG_FALLBACK"):
    try:
        return fn()
    except RuntimeError as exc:
        msg = str(exc)
        lowered = msg.lower()
        if "cusolver" in lowered or "cusolverdncreate" in lowered:
            fallback = os.environ.get(fallback_env, "magma")
            print(f"[linalg] detected cuSolver failure: {msg}")
            if fallback:
                torch.backends.cuda.preferred_linalg_library(fallback)
                print(f"[linalg] retrying with fallback backend={fallback} (set {fallback_env} to override)")
                return fn()
        raise


def _parse_yes_no(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"Unrecognized yes/no value: {value}")


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _select_workspace(runtime_cfg: dict, use_cola: bool, override: str | None) -> Path:
    if override:
        return _resolve_path(override)
    log_root = runtime_cfg.get("log_root", "logs")
    log_root_path = _resolve_path(log_root)
    names = runtime_cfg.get("workspace_names", {}) or {}
    default_name = "examples/qwen3_svd_cola" if use_cola else "examples/qwen3_svd"
    workspace_name = names.get("cola" if use_cola else "non_cola", default_name)
    return (log_root_path / workspace_name).resolve()


def _select_save_dir(cfg: dict, workspace: Path, compressed_subdir: str) -> Path:
    save_cfg = cfg.get("save", {}) or {}
    compressed_dir = save_cfg.get("compressed_dir")
    if compressed_dir:
        return _resolve_path(compressed_dir)
    return (workspace / compressed_subdir).resolve()


def _select_backend(use_cola: bool, svd_cfg: dict) -> tuple[str, dict]:
    backend = svd_cfg.get("backend_when_cola" if use_cola else "backend_when_no_cola")
    if not backend:
        backend = svd_cfg.get("backend", "cola" if use_cola else "torch")
    base_cfg = dict(svd_cfg.get("backend_config") or {})
    specific = svd_cfg.get("cola" if use_cola else "torch") or svd_cfg.get("backend_config_no_cola") or {}
    merged_cfg = {**base_cfg, **specific}
    if use_cola and not merged_cfg:
        merged_cfg = {"algorithm": "dense"}
    return backend, merged_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-driven SVD on Qwen3-0.6B")
    parser.add_argument("--config", default="config/svd_qwen3.yaml", help="Path to configuration YAML.")
    parser.add_argument(
        "--cola",
        choices=["yes", "no"],
        default=None,
        help="Use CoLA backend (yes/no). CLI takes precedence over config; default is yes.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory for logs and outputs (otherwise derived from config).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    runtime_cfg = cfg.get("runtime", {}) or {}

    add_project_root_to_sys_path()

    use_cola = _parse_yes_no(args.cola)
    if use_cola is None:
        use_cola = _parse_yes_no(runtime_cfg.get("use_cola_default"))
    if use_cola is None:
        use_cola = True

    workspace = _select_workspace(runtime_cfg, use_cola, args.workspace)
    compressed_subdir = runtime_cfg.get("compressed_subdir", "compressed")
    save_dir = _select_save_dir(cfg, workspace, compressed_subdir)
    workspace.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    import transformers.modeling_utils as _mu
    if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
        _mu.ALL_PARALLEL_STYLES = {"colwise", "rowwise", "sequence", "context"}

    _configure_linalg_backend()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tok

    ids_cfg = (cfg.get("memory_inference", {}) or {}).get("input", {})
    bs = int(ids_cfg.get("batch_size", 1))
    seqlen = int(ids_cfg.get("seq_len", 128))

    print(f"input_batch_size: {bs}")
    print(f"input_seq_len: {seqlen}")

    t0 = time.perf_counter()
    original_params = params_count(model)
    mp = MemoryProfiler()
    inputs = random_ids_inputs(model, bs, seqlen, device)
    t_base0 = time.perf_counter()
    with mp.profile_execution("baseline_inference"):
        with torch.no_grad():
            _ = model(**inputs)
    t_base1 = time.perf_counter()
    baseline_report = mp_to_dict(mp)

    fact_cfg = cfg.get("factorization", {})
    objects = fact_cfg.get("objects", [])
    svd_cfg = (cfg.get("compression") or {}).get("svd", {}) or {}
    svd_backend, svd_backend_config = _select_backend(use_cola, svd_cfg)
    consolidator = ModelConsolidator(
        compression_method=fact_cfg.get("func_name", "svd"),
        target_modules=objects,
        method_overrides=fact_cfg.get("overrides"),
        rank=fact_cfg.get("rank", fact_cfg.get("svd_rank", 8)),
        svd_backend=svd_backend,
        svd_backend_config=svd_backend_config,
    )
    consolidator.initialize(ctx)
    t_comp0 = time.perf_counter()
    comp_result = _run_with_linalg_retry(
        lambda: consolidator.compress_model_with_surgery(model, method_overrides=fact_cfg.get("overrides"))
    )
    t_comp1 = time.perf_counter()

    mp2 = MemoryProfiler()
    inputs2 = random_ids_inputs(model, bs, seqlen, device)
    t_comp_inf0 = time.perf_counter()
    with mp2.profile_execution("compressed_inference"):
        with torch.no_grad():
            _ = model(**inputs2)
    t_comp_inf1 = time.perf_counter()
    compressed_report = mp_to_dict(mp2)

    compressed_params = params_count(model)
    ratio = float(compressed_params) / float(original_params) if original_params > 0 else 1.0
    mode_label = "with CoLA" if use_cola else "without CoLA"
    print(f"\nWorkspace: {workspace}")
    print(f"Saving compressed model to: {save_dir}")
    print(f"Using backend: {svd_backend} (CoLA={use_cola})")
    print(f"\nOriginal params: {original_params:,}")
    print(f"Compressed params: {compressed_params:,}")
    print(f"Compression ratio (compressed/original): {ratio:.4f}")
    print(f"\n== [{mode_label}] Memory (baseline) ==\n", json.dumps(baseline_report, indent=2))
    print(f"\n== [{mode_label}] Memory (compressed) ==\n", json.dumps(compressed_report, indent=2))

    layers_replaced = comp_result.parameters.get("layers_replaced", [])
    module_types = {}
    func_names = {}
    factor_sizes = {}
    for path in layers_replaced:
        stats = comp_result.parameters.get("surgery_stats", {}).get(path, {})
        if stats.get("layer_type"):
            module_types[path] = stats["layer_type"]
        mod = consolidator._get_module_by_name(model, path)
        func_name = getattr(mod, "func_name", None) or getattr(getattr(mod, "weight", None), "func_name", None)
        if isinstance(func_name, str):
            func_names[path] = func_name
        if isinstance(mod, FactorLinear):
            sizes = []
            for factor in getattr(getattr(mod, "weight", None), "factors", []) or []:
                if hasattr(factor, "weight") and hasattr(factor.weight, "shape"):
                    sizes.append(list(factor.weight.shape))
            if sizes:
                factor_sizes[path] = sizes

    manifest = {
        "type": "qwen3_svd_cola" if use_cola else "qwen3_svd",
        "base_model": model_name,
        "modules_replaced": layers_replaced,
        "save_format": "safetensors_state_dict",
    }
    if module_types:
        manifest["module_types"] = module_types
    if func_names:
        manifest["func_names"] = func_names
    if factor_sizes:
        manifest["factor_sizes"] = factor_sizes

    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\nSaved compressed model to: {save_dir}")

    total_time = time.perf_counter() - t0
    timings = {
        "total": total_time,
        "baseline_inference": t_base1 - t_base0,
        "compression": t_comp1 - t_comp0,
        "compressed_inference": t_comp_inf1 - t_comp_inf0,
    }
    print("\n== Timings (s) ==")
    for k, v in timings.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
