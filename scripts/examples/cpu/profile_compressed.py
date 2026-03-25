#!/usr/bin/env python3
"""
Example: Load a saved compressed model (SVD) and profile with optional CoLA/non-CoLA selection.

Usage:
  HF_TOKEN=... python scripts/examples/profile_compressed.py [--cola yes|no] [--config path] [--workspace path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Ensure project root is on sys.path to import modules when running directly
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_project_root_to_sys_path() -> None:
    """Include repo root on sys.path so goldcrest imports work when run directly."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def import_ALL_PARALLEL_STYLES() -> None:
    """Backfill transformers modeling utils parallel styles if missing."""
    import transformers.modeling_utils as _mu

    if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
        _mu.ALL_PARALLEL_STYLES = {"colwise", "rowwise", "sequence", "context"}


def random_ids_inputs(model, batch_size: int, seq_len: int, device: str):
    """Generate random token ids and attention mask for causal LM profiling."""
    import torch

    vocab = getattr(getattr(model, "config", None), "vocab_size", 32000)
    input_ids = torch.randint(low=0, high=int(vocab), size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mp_to_dict(mp):
    """Convert MemoryProfiler profiles to a plain dict."""
    return {name: profile.to_dict() for name, profile in (mp.profiles or {}).items()}


add_project_root_to_sys_path()

import torch
from transformers import AutoTokenizer

from goldcrest.framework.compressed_io import load_compressed_from_safetensors, Manifest
from goldcrest.framework.memory_profiler import MemoryProfiler


def _configure_linalg_backend():
    """Allow overriding CUDA linalg backend via env for debugging."""
    preferred = os.environ.get("TORCH_LINALG_PREFERRED")
    if preferred:
        torch.backends.cuda.preferred_linalg_library(preferred)
        print(f"[linalg] TORCH_LINALG_PREFERRED={preferred} -> set preferred backend")
    else:
        current = torch.backends.cuda.preferred_linalg_library()
        print(f"[linalg] using default backend: {current}")
    return torch.backends.cuda.preferred_linalg_library()


def _resolve_device(preferred: str) -> str:
    if not preferred or preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] Requested CUDA device '{preferred}' but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return preferred


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
    default_name = "examples/gemma3_svd_cola" if use_cola else "examples/gemma3_svd"
    workspace_name = names.get("cola" if use_cola else "non_cola", default_name)
    return (log_root_path / workspace_name).resolve()


def _select_save_dir(cfg: dict, workspace: Path, compressed_subdir: str) -> Path:
    save_cfg = cfg.get("save", {}) or {}
    compressed_dir = save_cfg.get("compressed_dir")
    if compressed_dir:
        return _resolve_path(compressed_dir)
    return (workspace / compressed_subdir).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a compressed SVD model (CoLA or non-CoLA).")
    parser.add_argument("--config", default="config/profiles/profile/svd_gemma3.yaml", help="Path to configuration YAML.")
    parser.add_argument(
        "--cola",
        choices=["yes", "no"],
        default=None,
        help="Load CoLA-compressed model (yes/no). CLI overrides config; default is yes.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory for locating the compressed model.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    runtime_cfg = cfg.get("runtime", {}) or {}
    t_total0 = time.perf_counter()

    use_cola = _parse_yes_no(args.cola)
    if use_cola is None:
        use_cola = _parse_yes_no(runtime_cfg.get("use_cola_default"))
    if use_cola is None:
        use_cola = True

    import_ALL_PARALLEL_STYLES()

    workspace = _select_workspace(runtime_cfg, use_cola, args.workspace)
    compressed_subdir = runtime_cfg.get("compressed_subdir", "compressed")
    save_dir = _select_save_dir(cfg, workspace, compressed_subdir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Compressed model directory not found: {save_dir}")

    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        base_model = manifest.base_model or cfg["model"]["name"]
    else:
        base_model = cfg["model"]["name"]

    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=token)

    evaluation = (cfg.get("evaluation", {}) or {})
    eval_device = _resolve_device(str(evaluation.get("device", "cpu")))

    _configure_linalg_backend()

    model = load_compressed_from_safetensors(base_model, save_dir, device=eval_device, debug=True)
    model.eval()

    mp = MemoryProfiler()
    ids_cfg = (cfg.get("memory_inference", {}) or {}).get("input", {})
    bs = int(ids_cfg.get("batch_size", 1))
    seqlen = int(ids_cfg.get("seq_len", 128))
    print(f"input_batch_size: {bs}")
    print(f"input_seq_len: {seqlen}")
    inputs = random_ids_inputs(model, bs, seqlen, eval_device)
    t_inf0 = time.perf_counter()
    with mp.profile_execution("compressed_inference"):
        with torch.no_grad():
            _ = model(**inputs)
    t_inf1 = time.perf_counter()
    mode_label = "with CoLA" if use_cola else "without CoLA"
    print(f"\n== [{mode_label}] Memory (compressed, initially loaded) ==\n", json.dumps(mp_to_dict(mp), indent=2))

    inference_time = t_inf1 - t_inf0
    timings = {
        "total": time.perf_counter() - t_total0,
        "inference": inference_time,
    }
    print("\n== Timings (s) ==")
    for k, v in timings.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
