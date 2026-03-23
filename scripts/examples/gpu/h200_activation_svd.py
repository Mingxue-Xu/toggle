#!/usr/bin/env python3
"""
Strategy 1: Activation-Guided SVD Compression for H200 GPU (140GB).

Uses ActivationDrivenSVDWeightsCompressionPlugin which internally:
  1) Runs ActivationMetricsPlugin for activation analysis
  2) Uses LayerSVDRankDecider to compute per-layer ranks
  3) Applies ModelConsolidator for SVD compression with model surgery

Optimized for large-scale models (8B-70B parameters) on H200 GPU.
This script demonstrates proper plugin-based architecture following the
modular design documented in GPU_PLAN.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_activation_svd.py [--config path] [--model name]
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.analysis.activation_metrics import ActivationMetricsPlugin
from src.plugins.compression.svd_activation import ActivationDrivenSVDWeightsCompressionPlugin


# H200 GPU model recommendations
RECOMMENDED_MODELS = {
    "quick": "meta-llama/Llama-3.1-8B",
    "mid": "Qwen/Qwen2.5-14B",
    "large": "google/gemma-2-27b",
    "flagship": "meta-llama/Llama-3.1-70B",
}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Activation-Guided SVD Compression")
    parser.add_argument("--config", default=None, help="Optional config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None,
                        help="Use preset model configuration")
    return parser.parse_args()


def get_default_config() -> dict:
    """Get default configuration for H200 GPU."""
    return {
        "model": {
            "name": "meta-llama/Llama-3.1-8B",
        },
        "auth": {
            "hf_token_env": "HF_TOKEN",
        },
        "runtime": {
            "workspace": "logs/gpu_h200_activation_svd",
        },
        "svd_compression": {
            "min_rank": 32,
            "max_rank": 256,
            "target_param_ratio": 0.5,
        },
        "factorization": {
            "objects": [
                "model.layers[*].self_attn.q_proj",
                "model.layers[*].self_attn.k_proj",
                "model.layers[*].self_attn.v_proj",
                "model.layers[*].self_attn.o_proj",
                "model.layers[*].mlp.gate_proj",
                "model.layers[*].mlp.up_proj",
                "model.layers[*].mlp.down_proj",
            ],
            "rank": 128,
        },
        "compression": {
            "svd": {
                "backend": "cola",
                "cola": {"algorithm": "lanczos"},
            },
        },
    }


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU activation SVD requires CUDA. No GPU detected.")

    # Load config or use defaults
    if args.config:
        cfg_path = _resolve_path(args.config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
        else:
            print(f"[h200_activation_svd] Config not found, using defaults")
            cfg = get_default_config()
    else:
        cfg = get_default_config()

    # Handle model selection
    if args.preset:
        model_name = RECOMMENDED_MODELS[args.preset]
        print(f"Using preset '{args.preset}': {model_name}")
    elif args.model:
        model_name = args.model
    else:
        model_name = cfg["model"]["name"]

    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/gpu_h200_activation_svd"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h200_activation_svd] H200 GPU Activation-Guided SVD Compression")
    print(f"[h200_activation_svd] Model: {model_name}")
    print(f"[h200_activation_svd] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[h200_activation_svd] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"[h200_activation_svd] Loading model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=token,
        device_map="auto",
    )
    model.eval()

    original_params = params_count(model)
    print(f"[h200_activation_svd] Original params: {original_params:,}")
    print(f"[h200_activation_svd] GPU Memory after load: {get_gpu_memory_gb():.2f} GB")

    # Create pipeline context with state management
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Step 1: Activation Analysis using ActivationMetricsPlugin
    # This stores the report in state at "analysis.activation_metrics"
    print("[h200_activation_svd] Running activation analysis...")
    act_plugin = ActivationMetricsPlugin(**cfg)
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h200_activation_svd] Analysis completed in {t_analysis:.2f}s")
    print(f"[h200_activation_svd] GPU Memory after analysis: {get_gpu_memory_gb():.2f} GB")

    # Step 2: Apply SVD Compression using ActivationDrivenSVDWeightsCompressionPlugin
    # This plugin internally uses LayerSVDRankDecider for rank calculation
    # and ModelConsolidator for the actual compression
    print("[h200_activation_svd] Applying SVD compression via ActivationDrivenSVDWeightsCompressionPlugin...")
    svd_plugin = ActivationDrivenSVDWeightsCompressionPlugin(**cfg)
    svd_plugin.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = svd_plugin.do_execute(model=model, report=act_report)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    if ratio >= 0.99:
        print(f"\n[h200_activation_svd] WARNING: Compression ratio {ratio:.4f} indicates no effective compression!")

    # Extract layer ranks from the decider result
    layer_ranks = comp_result.get("decider", {}).get("layer_ranks", {})

    print(f"\n[h200_activation_svd] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")
    print(f"  GPU Memory:        {get_gpu_memory_gb():.2f} GB")
    print(f"  Layers compressed: {len(comp_result.get('compression', {}).get('layers_replaced', []))}")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h200_activation_svd",
        "base_model": model_name,
        "strategy": "activation_guided_svd",
        "compression_ratio": ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h200_activation_svd] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "activation_svd_report.json"
    report = {
        "model": model_name,
        "strategy": "h200_activation_guided_svd",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
        "layer_ranks": layer_ranks,
        "gpu_memory_gb": get_gpu_memory_gb(),
        "dtype": "bfloat16",
        "decider_result": comp_result.get("decider", {}),
        "compression_result": comp_result.get("compression", {}),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[h200_activation_svd] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
