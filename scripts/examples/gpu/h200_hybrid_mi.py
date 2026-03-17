#!/usr/bin/env python3
"""
Strategy: Mutual Information Hybrid Analysis for H200 GPU (140GB).

Uses ActivationMetricsPlugin with mutual information metrics for
hybrid compression strategy selection. Optimized for H200 GPU with BF16 precision.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_hybrid_mi.py [options]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.analysis.activation_metrics import ActivationMetricsPlugin
from src.plugins.compression.consolidator import ModelConsolidator


RECOMMENDED_MODELS = {
    "quick": "meta-llama/Llama-3.1-8B",
    "mid": "Qwen/Qwen2.5-14B",
    "large": "google/gemma-2-27b",
    "flagship": "meta-llama/Llama-3.1-70B",
}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Hybrid MI Analysis")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--ratio", type=float, default=0.5, help="Target compression ratio")
    parser.add_argument("--workspace", default="logs/gpu_h200_hybrid_mi", help="Workspace directory")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU hybrid MI requires CUDA.")

    model_name = RECOMMENDED_MODELS[args.preset] if args.preset else args.model
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[h200_hybrid_mi] H200 GPU Hybrid Mutual Information Analysis")
    print(f"[h200_hybrid_mi] Model: {model_name}")
    print(f"[h200_hybrid_mi] Target ratio: {args.ratio}")

    print(f"[h200_hybrid_mi] Loading model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()

    original_params = params_count(model)
    print(f"[h200_hybrid_mi] Original params: {original_params:,}")
    print(f"[h200_hybrid_mi] GPU Memory: {get_gpu_memory_gb():.2f} GB")

    ctx = PipelineContext(config={}, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Step 1: Activation Metrics Analysis (including MI)
    print("[h200_hybrid_mi] Analyzing activation metrics...")
    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    analysis_time = time.perf_counter() - t0
    print(f"[h200_hybrid_mi] Analysis time: {analysis_time:.2f}s")

    # Step 2: Select compression method per layer based on MI
    layer_methods = {}
    per_layer = act_report.get("per_layer", [])

    for layer in per_layer:
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        mi = stats.get("mutual_information", 0.5)
        spectral = stats.get("spectral_norm.median", 1.0)

        # Hybrid strategy: high MI = SVD, low MI = Tucker
        if mi > 0.7:
            layer_methods[name] = "svd"
        elif mi > 0.4:
            layer_methods[name] = "tucker"
        else:
            layer_methods[name] = "cp"

    # Step 3: Apply Hybrid Compression using per-layer method selection
    print("[h200_hybrid_mi] Applying hybrid compression with MI-based method selection...")

    proj_suffixes = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    rank = max(1, int(args.ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    rank = min(rank, min(sample_m, sample_n))
    tucker_ranks = [min(rank, sample_m), min(rank, sample_n)]
    cp_rank = min(rank, min(sample_m, sample_n))

    # Build per-module overrides from MI-based layer_methods.
    # layer_methods maps block names (e.g. "model.layers.0") to methods ("svd"/"tucker"/"cp").
    # Expand to each projection within the block.
    method_overrides = []
    for block_name, method in layer_methods.items():
        for suffix in proj_suffixes:
            override = {"pattern": f"{block_name}.{suffix}", "method": method}
            if method == "svd":
                override["rank"] = rank
            elif method == "tucker":
                override["tucker_ranks"] = tucker_ranks
            elif method == "cp":
                override["cp_rank"] = cp_rank
            method_overrides.append(override)

    method_counts = {}
    for m in layer_methods.values():
        method_counts[m] = method_counts.get(m, 0) + 1
    print(f"[h200_hybrid_mi] Method selection: {method_counts}")
    print(f"[h200_hybrid_mi] Total overrides: {len(method_overrides)}")

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=target_modules,
        rank=rank,
        svd_backend="torch",
        method_overrides=method_overrides,
    )
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    result = consolidator.compress_model_with_surgery(model)
    compress_time = time.perf_counter() - t0

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    if ratio >= 0.99:
        layers_replaced = result.parameters.get("layers_replaced", [])
        print(f"\n[h200_hybrid_mi] WARNING: Compression ratio {ratio:.4f} indicates no effective compression!")
        print(f"  Layers replaced: {len(layers_replaced)}")
        print(f"  This likely means model surgery failed for all layers.")

    print(f"\n[h200_hybrid_mi] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f}")
    print(f"  Analysis time:     {analysis_time:.2f}s")
    print(f"  Compression time:  {compress_time:.2f}s")
    print(f"  GPU Memory:        {get_gpu_memory_gb():.2f} GB")

    # Save
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h200_hybrid_mi",
        "base_model": model_name,
        "compression_ratio": ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"[h200_hybrid_mi] Saved to: {save_dir}")

    report = {
        "model": model_name,
        "strategy": "h200_hybrid_mi",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "analysis_time_sec": analysis_time,
        "compression_time_sec": compress_time,
        "layer_methods": layer_methods,
        "gpu_memory_gb": get_gpu_memory_gb(),
    }
    (workspace / "hybrid_mi_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
