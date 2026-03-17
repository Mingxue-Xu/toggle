#!/usr/bin/env python3
"""
Strategy: Entropy-Based SVD Rank Selection for H200 GPU (140GB).

Uses WeightMetricsAnalyzer to compute entropy-based metrics for optimal
SVD rank selection per layer. Optimized for H200 GPU with BF16 precision.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_entropy_svd.py [options]
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
from src.plugins.analysis.weight_metrics import WeightMetricsAnalyzer
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
    parser = argparse.ArgumentParser(description="H200 GPU Entropy-Based SVD")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--ratio", type=float, default=0.5, help="Target compression ratio")
    parser.add_argument("--workspace", default="logs/gpu_h200_entropy_svd", help="Workspace directory")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU entropy SVD requires CUDA.")

    model_name = RECOMMENDED_MODELS[args.preset] if args.preset else args.model
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[h200_entropy_svd] H200 GPU Entropy-Based SVD")
    print(f"[h200_entropy_svd] Model: {model_name}")
    print(f"[h200_entropy_svd] Target ratio: {args.ratio}")

    print(f"[h200_entropy_svd] Loading model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()

    original_params = params_count(model)
    print(f"[h200_entropy_svd] Original params: {original_params:,}")
    print(f"[h200_entropy_svd] GPU Memory: {get_gpu_memory_gb():.2f} GB")

    ctx = PipelineContext(config={}, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Step 1: Weight Metrics Analysis
    print("[h200_entropy_svd] Analyzing weight metrics...")
    from src.plugins.analysis.metric_utils import BasicMetricsBackend
    analyzer = WeightMetricsAnalyzer(backend=BasicMetricsBackend())
    t0 = time.perf_counter()
    per_layer = analyzer.analyze_model(
        model=model,
        metrics_to_use=["entropy", "stable_rank"],
        selection={"module_types": {"Linear"}, "parameter_kinds": {"weight"}},
        compute={"move_to_cpu": True},
    )
    analysis_time = time.perf_counter() - t0
    print(f"[h200_entropy_svd] Analysis time: {analysis_time:.2f}s")

    # Step 2: Determine per-layer ranks based on entropy
    layer_ranks = {}
    min_rank, max_rank = 32, 256

    for layer in per_layer:
        name = layer.get("name", "")
        entropy = layer.get("values", {}).get("entropy", 0.5)
        # Higher entropy = more information = higher rank
        rank = int(min_rank + entropy * (max_rank - min_rank))
        layer_ranks[name] = rank

    # Step 3: Apply SVD Compression using per-layer entropy-based ranks
    print("[h200_entropy_svd] Applying SVD compression with entropy-based ranks...")
    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    # Build per-module overrides from entropy-based layer_ranks.
    # layer_ranks maps full parameter names (e.g. "model.layers.0.self_attn.q_proj.weight")
    # to ranks. We convert to module paths (without ".weight") for the consolidator.
    method_overrides = []
    for param_name, rank_val in layer_ranks.items():
        # Strip trailing ".weight" or ".bias" to get the module path
        module_path = param_name
        if module_path.endswith(".weight"):
            module_path = module_path[:-len(".weight")]
        elif module_path.endswith(".bias"):
            module_path = module_path[:-len(".bias")]
        method_overrides.append({
            "pattern": module_path,
            "method": "svd",
            "rank": rank_val,
        })

    # Fallback global rank for any layers not in overrides
    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    fallback_rank = max(1, int(args.ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    fallback_rank = min(fallback_rank, min(sample_m, sample_n))

    print(f"[h200_entropy_svd] Entropy rank range: {min(layer_ranks.values())}-{max(layer_ranks.values())}, "
          f"fallback rank: {fallback_rank}, overrides: {len(method_overrides)}")

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=target_modules,
        rank=fallback_rank,
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
        print(f"\n[h200_entropy_svd] WARNING: Compression ratio {ratio:.4f} indicates no effective compression!")
        print(f"  Layers replaced: {len(layers_replaced)}")
        print(f"  This likely means model surgery failed for all layers.")

    print(f"\n[h200_entropy_svd] Results:")
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
        "type": "h200_entropy_svd",
        "base_model": model_name,
        "compression_ratio": ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"[h200_entropy_svd] Saved to: {save_dir}")

    report = {
        "model": model_name,
        "strategy": "h200_entropy_svd",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "analysis_time_sec": analysis_time,
        "compression_time_sec": compress_time,
        "layer_ranks": layer_ranks,
        "gpu_memory_gb": get_gpu_memory_gb(),
    }
    (workspace / "entropy_svd_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
