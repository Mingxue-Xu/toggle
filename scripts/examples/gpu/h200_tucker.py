#!/usr/bin/env python3
"""
Strategy: Tucker Decomposition for H200 GPU (140GB).

Applies Tucker decomposition using TensorLy for large-scale LLM compression.
Optimized for H200 GPU with BF16 precision.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_tucker.py [options]
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

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.plugins.compression.consolidator import ModelConsolidator


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
    parser = argparse.ArgumentParser(description="H200 GPU Tucker Decomposition")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--ratio", type=float, default=0.5, help="Target compression ratio")
    parser.add_argument("--workspace", default="logs/gpu_h200_tucker", help="Workspace directory")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU Tucker requires CUDA.")

    model_name = RECOMMENDED_MODELS[args.preset] if args.preset else args.model
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[h200_tucker] H200 GPU Tucker Decomposition")
    print(f"[h200_tucker] Model: {model_name}")
    print(f"[h200_tucker] Target ratio: {args.ratio}")

    print(f"[h200_tucker] Loading model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()

    original_params = params_count(model)
    print(f"[h200_tucker] Original params: {original_params:,}")
    print(f"[h200_tucker] GPU Memory: {get_gpu_memory_gb():.2f} GB")

    ctx = PipelineContext(config={}, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Estimate Tucker ranks
    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    rank = max(1, int(args.ratio * min(sample_m, sample_n) * 0.7))

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].mlp.gate_proj",
    ]

    print(f"[h200_tucker] Applying Tucker decomposition (rank={rank})...")
    consolidator = ModelConsolidator(
        compression_method="tucker",
        target_modules=target_modules,
        tucker_ranks=[min(rank, sample_m), min(rank, sample_n)],
    )
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    result = consolidator.compress_model_with_surgery(model)
    compress_time = time.perf_counter() - t0

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    if ratio >= 0.99:
        layers_replaced = result.parameters.get("layers_replaced", [])
        print(f"\n[h200_tucker] WARNING: Compression ratio {ratio:.4f} indicates no effective compression!")
        print(f"  Layers replaced: {len(layers_replaced)}/{len(target_modules)}")
        print(f"  This likely means model surgery failed for all layers.")

    print(f"\n[h200_tucker] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f}")
    print(f"  Compression time:  {compress_time:.2f}s")
    print(f"  GPU Memory:        {get_gpu_memory_gb():.2f} GB")

    # Save
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h200_tucker",
        "base_model": model_name,
        "compression_ratio": ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"[h200_tucker] Saved to: {save_dir}")

    report = {
        "model": model_name,
        "strategy": "h200_tucker",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "compression_time_sec": compress_time,
        "gpu_memory_gb": get_gpu_memory_gb(),
    }
    (workspace / "tucker_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
