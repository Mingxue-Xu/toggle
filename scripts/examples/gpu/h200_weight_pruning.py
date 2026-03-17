#!/usr/bin/env python3
"""
Strategy: Block-Level Weight Pruning for H200 GPU (140GB).

Applies structured weight pruning using LayerMetricSelector for large-scale LLMs.
Optimized for H200 GPU with BF16 precision.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_weight_pruning.py [options]
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
    parser = argparse.ArgumentParser(description="H200 GPU Weight Pruning")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity ratio")
    parser.add_argument("--workspace", default="logs/gpu_h200_weight_pruning", help="Workspace directory")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU weight pruning requires CUDA.")

    model_name = RECOMMENDED_MODELS[args.preset] if args.preset else args.model
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[h200_weight_pruning] H200 GPU Weight Pruning")
    print(f"[h200_weight_pruning] Model: {model_name}")
    print(f"[h200_weight_pruning] Target sparsity: {args.sparsity}")

    print(f"[h200_weight_pruning] Loading model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()

    original_params = params_count(model)
    print(f"[h200_weight_pruning] Original params: {original_params:,}")
    print(f"[h200_weight_pruning] GPU Memory: {get_gpu_memory_gb():.2f} GB")

    ctx = PipelineContext(config={}, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    print(f"[h200_weight_pruning] Applying weight pruning (sparsity={args.sparsity})...")

    # Magnitude pruning: zero out the smallest weights per target module
    import fnmatch
    import re

    def _matches_target(name, patterns):
        for pat in patterns:
            # Convert bracket glob to regex: model.layers[*].mlp.up_proj -> model\.layers\.\d+\.mlp\.up_proj
            regex = pat.replace(".", r"\.").replace("[*]", r"\.\d+")
            if re.search(regex, name):
                return True
        return False

    t0 = time.perf_counter()
    pruned_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not _matches_target(name, target_modules):
            continue
        w = module.weight.data
        w_abs = w.abs().float()
        flat = w_abs.reshape(-1)
        # torch.quantile() can't handle tensors >2^24 elements; sample instead
        if flat.numel() > 2**24:
            indices = torch.randperm(flat.numel(), device=flat.device)[:2**24]
            threshold = torch.quantile(flat[indices], args.sparsity)
        else:
            threshold = torch.quantile(flat, args.sparsity)
        mask = w.abs() >= threshold
        module.weight.data = w * mask
        pruned_layers += 1
    compress_time = time.perf_counter() - t0

    # Count non-zero params
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    effective_ratio = nonzero_params / original_params

    print(f"\n[h200_weight_pruning] Results:")
    print(f"  Original params:    {original_params:,}")
    print(f"  Non-zero params:    {nonzero_params:,}")
    print(f"  Effective ratio:    {effective_ratio:.4f}")
    print(f"  Compression time:   {compress_time:.2f}s")
    print(f"  GPU Memory:         {get_gpu_memory_gb():.2f} GB")

    # Save
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h200_weight_pruning",
        "base_model": model_name,
        "sparsity": args.sparsity,
        "effective_ratio": effective_ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"[h200_weight_pruning] Saved to: {save_dir}")

    report = {
        "model": model_name,
        "strategy": "h200_weight_pruning",
        "original_params": original_params,
        "nonzero_params": nonzero_params,
        "effective_ratio": effective_ratio,
        "compression_time_sec": compress_time,
        "gpu_memory_gb": get_gpu_memory_gb(),
    }
    (workspace / "weight_pruning_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
