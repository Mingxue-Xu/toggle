#!/usr/bin/env python3
"""
Strategy 6: Sparsity-Aware Tensor-Train Decomposition for H100 80GB.

Uses sparsity metric from weight analysis to identify layers suitable for
Tensor-Train decomposition.

Usage:
  HF_TOKEN=... python scripts/examples/h100_tensor_train.py [--config path] [--model name]
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
from src.plugins.analysis.weight_metrics import WeightMetricsPlugin, WeightMetricsAnalyzer
from src.plugins.analysis.metric_utils import BasicMetricsBackend
from src.plugins.compression.consolidator import ModelConsolidator


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tensor-Train Decomposition for H100")
    parser.add_argument("--config", default="config/h100_tensor_train.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--ranks", default=None, help="Override TT ranks (comma-separated)")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_tensor_train"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_tensor_train] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_tensor_train] Original params: {original_params:,}")

    # Step 1: Weight Analysis for sparsity measurement
    print("[h100_tensor_train] Running weight analysis for sparsity measurement...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Use WeightMetricsAnalyzer directly for detailed control
    backend = BasicMetricsBackend()
    analyzer = WeightMetricsAnalyzer(backend)

    t0 = time.perf_counter()
    results = analyzer.analyze_model(
        model=model,
        metrics_to_use=["sparsity", "l2_norm", "std"],
        selection={"include_names": ["mlp", "self_attn"], "parameter_kinds": ["weight"]},
        compute={"move_to_cpu": True},
    )
    t_analysis = time.perf_counter() - t0
    print(f"[h100_tensor_train] Weight analysis completed in {t_analysis:.2f}s")

    # Identify sparse layers suitable for TT decomposition
    layer_sparsity = {}
    sparse_layers = []
    sparsity_threshold = cfg.get("sparsity_threshold", 0.1)

    for layer in results:
        name = layer.get("name", "")
        values = layer.get("values", {})
        sparsity = values.get("sparsity", 0)
        layer_sparsity[name] = sparsity
        if sparsity > sparsity_threshold:
            sparse_layers.append(name)

    print(f"[h100_tensor_train] Found {len(sparse_layers)} layers with sparsity > {sparsity_threshold}")

    # Step 2: Apply Tensor-Train Decomposition
    print("[h100_tensor_train] Applying Tensor-Train decomposition...")
    fact_cfg = cfg.get("factorization", {})
    objects = fact_cfg.get("objects", [
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ])

    # Determine TT ranks
    if args.ranks:
        tensor_ranks = [int(r) for r in args.ranks.split(",")]
    else:
        tensor_ranks = fact_cfg.get("tensor_ranks", [1, 32, 32, 1])

    consolidator = ModelConsolidator(
        compression_method="tensor_train",
        target_modules=objects,
        tensor_ranks=tensor_ranks,
    )
    consolidator.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    print(f"\n[h100_tensor_train] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  TT ranks:          {tensor_ranks}")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_tensor_train",
        "base_model": model_name,
        "strategy": "tensor_train_decomposition",
        "compression_ratio": ratio,
        "tensor_ranks": tensor_ranks,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_tensor_train] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "tensor_train_report.json"
    report = {
        "model": model_name,
        "strategy": "tensor_train_decomposition",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "tensor_ranks": tensor_ranks,
        "layer_sparsity": layer_sparsity,
        "sparse_layers": sparse_layers,
        "sparsity_threshold": sparsity_threshold,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_tensor_train] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
