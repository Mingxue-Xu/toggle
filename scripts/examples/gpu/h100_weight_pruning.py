#!/usr/bin/env python3
"""
Strategy 2: Weight-Metrics-Guided Block Pruning for H100 80GB.

Uses WeightMetricsPlugin to compute l2_norm per layer, then prunes blocks
with lowest importance scores.

Usage:
  HF_TOKEN=... python scripts/examples/h100_weight_pruning.py [--config path] [--model name]
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
from src.plugins.analysis.weight_metrics import WeightMetricsPlugin
from src.plugins.compression.pruning import PruningPlugin
from src.plugins.compression.consolidator import ModelConsolidator


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weight-Guided Block Pruning for H100")
    parser.add_argument("--config", default="config/h100_weight_pruning.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--ratio", type=float, default=None, help="Override pruning ratio")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_weight_pruning"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_weight_pruning] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_weight_pruning] Original params: {original_params:,}")

    # Step 1: Weight Analysis
    print("[h100_weight_pruning] Running weight analysis...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    weight_plugin = WeightMetricsPlugin()
    weight_plugin.initialize(ctx)
    t0 = time.perf_counter()
    weight_report = weight_plugin.do_execute(model=model)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_weight_pruning] Weight analysis completed in {t_analysis:.2f}s")

    # Extract layer scores
    layer_scores = {}
    for layer in weight_report.get("per_layer", []):
        name = layer.get("name", "")
        values = layer.get("values", {})
        l2 = values.get("l2_norm", 0)
        layer_scores[name] = l2

    print(f"[h100_weight_pruning] Analyzed {len(layer_scores)} layers")

    # Step 2: Block Pruning
    print("[h100_weight_pruning] Applying block pruning...")
    pruning_cfg = cfg.get("pruning", {})
    ratio = args.ratio or pruning_cfg.get("ratio_deleted_block", 0.2)

    pruning_plugin = PruningPlugin()
    pruning_plugin.initialize(ctx)
    pruning_plugin._configuration = {
        "pruning": {
            "selection_metric_type": pruning_cfg.get("selection_metric_type", "l2_norm.median"),
            "ratio_deleted_block": ratio,
            "min_keep": pruning_cfg.get("min_keep", 4),
            "container_path": pruning_cfg.get("container_path", "model.layers"),
            "group_prefix": pruning_cfg.get("group_prefix", "layers"),
            "aggregator": pruning_cfg.get("aggregator", "mean"),
        }
    }

    t1 = time.perf_counter()
    pruning_result = pruning_plugin.do_execute(model=model)
    t_prune = time.perf_counter() - t1

    pruned_indices = pruning_result.get("selected_indices", [])
    pruning_stats = pruning_result.get("pruning_stats", {})

    compressed_params = params_count(model)
    compression_ratio = compressed_params / original_params

    print(f"\n[h100_weight_pruning] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {compression_ratio:.4f} ({(1-compression_ratio)*100:.1f}% reduction)")
    print(f"  Blocks pruned:     {len(pruned_indices)}")
    print(f"  Pruned indices:    {pruned_indices}")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Pruning time:      {t_prune:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_weight_pruning",
        "base_model": model_name,
        "strategy": "weight_guided_pruning",
        "compression_ratio": compression_ratio,
        "pruned_blocks": pruned_indices,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_weight_pruning] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "weight_pruning_report.json"
    report = {
        "model": model_name,
        "strategy": "weight_guided_pruning",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": compression_ratio,
        "pruned_blocks": pruned_indices,
        "layer_scores": layer_scores,
        "analysis_time_sec": t_analysis,
        "pruning_time_sec": t_prune,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_weight_pruning] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
