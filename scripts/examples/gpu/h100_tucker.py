#!/usr/bin/env python3
"""
Strategy 3: Tucker Decomposition with Rank Estimation for H100 80GB.

Uses tucker_rank_est from activation analysis to determine optimal Tucker ranks
per layer.

Usage:
  HF_TOKEN=... python scripts/examples/h100_tucker.py [--config path] [--model name]
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

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.plugins.analysis.activation_metrics import ActivationMetricsPlugin
from goldcrest.plugins.compression.consolidator import ModelConsolidator


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tucker Decomposition for H100")
    parser.add_argument("--config", default="config/h100_tucker.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--ranks", default=None, help="Override tucker ranks (comma-separated)")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_tucker"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_tucker] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_tucker] Original params: {original_params:,}")

    # Step 1: Activation Analysis for Tucker rank estimation
    print("[h100_tucker] Running activation analysis for rank estimation...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_tucker] Analysis completed in {t_analysis:.2f}s")

    # Extract tucker_rank_est from analysis
    layer_tucker_ranks = {}
    for layer in act_report.get("per_layer", []):
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        tucker_est = stats.get("tucker_rank_est.median", None)
        if tucker_est:
            # Parse tuple string like "(32, 64)"
            try:
                if isinstance(tucker_est, str):
                    ranks = eval(tucker_est)
                else:
                    ranks = tucker_est
                layer_tucker_ranks[name] = list(ranks)
            except Exception:
                pass

    print(f"[h100_tucker] Estimated Tucker ranks for {len(layer_tucker_ranks)} layers")

    # Step 2: Apply Tucker Decomposition
    print("[h100_tucker] Applying Tucker decomposition...")
    fact_cfg = cfg.get("factorization", {})
    objects = fact_cfg.get("objects", [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].mlp.gate_proj",
    ])

    # Determine tucker ranks
    if args.ranks:
        tucker_ranks = [int(r) for r in args.ranks.split(",")]
    else:
        tucker_ranks = fact_cfg.get("tucker_ranks", [64, 64])

    consolidator = ModelConsolidator(
        compression_method="tucker",
        target_modules=objects,
        tucker_ranks=tucker_ranks,
    )
    consolidator.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    print(f"\n[h100_tucker] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  Tucker ranks:      {tucker_ranks}")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_tucker",
        "base_model": model_name,
        "strategy": "tucker_decomposition",
        "compression_ratio": ratio,
        "tucker_ranks": tucker_ranks,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_tucker] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "tucker_report.json"
    report = {
        "model": model_name,
        "strategy": "tucker_decomposition",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "tucker_ranks": tucker_ranks,
        "estimated_ranks": layer_tucker_ranks,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_tucker] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
