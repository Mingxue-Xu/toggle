#!/usr/bin/env python3
"""
Strategy 4: CP Decomposition with Rank Estimation for H100 80GB.

Uses cp_rank_est metric from activation analysis to determine optimal CP rank
for each tensor.

Usage:
  HF_TOKEN=... python scripts/examples/h100_cp.py [--config path] [--model name]
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
from src.plugins.compression.consolidator import ModelConsolidator


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CP Decomposition for H100")
    parser.add_argument("--config", default="config/h100_cp.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--rank", type=int, default=None, help="Override CP rank")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_cp"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_cp] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_cp] Original params: {original_params:,}")

    # Step 1: Activation Analysis for CP rank estimation
    print("[h100_cp] Running activation analysis for rank estimation...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_cp] Analysis completed in {t_analysis:.2f}s")

    # Extract cp_rank_est from analysis
    layer_cp_ranks = {}
    cp_rank_values = []
    for layer in act_report.get("per_layer", []):
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        cp_est = stats.get("cp_rank_est.median", None)
        if cp_est is not None:
            try:
                rank = int(float(cp_est))
                layer_cp_ranks[name] = rank
                cp_rank_values.append(rank)
            except Exception:
                pass

    # Use median of estimated ranks or default
    if cp_rank_values:
        estimated_rank = int(sorted(cp_rank_values)[len(cp_rank_values) // 2])
        print(f"[h100_cp] Estimated CP rank (median): {estimated_rank}")
    else:
        estimated_rank = 64

    # Step 2: Apply CP Decomposition
    print("[h100_cp] Applying CP decomposition...")
    fact_cfg = cfg.get("factorization", {})
    objects = fact_cfg.get("objects", [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.v_proj",
    ])

    cp_rank = args.rank or fact_cfg.get("cp_rank", estimated_rank)

    consolidator = ModelConsolidator(
        compression_method="cp",
        target_modules=objects,
        cp_rank=cp_rank,
    )
    consolidator.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    print(f"\n[h100_cp] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  CP rank:           {cp_rank}")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_cp",
        "base_model": model_name,
        "strategy": "cp_decomposition",
        "compression_ratio": ratio,
        "cp_rank": cp_rank,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_cp] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "cp_report.json"
    report = {
        "model": model_name,
        "strategy": "cp_decomposition",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "cp_rank": cp_rank,
        "estimated_ranks": layer_cp_ranks,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_cp] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
