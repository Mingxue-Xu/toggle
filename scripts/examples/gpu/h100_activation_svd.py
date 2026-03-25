#!/usr/bin/env python3
"""
Strategy 1: Activation-Guided SVD Compression for H100 80GB.

Uses ActivationDrivenSVDWeightsCompressionPlugin which internally:
  1) Runs ActivationMetricsPlugin for activation analysis
  2) Uses LayerSVDRankDecider to compute per-layer ranks
  3) Applies ModelConsolidator for SVD compression with model surgery

This script demonstrates proper plugin-based architecture following the
modular design documented in CPU_PLAN.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h100_activation_svd.py [--config path] [--model name]
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
from goldcrest.plugins.compression.svd_activation import ActivationDrivenSVDWeightsCompressionPlugin


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-Guided SVD Compression for H100")
    parser.add_argument("--config", default="config/h100_activation_svd.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_activation_svd"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_activation_svd] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_activation_svd] Original params: {original_params:,}")

    # Create pipeline context with state management
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    # Step 1: Activation Analysis using ActivationMetricsPlugin
    # This stores the report in state at "analysis.activation_metrics"
    print("[h100_activation_svd] Running activation analysis...")
    act_plugin = ActivationMetricsPlugin(**cfg)
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_activation_svd] Analysis completed in {t_analysis:.2f}s")

    # Step 2: Apply SVD Compression using ActivationDrivenSVDWeightsCompressionPlugin
    # This plugin internally uses LayerSVDRankDecider for rank calculation
    # and ModelConsolidator for the actual compression
    print("[h100_activation_svd] Applying SVD compression via ActivationDrivenSVDWeightsCompressionPlugin...")
    svd_plugin = ActivationDrivenSVDWeightsCompressionPlugin(**cfg)
    svd_plugin.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = svd_plugin.do_execute(model=model, report=act_report)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    # Extract layer ranks from the decider result
    layer_ranks = comp_result.get("decider", {}).get("layer_ranks", {})

    print(f"\n[h100_activation_svd] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")
    print(f"  Layers compressed: {len(comp_result.get('compression', {}).get('layers_replaced', []))}")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_activation_svd",
        "base_model": model_name,
        "strategy": "activation_guided_svd",
        "compression_ratio": ratio,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_activation_svd] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "activation_svd_report.json"
    report = {
        "model": model_name,
        "strategy": "activation_guided_svd",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
        "layer_ranks": layer_ranks,
        "decider_result": comp_result.get("decider", {}),
        "compression_result": comp_result.get("compression", {}),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[h100_activation_svd] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
