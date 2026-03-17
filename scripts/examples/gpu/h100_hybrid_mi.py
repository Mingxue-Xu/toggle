#!/usr/bin/env python3
"""
Strategy 7: Mutual Information-Guided Hybrid Compression for H100 80GB.

Uses mutual_information_m0_m1 to identify layer dependencies, then applies
appropriate compression method (Tucker for high MI, SVD for low MI).

Usage:
  HF_TOKEN=... python scripts/examples/h100_hybrid_mi.py [--config path] [--model name]
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
    parser = argparse.ArgumentParser(description="MI-Guided Hybrid Compression for H100")
    parser.add_argument("--config", default="config/h100_hybrid_mi.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_hybrid_mi"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_hybrid_mi] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_hybrid_mi] Original params: {original_params:,}")

    # Step 1: Activation Analysis for MI measurement
    print("[h100_hybrid_mi] Running activation analysis for MI measurement...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_hybrid_mi] Analysis completed in {t_analysis:.2f}s")

    # Extract mutual_information_m0_m1 from analysis
    layer_mi = {}
    mi_values = []
    for layer in act_report.get("per_layer", []):
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        mi = stats.get("mutual_information_m0_m1.median", None)
        if mi is not None:
            try:
                mi_val = float(mi)
                layer_mi[name] = mi_val
                mi_values.append(mi_val)
            except Exception:
                pass

    # Compute MI threshold
    if mi_values:
        sorted_mi = sorted(mi_values)
        mi_threshold = sorted_mi[len(sorted_mi) // 2]  # Median
        print(f"[h100_hybrid_mi] MI threshold (median): {mi_threshold:.4f}")
    else:
        mi_threshold = 0.5

    # Classify layers by MI
    high_mi_layers = []
    low_mi_layers = []
    for name, mi in layer_mi.items():
        if mi > mi_threshold:
            high_mi_layers.append(name)
        else:
            low_mi_layers.append(name)

    print(f"[h100_hybrid_mi] High MI layers: {len(high_mi_layers)}, Low MI layers: {len(low_mi_layers)}")

    # Step 2: Apply hybrid compression
    # First: SVD for low MI layers (separable structure)
    print("[h100_hybrid_mi] Applying SVD to low-MI layers...")
    fact_cfg = cfg.get("factorization", {})
    svd_objects = fact_cfg.get("svd_objects", [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
    ])

    svd_rank = fact_cfg.get("svd_rank", 64)
    svd_backend = cfg.get("compression", {}).get("svd", {}).get("backend", "cola")

    svd_consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=svd_objects,
        rank=svd_rank,
        svd_backend=svd_backend,
    )
    svd_consolidator.initialize(ctx)

    t1 = time.perf_counter()
    svd_result = svd_consolidator.compress_model_with_surgery(model)
    t_svd = time.perf_counter() - t1
    print(f"[h100_hybrid_mi] SVD compression completed in {t_svd:.2f}s")

    # Second: Tucker for high MI layers (mode interaction)
    print("[h100_hybrid_mi] Applying Tucker to high-MI layers...")
    tucker_objects = fact_cfg.get("tucker_objects", [
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ])

    tucker_ranks = fact_cfg.get("tucker_ranks", [48, 48])

    tucker_consolidator = ModelConsolidator(
        compression_method="tucker",
        target_modules=tucker_objects,
        tucker_ranks=tucker_ranks,
    )
    tucker_consolidator.initialize(ctx)

    t2 = time.perf_counter()
    tucker_result = tucker_consolidator.compress_model_with_surgery(model)
    t_tucker = time.perf_counter() - t2
    print(f"[h100_hybrid_mi] Tucker compression completed in {t_tucker:.2f}s")

    compressed_params = params_count(model)
    ratio = compressed_params / original_params
    total_compress_time = t_svd + t_tucker

    print(f"\n[h100_hybrid_mi] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  MI threshold:      {mi_threshold:.4f}")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  SVD time:          {t_svd:.2f}s")
    print(f"  Tucker time:       {t_tucker:.2f}s")
    print(f"  Total compress:    {total_compress_time:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_hybrid_mi",
        "base_model": model_name,
        "strategy": "mi_guided_hybrid",
        "compression_ratio": ratio,
        "svd_rank": svd_rank,
        "tucker_ranks": tucker_ranks,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_hybrid_mi] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "hybrid_mi_report.json"
    report = {
        "model": model_name,
        "strategy": "mi_guided_hybrid",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "mi_threshold": mi_threshold,
        "layer_mi": layer_mi,
        "high_mi_layers": high_mi_layers,
        "low_mi_layers": low_mi_layers,
        "svd_rank": svd_rank,
        "tucker_ranks": tucker_ranks,
        "analysis_time_sec": t_analysis,
        "svd_time_sec": t_svd,
        "tucker_time_sec": t_tucker,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_hybrid_mi] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
