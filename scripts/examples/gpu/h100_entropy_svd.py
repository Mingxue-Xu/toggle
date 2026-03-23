#!/usr/bin/env python3
"""
Strategy 5: Entropy-Based Selective SVD Compression for H100 80GB.

Uses histogram_entropy from activation analysis to identify layers with low
information content for aggressive compression.

Usage:
  HF_TOKEN=... python scripts/examples/h100_entropy_svd.py [--config path] [--model name]
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    parser = argparse.ArgumentParser(description="Entropy-Based SVD Compression for H100")
    parser.add_argument("--config", default="config/h100_entropy_svd.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    return parser.parse_args()


def _to_bracket_module_path(path: str, group_prefix: str = "layers") -> str:
    pattern = rf"(^|\.)(%s)\.(\d+)(?=\.|$)" % re.escape(group_prefix)
    return re.sub(pattern, lambda m: f"{m.group(1)}{m.group(2)}[{m.group(3)}]", path)


def _get_reduced_stat(stats: dict, metric_key: str, reducer: str = "median"):
    nested = stats.get(metric_key)
    if isinstance(nested, dict) and reducer in nested:
        return nested.get(reducer)
    return stats.get(f"{metric_key}.{reducer}")


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_entropy_svd"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_entropy_svd] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_entropy_svd] Original params: {original_params:,}")

    # Step 1: Activation Analysis for entropy measurement
    print("[h100_entropy_svd] Running activation analysis for entropy measurement...")
    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    t_analysis = time.perf_counter() - t0
    print(f"[h100_entropy_svd] Analysis completed in {t_analysis:.2f}s")

    # Extract histogram_entropy from analysis
    layer_entropy = {}
    entropy_values = []
    for layer in act_report.get("per_layer", []):
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        entropy = _get_reduced_stat(stats, "compute_lda_matrix.histogram_entropy", "median")
        if entropy is None:
            entropy = _get_reduced_stat(stats, "histogram_entropy", "median")
        if entropy is not None:
            try:
                ent = float(entropy)
                layer_entropy[_to_bracket_module_path(name)] = ent
                entropy_values.append(ent)
            except Exception:
                pass

    # Compute entropy thresholds for rank assignment
    if entropy_values:
        sorted_entropy = sorted(entropy_values)
        low_threshold = sorted_entropy[len(sorted_entropy) // 3]
        high_threshold = sorted_entropy[2 * len(sorted_entropy) // 3]
        print(f"[h100_entropy_svd] Entropy thresholds: low={low_threshold:.2f}, high={high_threshold:.2f}")
    else:
        low_threshold = 2.0
        high_threshold = 4.0

    # Step 2: Build layer-specific overrides based on entropy
    entropy_cfg = cfg.get("entropy_compression", {})
    low_rank = entropy_cfg.get("low_entropy_rank", 32)
    mid_rank = entropy_cfg.get("mid_entropy_rank", 64)
    high_rank = entropy_cfg.get("high_entropy_rank", 128)

    layer_ranks = {}
    for name, entropy in layer_entropy.items():
        if entropy < low_threshold:
            layer_ranks[name] = low_rank
        elif entropy < high_threshold:
            layer_ranks[name] = mid_rank
        else:
            layer_ranks[name] = high_rank

    print(f"[h100_entropy_svd] Assigned ranks to {len(layer_ranks)} layers")
    rank_distribution = {low_rank: 0, mid_rank: 0, high_rank: 0}
    for r in layer_ranks.values():
        if r in rank_distribution:
            rank_distribution[r] += 1
    print(f"[h100_entropy_svd] Rank distribution: {rank_distribution}")

    # Step 3: Apply SVD Compression with entropy-based ranks
    print("[h100_entropy_svd] Applying entropy-based SVD compression...")
    fact_cfg = cfg.get("factorization", {})
    objects = fact_cfg.get("objects", [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ])

    # Build overrides from entropy analysis
    base_overrides = list(fact_cfg.get("overrides", []) or [])
    computed_overrides = [
        {
            "pattern": module_path,
            "func_name": "svd",
            "rank": int(rank),
            "granularity": "matrix",
        }
        for module_path, rank in layer_ranks.items()
    ]
    overrides = computed_overrides + base_overrides

    svd_backend = cfg.get("compression", {}).get("svd", {}).get("backend", "cola")
    svd_backend_config = cfg.get("compression", {}).get("svd", {}).get("cola", {"algorithm": "lanczos"})

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=objects,
        method_overrides=overrides if overrides else None,
        rank=mid_rank,  # Default to mid rank
        svd_backend=svd_backend,
        svd_backend_config=svd_backend_config,
    )
    consolidator.initialize(ctx)

    t1 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    t_compress = time.perf_counter() - t1

    compressed_params = params_count(model)
    ratio = compressed_params / original_params

    print(f"\n[h100_entropy_svd] Results:")
    print(f"  Original params:   {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {ratio:.4f} ({(1-ratio)*100:.1f}% reduction)")
    print(f"  Analysis time:     {t_analysis:.2f}s")
    print(f"  Compression time:  {t_compress:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_entropy_svd",
        "base_model": model_name,
        "strategy": "entropy_based_svd",
        "compression_ratio": ratio,
        "rank_distribution": rank_distribution,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_entropy_svd] Saved to: {save_dir}")

    # Save report
    report_path = workspace / "entropy_svd_report.json"
    report = {
        "model": model_name,
        "strategy": "entropy_based_svd",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": ratio,
        "layer_entropy": layer_entropy,
        "layer_ranks": layer_ranks,
        "applied_overrides": len(overrides),
        "entropy_thresholds": {"low": low_threshold, "high": high_threshold},
        "rank_distribution": rank_distribution,
        "analysis_time_sec": t_analysis,
        "compression_time_sec": t_compress,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_entropy_svd] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
