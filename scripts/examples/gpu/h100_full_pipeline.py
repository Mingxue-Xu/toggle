#!/usr/bin/env python3
"""
Strategy 8: Full Pipeline - Analysis + Pruning + SVD + Evaluation for H100 80GB.

Complete pipeline using both ActivationMetricsPlugin and WeightMetricsPlugin
for maximum compression: analyze -> prune blocks -> apply SVD -> evaluate.

Now includes LMHarness evaluation step for benchmarking compressed model quality.

Usage:
  HF_TOKEN=... python scripts/examples/h100_full_pipeline.py [--config path] [--model name]

  Additional options:
    --eval-tasks TASK1,TASK2  Evaluation tasks (default: hellaswag,arc_easy)
    --eval-limit N            Limit evaluation samples (default: None)
    --skip-eval               Skip evaluation step
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
from src.plugins.analysis.weight_metrics import WeightMetricsPlugin
from src.plugins.compression.pruning import PruningPlugin
from src.plugins.compression.consolidator import ModelConsolidator
from toggle.src.plugins.evaluation.lm_eval import LMHarness


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full Compression Pipeline for H100")
    parser.add_argument("--config", default="config/h100_full_pipeline.yaml", help="Config YAML path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--workspace", default=None, help="Override workspace directory")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Override pruning ratio")
    parser.add_argument("--svd-rank", type=int, default=None, help="Override SVD rank")
    parser.add_argument("--eval-tasks", default="hellaswag,arc_easy", help="Evaluation tasks (comma-separated)")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = args.model or cfg["model"]["name"]
    token = os.environ.get(cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN"))

    workspace = Path(args.workspace) if args.workspace else _resolve_path(cfg.get("runtime", {}).get("workspace", "logs/h100_full_pipeline"))
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[h100_full_pipeline] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    original_params = params_count(model)
    print(f"[h100_full_pipeline] Original params: {original_params:,}")

    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    timings = {}

    # ========== Step 1: Activation Analysis ==========
    print("\n[h100_full_pipeline] Step 1: Activation Analysis")
    act_plugin = ActivationMetricsPlugin()
    act_plugin.initialize(ctx)
    t0 = time.perf_counter()
    act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
    timings["activation_analysis"] = time.perf_counter() - t0
    print(f"  Completed in {timings['activation_analysis']:.2f}s")

    # Extract key metrics
    spectral_norms = {}
    entropy_values = {}
    for layer in act_report.get("per_layer", []):
        name = layer.get("name", "")
        stats = layer.get("statistics", {})
        spectral_norms[name] = stats.get("spectral_norm.median", 0)
        entropy_values[name] = stats.get("histogram_entropy.median", 0)

    # ========== Step 2: Weight Analysis ==========
    print("\n[h100_full_pipeline] Step 2: Weight Analysis")
    weight_plugin = WeightMetricsPlugin()
    weight_plugin.initialize(ctx)
    t1 = time.perf_counter()
    weight_report = weight_plugin.do_execute(model=model)
    timings["weight_analysis"] = time.perf_counter() - t1
    print(f"  Completed in {timings['weight_analysis']:.2f}s")

    # Extract weight metrics
    layer_l2_norms = {}
    layer_sparsity = {}
    for layer in weight_report.get("per_layer", []):
        name = layer.get("name", "")
        values = layer.get("values", {})
        layer_l2_norms[name] = values.get("l2_norm", 0)
        layer_sparsity[name] = values.get("sparsity", 0)

    # ========== Step 3: Block Pruning ==========
    print("\n[h100_full_pipeline] Step 3: Block Pruning")
    params_before_pruning = params_count(model)

    pruning_cfg = cfg.get("pruning", {})
    prune_ratio = args.prune_ratio or pruning_cfg.get("ratio_deleted_block", 0.15)

    pruning_plugin = PruningPlugin()
    pruning_plugin.initialize(ctx)
    pruning_plugin._configuration = {
        "pruning": {
            "selection_metric_type": pruning_cfg.get("selection_metric_type", "l2_norm.median"),
            "ratio_deleted_block": prune_ratio,
            "min_keep": pruning_cfg.get("min_keep", 4),
            "container_path": pruning_cfg.get("container_path", "model.layers"),
            "group_prefix": pruning_cfg.get("group_prefix", "layers"),
            "aggregator": pruning_cfg.get("aggregator", "mean"),
        }
    }

    t2 = time.perf_counter()
    pruning_result = pruning_plugin.do_execute(model=model)
    timings["pruning"] = time.perf_counter() - t2

    pruned_indices = pruning_result.get("selected_indices", [])
    params_after_pruning = params_count(model)
    pruning_reduction = 1 - (params_after_pruning / params_before_pruning)

    print(f"  Pruned {len(pruned_indices)} blocks: {pruned_indices}")
    print(f"  Params after pruning: {params_after_pruning:,} ({pruning_reduction*100:.1f}% reduction)")
    print(f"  Completed in {timings['pruning']:.2f}s")

    # ========== Step 4: SVD Compression ==========
    print("\n[h100_full_pipeline] Step 4: SVD Compression")
    params_before_svd = params_count(model)

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

    svd_rank = args.svd_rank or fact_cfg.get("rank", 48)
    svd_backend = cfg.get("compression", {}).get("svd", {}).get("backend", "cola")
    svd_backend_config = cfg.get("compression", {}).get("svd", {}).get("cola", {"algorithm": "lanczos"})

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=objects,
        rank=svd_rank,
        svd_backend=svd_backend,
        svd_backend_config=svd_backend_config,
    )
    consolidator.initialize(ctx)

    t3 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    timings["svd_compression"] = time.perf_counter() - t3

    params_after_svd = params_count(model)
    svd_reduction = 1 - (params_after_svd / params_before_svd)

    print(f"  SVD rank: {svd_rank}")
    print(f"  Params after SVD: {params_after_svd:,} ({svd_reduction*100:.1f}% reduction from pre-SVD)")
    print(f"  Completed in {timings['svd_compression']:.2f}s")

    # ========== Step 5: LMHarness Evaluation ==========
    evaluation_results = {}
    if not args.skip_eval:
        print("\n[h100_full_pipeline] Step 5: LMHarness Evaluation")
        eval_tasks = args.eval_tasks.split(",")
        evaluator = LMHarness(tasks=eval_tasks, backend="auto")

        for task in eval_tasks:
            print(f"  Evaluating: {task}")
            t_eval = time.perf_counter()
            result = evaluator.evaluate_task(model, tokenizer, task, limit=args.eval_limit)
            eval_time = time.perf_counter() - t_eval
            timings[f"eval_{task}"] = eval_time

            evaluation_results[task] = {
                **result.metrics,
                "eval_time_sec": eval_time,
            }
            acc = result.metrics.get("accuracy", result.metrics.get("acc", "N/A"))
            print(f"    Accuracy: {acc}")
            print(f"    Time: {eval_time:.2f}s")

        timings["evaluation_total"] = sum(timings.get(f"eval_{t}", 0) for t in eval_tasks)
        print(f"  Total evaluation time: {timings['evaluation_total']:.2f}s")
    else:
        print("\n[h100_full_pipeline] Step 5: Evaluation SKIPPED")

    # ========== Final Results ==========
    compressed_params = params_count(model)
    total_ratio = compressed_params / original_params
    total_time = sum(timings.values())

    print(f"\n{'='*60}")
    print(f"[h100_full_pipeline] FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Original params:    {original_params:,}")
    print(f"  After pruning:      {params_after_pruning:,}")
    print(f"  After SVD:          {compressed_params:,}")
    print(f"  Total compression:  {total_ratio:.4f} ({(1-total_ratio)*100:.1f}% reduction)")
    print(f"\n  Timings:")
    for step, t in timings.items():
        print(f"    {step}: {t:.2f}s")
    print(f"    TOTAL: {total_time:.2f}s")

    # Save compressed model
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "h100_full_pipeline",
        "base_model": model_name,
        "strategy": "full_pipeline",
        "compression_ratio": total_ratio,
        "pruned_blocks": pruned_indices,
        "svd_rank": svd_rank,
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    print(f"\n[h100_full_pipeline] Saved to: {save_dir}")

    # Save comprehensive report
    report_path = workspace / "full_pipeline_report.json"
    report = {
        "model": model_name,
        "strategy": "full_pipeline",
        "original_params": original_params,
        "params_after_pruning": params_after_pruning,
        "params_after_svd": compressed_params,
        "compression_ratio": total_ratio,
        "pruning": {
            "ratio": prune_ratio,
            "pruned_indices": pruned_indices,
            "reduction": pruning_reduction,
        },
        "svd": {
            "rank": svd_rank,
            "reduction": svd_reduction,
        },
        "analysis": {
            "spectral_norms": spectral_norms,
            "entropy_values": entropy_values,
            "l2_norms": layer_l2_norms,
            "sparsity": layer_sparsity,
        },
        "evaluation": evaluation_results,
        "timings": timings,
        "total_time_sec": total_time,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[h100_full_pipeline] Report saved to: {report_path}")

    # Print evaluation summary if available
    if evaluation_results:
        print(f"\n{'='*60}")
        print(f"[h100_full_pipeline] EVALUATION SUMMARY")
        print(f"{'='*60}")
        for task, metrics in evaluation_results.items():
            acc = metrics.get("accuracy", metrics.get("acc", "N/A"))
            print(f"  {task}: {acc}")


if __name__ == "__main__":
    main()
