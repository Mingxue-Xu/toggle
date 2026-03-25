#!/usr/bin/env python3
"""
Comparison: Baseline vs Compressed Model Evaluation - H200 GPU Version.

Compares uncompressed baseline against compressed models across multiple
metrics for JMLR MLOSS "Related Software" section requirements.

Optimized for H200 GPU with 140GB VRAM and BF16 precision.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/comparison_baseline_compressed.py [options]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.plugins.compression.consolidator import ModelConsolidator
from goldcrest.plugins.evaluation.lm_eval import LMHarness


MODEL_TIERS = {
    "mid": ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B"],
    "large": ["google/gemma-2-27b", "mistralai/Mixtral-8x7B-v0.1"],
    "flagship": ["meta-llama/Llama-3.1-70B"],
}

DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]


@dataclass
class ComparisonResult:
    """Single comparison result."""
    model: str
    is_baseline: bool
    method: str
    compression_ratio: float
    task: str
    metric_name: str
    metric_value: float
    original_params: int
    compressed_params: int
    param_ratio: float
    evaluation_time_sec: float
    gpu_memory_gb: float
    timestamp: str


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def compress_model(model, method: str, ratio: float, ctx) -> Dict[str, Any]:
    """Compress model with specified method."""
    original_params = params_count(model)

    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    if method == "svd":
        rank = max(1, int(ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
        kwargs = {"rank": min(rank, min(sample_m, sample_n)), "svd_backend": "cola"}
    elif method == "tucker":
        rank = max(1, int(ratio * min(sample_m, sample_n) * 0.7))
        kwargs = {"tucker_ranks": [min(rank, sample_m), min(rank, sample_n)]}
    elif method == "cp":
        rank = max(1, int(ratio * (sample_m * sample_n) / (sample_m + sample_n)))
        kwargs = {"cp_rank": min(rank, min(sample_m, sample_n))}
    else:  # tensor_train
        rank = max(1, int(ratio * min(sample_m, sample_n) * 0.5))
        kwargs = {"tensor_ranks": [1, min(rank, sample_m), min(rank, sample_n), 1]}

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    consolidator = ModelConsolidator(
        compression_method=method,
        target_modules=target_modules,
        **kwargs,
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

    compressed_params = params_count(model)
    return {
        "original_params": original_params,
        "compressed_params": compressed_params,
        "actual_ratio": compressed_params / original_params,
    }


def run_comparison(
    model_name: str,
    methods: List[str],
    ratios: List[float],
    tasks: List[str],
    workspace: Path,
    eval_limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[ComparisonResult]:
    """Run baseline vs compressed comparison."""
    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU comparison requires CUDA.")

    device = "cuda"
    timestamp = datetime.now().isoformat()
    results: List[ComparisonResult] = []

    print(f"\n{'='*60}")
    print(f"H200 GPU Baseline vs Compressed Comparison")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    evaluator = LMHarness(tasks=tasks, backend="auto")

    # === BASELINE ===
    print("\n[Baseline] Loading and evaluating...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    baseline_model.eval()
    baseline_params = params_count(baseline_model)
    gpu_mem = get_gpu_memory_gb()
    print(f"  Parameters: {baseline_params:,}, GPU: {gpu_mem:.2f} GB")

    for task in tasks:
        t0 = time.perf_counter()
        result = evaluator.evaluate_task(baseline_model, tokenizer, task, limit=eval_limit)
        eval_time = time.perf_counter() - t0

        for metric_name, metric_value in result.metrics.items():
            if isinstance(metric_value, (int, float)):
                results.append(ComparisonResult(
                    model=model_name,
                    is_baseline=True,
                    method="baseline",
                    compression_ratio=1.0,
                    task=task,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    original_params=baseline_params,
                    compressed_params=baseline_params,
                    param_ratio=1.0,
                    evaluation_time_sec=eval_time,
                    gpu_memory_gb=gpu_mem,
                    timestamp=timestamp,
                ))
        print(f"  {task}: {result.metrics.get('accuracy', 'N/A')}")

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    # === COMPRESSED ===
    for method in methods:
        for ratio in ratios:
            print(f"\n[{method.upper()}] Ratio: {ratio}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
                token=hf_token, device_map="auto",
            )
            model.eval()

            method_workspace = workspace / f"{method}_{ratio}"
            method_workspace.mkdir(parents=True, exist_ok=True)
            ctx = PipelineContext(config={}, workspace_dir=method_workspace)
            ctx.state.model = model
            ctx.state.tokenizer = tokenizer

            try:
                comp_info = compress_model(model, method, ratio, ctx)
                gpu_mem = get_gpu_memory_gb()
                print(f"  Compression: {comp_info['actual_ratio']:.4f}, GPU: {gpu_mem:.2f} GB")

                for task in tasks:
                    t0 = time.perf_counter()
                    result = evaluator.evaluate_task(model, tokenizer, task, limit=eval_limit)
                    eval_time = time.perf_counter() - t0

                    for metric_name, metric_value in result.metrics.items():
                        if isinstance(metric_value, (int, float)):
                            results.append(ComparisonResult(
                                model=model_name,
                                is_baseline=False,
                                method=method,
                                compression_ratio=ratio,
                                task=task,
                                metric_name=metric_name,
                                metric_value=float(metric_value),
                                original_params=comp_info["original_params"],
                                compressed_params=comp_info["compressed_params"],
                                param_ratio=comp_info["actual_ratio"],
                                evaluation_time_sec=eval_time,
                                gpu_memory_gb=gpu_mem,
                                timestamp=timestamp,
                            ))
                    print(f"  {task}: {result.metrics.get('accuracy', 'N/A')}")

            except Exception as e:
                print(f"  ERROR: {e}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

    return results


def save_results(results: List[ComparisonResult], output_dir: Path):
    """Save comparison results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison_results.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    summary_path = output_dir / "comparison_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# H200 GPU Baseline vs Compressed Comparison\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Results\n\n")
        f.write("| Method | Ratio | Task | Accuracy | vs Baseline |\n")
        f.write("|--------|-------|------|----------|-------------|\n")

        baseline_acc = {}
        for r in results:
            if r.is_baseline and r.metric_name == "accuracy":
                baseline_acc[r.task] = r.metric_value

        for r in sorted(results, key=lambda x: (x.method, x.compression_ratio, x.task)):
            if r.metric_name != "accuracy":
                continue
            base = baseline_acc.get(r.task, 0)
            diff = r.metric_value - base if not r.is_baseline else 0
            f.write(f"| {r.method} | {r.compression_ratio:.2f} | {r.task} | ")
            f.write(f"{r.metric_value:.4f} | {diff:+.4f} |\n")

    print(f"\nResults saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Baseline vs Compressed Comparison")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--methods", default="svd,tucker,cp,tensor_train", help="Methods")
    parser.add_argument("--ratios", default="0.5", help="Compression ratios")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS[:3]), help="Tasks")
    parser.add_argument("--eval-limit", type=int, default=100, help="Eval sample limit")
    parser.add_argument("--workspace", default="logs/gpu_comparison", help="Workspace")
    parser.add_argument("--tier", choices=["mid", "large", "flagship"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = MODEL_TIERS[args.tier][0] if args.tier else args.model
    methods = args.methods.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    tasks = args.tasks.split(",")
    workspace = Path(args.workspace).resolve()
    hf_token = os.environ.get("HF_TOKEN")

    results = run_comparison(
        model_name=model_name,
        methods=methods,
        ratios=ratios,
        tasks=tasks,
        workspace=workspace,
        eval_limit=args.eval_limit,
        hf_token=hf_token,
    )

    save_results(results, workspace)


if __name__ == "__main__":
    main()
