#!/usr/bin/env python3
"""
Comparison Runs: Baseline vs Compressed Model Evaluation.

Runs systematic comparison between uncompressed (baseline) and compressed
models using LM Evaluation tasks. Required for the Related Software section
in JMLR MLOSS papers.

Demonstrates:
- BaselineEvaluator for uncompressed model profiling
- LMHarness for task evaluation on both baseline and compressed
- Method comparison table generation

Usage:
  HF_TOKEN=... python scripts/examples/comparison_baseline_compressed.py [options]

  Options:
    --model MODEL             Model name (default: meta-llama/Llama-3.2-1B-Instruct)
    --methods METHOD1,...     Compression methods to compare (default: svd,tucker)
    --ratio RATIO             Target compression ratio (default: 0.5)
    --tasks TASK1,TASK2,...   Evaluation tasks (default: hellaswag,arc_easy,piqa)
    --workspace PATH          Workspace directory (default: logs/comparison)
    --limit N                 Limit evaluation samples (for testing)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.compression.consolidator import ModelConsolidator
from src.plugins.evaluation.lm_eval import LMHarness


@dataclass
class ComparisonResult:
    """Single comparison result entry."""
    model: str
    variant: str  # "baseline" or compression method
    compression_ratio: float
    task: str
    metric_name: str
    metric_value: float
    params: int
    inference_time_ms: Optional[float]
    memory_mb: Optional[float]
    timestamp: str


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_baseline(
    model: torch.nn.Module,
    tokenizer,
    device: str,
) -> Dict[str, Any]:
    """Profile baseline model with inline profiling."""
    import psutil

    results = {}
    model.eval()
    model.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    results["params"] = {
        "total_parameters": float(total_params),
        "parameters_millions": float(total_params / 1e6)
    }

    # Memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    results["memory"] = {
        "model_memory_mb": memory_mb,
        "model_gpu_memory_mb": gpu_memory
    }

    # Inference speed (simple benchmark)
    sample_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)

    # Time inference
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)
    elapsed = time.perf_counter() - start

    results["inference"] = {
        "avg_inference_time_ms": (elapsed / 10) * 1000,
        "samples_per_second": 10 / elapsed
    }

    return results


def evaluate_model_tasks(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    limit: Optional[int] = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluate model on LM tasks."""
    evaluator = LMHarness(tasks=tasks)
    task_results = {}
    total_time = 0.0

    for task in tasks:
        t0 = time.perf_counter()
        result = evaluator.evaluate_task(model, tokenizer, task, limit=limit)
        eval_time = time.perf_counter() - t0
        total_time += eval_time

        task_results[task] = {
            **result.metrics,
            "eval_time_sec": eval_time,
        }

    return total_time, task_results


def compress_model_for_comparison(
    model: torch.nn.Module,
    method: str,
    target_ratio: float,
    ctx: PipelineContext,
) -> Tuple[float, Dict[str, Any]]:
    """Compress model using specified method."""
    original_params = params_count(model)

    # Find a representative linear layer for rank estimation
    sample_m, sample_n = 512, 512
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    # Estimate rank from target ratio
    if method == "svd":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    elif method == "tucker":
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.7))
    elif method == "cp":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n)))
    else:  # tensor_train
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.5))

    rank = min(rank, min(sample_m, sample_n))

    # Target modules
    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    # Configure consolidator
    consolidator_kwargs = {
        "compression_method": method,
        "target_modules": target_modules,
    }

    if method == "svd":
        consolidator_kwargs.update({
            "rank": rank,
            "svd_backend": "torch",
        })
    elif method == "tucker":
        consolidator_kwargs["tucker_ranks"] = [rank, rank]
    elif method == "cp":
        consolidator_kwargs["cp_rank"] = rank
    elif method == "tensor_train":
        consolidator_kwargs["tensor_ranks"] = [1, rank, rank, 1]

    consolidator = ModelConsolidator(**consolidator_kwargs)
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    result = consolidator.compress_model_with_surgery(model)
    compression_time = time.perf_counter() - t0

    compressed_params = params_count(model)
    actual_ratio = compressed_params / original_params

    return compression_time, {
        "original_params": original_params,
        "compressed_params": compressed_params,
        "actual_ratio": actual_ratio,
        "rank": rank,
        "layers_replaced": len(result.parameters.get("layers_replaced", [])),
    }


def run_comparison(
    model_name: str,
    methods: List[str],
    target_ratio: float,
    tasks: List[str],
    workspace: Path,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[ComparisonResult]:
    """Run full comparison between baseline and compressed models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().isoformat()
    results: List[ComparisonResult] = []

    print(f"\n{'='*60}")
    print(f"Comparison: {model_name}")
    print(f"Methods: {methods}")
    print(f"Target ratio: {target_ratio}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )

    # === BASELINE EVALUATION ===
    print(f"\n[Baseline] Loading and evaluating...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
    )
    baseline_model.to(device).eval()
    baseline_params = params_count(baseline_model)

    # Profile baseline
    print(f"[Baseline] Profiling...")
    baseline_profile = profile_baseline(baseline_model, tokenizer, device)
    baseline_inference_ms = baseline_profile.get("inference", {}).get("avg_inference_time_ms")
    baseline_memory_mb = baseline_profile.get("memory", {}).get("model_gpu_memory_mb")

    # Evaluate baseline on tasks
    print(f"[Baseline] Evaluating tasks...")
    _, baseline_tasks = evaluate_model_tasks(baseline_model, tokenizer, tasks, limit)

    # Record baseline results
    for task, metrics in baseline_tasks.items():
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and metric_name != "eval_time_sec":
                results.append(ComparisonResult(
                    model=model_name,
                    variant="baseline",
                    compression_ratio=1.0,
                    task=task,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    params=baseline_params,
                    inference_time_ms=baseline_inference_ms,
                    memory_mb=baseline_memory_mb,
                    timestamp=timestamp,
                ))

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === COMPRESSED MODEL EVALUATIONS ===
    for method in methods:
        print(f"\n[{method.upper()}] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
        )
        model.to(device).eval()

        method_workspace = workspace / method
        method_workspace.mkdir(parents=True, exist_ok=True)
        ctx = PipelineContext(config={}, workspace_dir=method_workspace)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        try:
            # Compress
            print(f"[{method.upper()}] Compressing...")
            compression_time, comp_info = compress_model_for_comparison(
                model, method, target_ratio, ctx
            )
            print(f"  Compression time: {compression_time:.2f}s")
            print(f"  Actual ratio: {comp_info['actual_ratio']:.4f}")

            # Profile compressed
            print(f"[{method.upper()}] Profiling...")
            compressed_profile = profile_baseline(model, tokenizer, device)
            compressed_inference_ms = compressed_profile.get("inference", {}).get("avg_inference_time_ms")
            compressed_memory_mb = compressed_profile.get("memory", {}).get("model_gpu_memory_mb")

            # Evaluate compressed on tasks
            print(f"[{method.upper()}] Evaluating tasks...")
            _, compressed_tasks = evaluate_model_tasks(model, tokenizer, tasks, limit)

            # Record compressed results
            for task, metrics in compressed_tasks.items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and metric_name != "eval_time_sec":
                        results.append(ComparisonResult(
                            model=model_name,
                            variant=method,
                            compression_ratio=comp_info["actual_ratio"],
                            task=task,
                            metric_name=metric_name,
                            metric_value=float(metric_value),
                            params=comp_info["compressed_params"],
                            inference_time_ms=compressed_inference_ms,
                            memory_mb=compressed_memory_mb,
                            timestamp=timestamp,
                        ))

            # Save compressed model
            save_dir = method_workspace / "compressed"
            save_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "type": f"comparison_{method}",
                "base_model": model_name,
                "method": method,
                "compression_ratio": comp_info["actual_ratio"],
            }
            save_compressed_to_safetensors(model, manifest, save_dir)

        except Exception as e:
            print(f"  ERROR: {e}")
            for task in tasks:
                results.append(ComparisonResult(
                    model=model_name,
                    variant=method,
                    compression_ratio=0.0,
                    task=task,
                    metric_name="error",
                    metric_value=0.0,
                    params=0,
                    inference_time_ms=None,
                    memory_mb=None,
                    timestamp=timestamp,
                ))

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def generate_comparison_table(results: List[ComparisonResult], output_dir: Path):
    """Generate comparison tables in markdown format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Baseline vs Compressed Model Comparison\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Group by model
        models = sorted(set(r.model for r in results))

        for model in models:
            f.write(f"## {model}\n\n")

            # Get unique tasks
            model_results = [r for r in results if r.model == model]
            tasks = sorted(set(r.task for r in model_results))

            # Accuracy comparison table
            f.write("### Accuracy Comparison\n\n")
            f.write("| Method | Ratio | " + " | ".join(tasks) + " |\n")
            f.write("|--------|-------|" + "|".join(["------" for _ in tasks]) + "|\n")

            variants = sorted(set(r.variant for r in model_results))
            for variant in variants:
                variant_results = [r for r in model_results if r.variant == variant and r.metric_name == "accuracy"]
                if not variant_results:
                    continue

                ratio = variant_results[0].compression_ratio if variant_results else 1.0
                row_values = []
                for task in tasks:
                    task_result = next(
                        (r for r in variant_results if r.task == task),
                        None
                    )
                    if task_result:
                        row_values.append(f"{task_result.metric_value:.4f}")
                    else:
                        row_values.append("N/A")

                f.write(f"| {variant} | {ratio:.2f} | " + " | ".join(row_values) + " |\n")

            f.write("\n")

            # Performance comparison table
            f.write("### Performance Comparison\n\n")
            f.write("| Method | Params | Ratio | Inference (ms) | Memory (MB) |\n")
            f.write("|--------|--------|-------|----------------|-------------|\n")

            for variant in variants:
                variant_results = [r for r in model_results if r.variant == variant]
                if not variant_results:
                    continue

                sample = variant_results[0]
                params_m = sample.params / 1e6
                ratio = sample.compression_ratio
                inf_time = f"{sample.inference_time_ms:.2f}" if sample.inference_time_ms else "N/A"
                mem = f"{sample.memory_mb:.1f}" if sample.memory_mb else "N/A"

                f.write(f"| {variant} | {params_m:.1f}M | {ratio:.2f} | {inf_time} | {mem} |\n")

            f.write("\n")

    print(f"Comparison table saved to: {table_path}")
    return table_path


def save_results(results: List[ComparisonResult], output_dir: Path):
    """Save results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "comparison_results.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # Save to JSON
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    print(f"Results saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs Compressed Comparison")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--methods", default="svd,tucker", help="Compression methods")
    parser.add_argument("--ratio", type=float, default=0.5, help="Target compression ratio")
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa", help="Evaluation tasks")
    parser.add_argument("--workspace", default="logs/comparison", help="Workspace directory")
    parser.add_argument("--limit", type=int, default=None, help="Evaluation sample limit")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    methods = args.methods.split(",")
    target_ratio = args.ratio
    tasks = args.tasks.split(",")
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    print("Baseline vs Compressed Comparison")
    print("==================================")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Target ratio: {target_ratio}")
    print(f"Tasks: {tasks}")
    print(f"Workspace: {workspace}")

    results = run_comparison(
        model_name=model_name,
        methods=methods,
        target_ratio=target_ratio,
        tasks=tasks,
        workspace=workspace,
        limit=args.limit,
        hf_token=hf_token,
    )

    save_results(results, workspace)
    generate_comparison_table(results, workspace)

    print(f"\nCompleted {len(results)} comparison entries.")


if __name__ == "__main__":
    main()
