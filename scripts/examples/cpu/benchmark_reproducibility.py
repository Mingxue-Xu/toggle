#!/usr/bin/env python3
"""
Benchmark Reproducibility Runs for JMLR MLOSS Paper.

Runs systematic benchmarks across multiple models, compression methods, and
compression ratios with LM Evaluation Harness integration.

Following TensorLy's example of explicit benchmark numbers for paper.

Models:
- meta-llama/Llama-3.2-1B-Instruct
- Qwen/Qwen2.5-0.5B
- google/gemma-2b

Methods: SVD, Tucker, CP, TT
Compression Ratios: 0.3, 0.5, 0.7
Tasks: hellaswag, arc_easy, piqa

Usage:
  HF_TOKEN=... python scripts/examples/benchmark_reproducibility.py [options]

  Options:
    --models MODEL1,MODEL2,...   Comma-separated model names (default: all)
    --methods METHOD1,METHOD2,...  Comma-separated methods (default: all)
    --ratios RATIO1,RATIO2,...   Comma-separated ratios (default: 0.3,0.5,0.7)
    --tasks TASK1,TASK2,...      Comma-separated tasks (default: hellaswag,arc_easy,piqa)
    --output-dir PATH            Output directory for results (default: logs/benchmark_reproducibility)
    --limit N                    Limit evaluation samples per task (for testing)
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

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.compression.consolidator import ModelConsolidator
from toggle.src.plugins.evaluation.lm_eval import LMHarness
from toggle.src.plugins.evaluation.baseline_eval import UncompressedModelProfile


@dataclass
class BenchmarkResult:
    """Single benchmark result entry."""
    model: str
    method: str
    compression_ratio: float
    task: str
    metric_name: str
    metric_value: float
    original_params: int
    compressed_params: int
    param_ratio: float
    compression_time_sec: float
    evaluation_time_sec: float
    timestamp: str


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_default_config() -> Dict[str, Any]:
    """Get default benchmark configuration."""
    return {
        "models": [
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
        "methods": ["svd", "tucker", "cp", "tensor_train"],
        "compression_ratios": [0.3, 0.5, 0.7],
        "tasks": ["hellaswag", "arc_easy", "piqa"],
        "method_configs": {
            "svd": {
                "objects": [
                    "model.layers[*].self_attn.q_proj",
                    "model.layers[*].self_attn.k_proj",
                    "model.layers[*].self_attn.v_proj",
                    "model.layers[*].self_attn.o_proj",
                    "model.layers[*].mlp.gate_proj",
                    "model.layers[*].mlp.up_proj",
                    "model.layers[*].mlp.down_proj",
                ],
                "backend": "cola",
                "backend_config": {"algorithm": "lanczos"},
            },
            "tucker": {
                "objects": [
                    "model.layers[*].self_attn.q_proj",
                    "model.layers[*].self_attn.k_proj",
                    "model.layers[*].mlp.gate_proj",
                ],
            },
            "cp": {
                "objects": [
                    "model.layers[*].self_attn.q_proj",
                    "model.layers[*].self_attn.v_proj",
                ],
            },
            "tensor_train": {
                "objects": [
                    "model.layers[*].mlp.up_proj",
                    "model.layers[*].mlp.down_proj",
                ],
            },
        },
    }


def compute_rank_for_ratio(m: int, n: int, target_ratio: float, method: str) -> int:
    """Compute rank that achieves approximately target parameter ratio."""
    original_params = m * n

    if method == "svd":
        # SVD: U(m,r) + S(r) + Vt(r,n) = r*(m+n+1)
        # ratio = r*(m+n+1) / (m*n)
        r = int(target_ratio * original_params / (m + n + 1))
    elif method == "tucker":
        # Tucker: similar approximation
        r = int(target_ratio * min(m, n) * 0.7)
    elif method == "cp":
        # CP: r * (m + n)
        r = int(target_ratio * original_params / (m + n))
    else:  # tensor_train
        r = int(target_ratio * min(m, n) * 0.5)

    return max(1, min(r, min(m, n)))


def compress_model(
    model: torch.nn.Module,
    method: str,
    target_ratio: float,
    method_config: Dict[str, Any],
    ctx: PipelineContext,
) -> Tuple[float, Dict[str, Any]]:
    """Compress model using specified method and target ratio."""
    original_params = params_count(model)

    # Estimate rank from target ratio using a representative layer
    sample_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_layer = module
            break

    if sample_layer is not None:
        m, n = sample_layer.weight.shape
        rank = compute_rank_for_ratio(m, n, target_ratio, method)
    else:
        rank = 32

    # Configure consolidator based on method
    consolidator_kwargs = {
        "compression_method": method,
        "target_modules": method_config.get("objects", []),
    }

    if method == "svd":
        consolidator_kwargs.update({
            "rank": rank,
            "svd_backend": method_config.get("backend", "torch"),
            "svd_backend_config": method_config.get("backend_config", {}),
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
        "target_ratio": target_ratio,
        "rank": rank,
    }


def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    device: str,
    limit: Optional[int] = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluate model on specified tasks."""
    evaluator = LMHarness(tasks=tasks, backend="auto")

    task_results = {}
    total_time = 0.0

    for task in tasks:
        t0 = time.perf_counter()
        result = evaluator.evaluate_task(model, tokenizer, task, limit=limit)
        eval_time = time.perf_counter() - t0
        total_time += eval_time

        task_results[task] = result.metrics

    return total_time, task_results


def run_benchmark(
    models: List[str],
    methods: List[str],
    ratios: List[float],
    tasks: List[str],
    output_dir: Path,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    config = get_default_config()
    results: List[BenchmarkResult] = []
    timestamp = datetime.now().isoformat()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load model
        print(f"Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token
        )

        # First, evaluate baseline
        print(f"Evaluating baseline...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
        )
        base_model.to(device).eval()
        baseline_params = params_count(base_model)

        baseline_eval_time, baseline_metrics = evaluate_model(
            base_model, tokenizer, tasks, device, limit
        )

        # Record baseline results
        for task, metrics in baseline_metrics.items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    results.append(BenchmarkResult(
                        model=model_name,
                        method="baseline",
                        compression_ratio=1.0,
                        task=task,
                        metric_name=metric_name,
                        metric_value=float(metric_value),
                        original_params=baseline_params,
                        compressed_params=baseline_params,
                        param_ratio=1.0,
                        compression_time_sec=0.0,
                        evaluation_time_sec=baseline_eval_time,
                        timestamp=timestamp,
                    ))

        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Run compression methods
        for method in methods:
            method_config = config["method_configs"].get(method, {})

            for ratio in ratios:
                print(f"\n[{method}] Target ratio: {ratio}")

                # Load fresh model for each compression run
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
                )
                model.to(device).eval()

                workspace = output_dir / f"{model_name.replace('/', '_')}/{method}_{ratio}"
                workspace.mkdir(parents=True, exist_ok=True)
                ctx = PipelineContext(config={}, workspace_dir=workspace)
                ctx.state.model = model
                ctx.state.tokenizer = tokenizer

                try:
                    # Compress
                    compression_time, comp_info = compress_model(
                        model, method, ratio, method_config, ctx
                    )
                    print(f"  Compression: {compression_time:.2f}s, ratio: {comp_info['actual_ratio']:.4f}")

                    # Evaluate
                    eval_time, task_metrics = evaluate_model(
                        model, tokenizer, tasks, device, limit
                    )
                    print(f"  Evaluation: {eval_time:.2f}s")

                    # Record results
                    for task, metrics in task_metrics.items():
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                results.append(BenchmarkResult(
                                    model=model_name,
                                    method=method,
                                    compression_ratio=ratio,
                                    task=task,
                                    metric_name=metric_name,
                                    metric_value=float(metric_value),
                                    original_params=comp_info["original_params"],
                                    compressed_params=comp_info["compressed_params"],
                                    param_ratio=comp_info["actual_ratio"],
                                    compression_time_sec=compression_time,
                                    evaluation_time_sec=eval_time,
                                    timestamp=timestamp,
                                ))

                    # Save compressed model
                    save_dir = workspace / "compressed"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    manifest = {
                        "type": f"benchmark_{method}",
                        "base_model": model_name,
                        "method": method,
                        "target_ratio": ratio,
                        "actual_ratio": comp_info["actual_ratio"],
                    }
                    save_compressed_to_safetensors(model, manifest, save_dir)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Record failure
                    for task in tasks:
                        results.append(BenchmarkResult(
                            model=model_name,
                            method=method,
                            compression_ratio=ratio,
                            task=task,
                            metric_name="error",
                            metric_value=0.0,
                            original_params=0,
                            compressed_params=0,
                            param_ratio=0.0,
                            compression_time_sec=0.0,
                            evaluation_time_sec=0.0,
                            timestamp=timestamp,
                        ))

                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def save_results(results: List[BenchmarkResult], output_dir: Path):
    """Save results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # Save to JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Generate summary tables
    summary_path = output_dir / "benchmark_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Group by model
        models = sorted(set(r.model for r in results))
        for model in models:
            f.write(f"## {model}\n\n")

            # Create table for accuracy metrics
            f.write("### Accuracy Results\n\n")
            f.write("| Method | Ratio | Task | Accuracy |\n")
            f.write("|--------|-------|------|----------|\n")

            model_results = [r for r in results if r.model == model and r.metric_name == "accuracy"]
            for r in sorted(model_results, key=lambda x: (x.method, x.compression_ratio, x.task)):
                f.write(f"| {r.method} | {r.compression_ratio:.1f} | {r.task} | {r.metric_value:.4f} |\n")

            f.write("\n")

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Reproducibility Runs")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--methods", default=None, help="Comma-separated methods")
    parser.add_argument("--ratios", default=None, help="Comma-separated ratios")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks")
    parser.add_argument("--output-dir", default="logs/benchmark_reproducibility", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation samples")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_default_config()

    models = args.models.split(",") if args.models else config["models"]
    methods = args.methods.split(",") if args.methods else config["methods"]
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else config["compression_ratios"]
    tasks = args.tasks.split(",") if args.tasks else config["tasks"]
    output_dir = Path(args.output_dir).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Benchmark Reproducibility Runs")
    print(f"==============================")
    print(f"Models: {models}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Tasks: {tasks}")
    print(f"Output: {output_dir}")

    results = run_benchmark(
        models=models,
        methods=methods,
        ratios=ratios,
        tasks=tasks,
        output_dir=output_dir,
        limit=args.limit,
        hf_token=hf_token,
    )

    save_results(results, output_dir)

    print(f"\nCompleted {len(results)} benchmark entries.")


if __name__ == "__main__":
    main()
