#!/usr/bin/env python3
"""
Memory Profiling Runs for JMLR MLOSS Paper.

Uses MemoryInferenceProfiler to measure GPU memory reduction metrics
for compressed models. Production-ready evidence per Pearl pattern.

Demonstrates:
- GPU memory reduction metrics
- Inference speed comparison
- Memory vs accuracy tradeoff analysis

Usage:
  HF_TOKEN=... python scripts/examples/memory_profiling.py [options]

  Options:
    --model MODEL             Model name (default: Qwen/Qwen2.5-0.5B)
    --methods METHOD1,...     Compression methods (default: svd,tucker)
    --ratios RATIO1,...       Compression ratios (default: 0.3,0.5,0.7)
    --batch-size N            Batch size for profiling (default: 1)
    --seq-len N               Sequence length for profiling (default: 128)
    --warmup N                Warmup iterations (default: 3)
    --repeats N               Profiling repeats (default: 10)
    --workspace PATH          Workspace directory (default: logs/memory_profiling)
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.memory_profiler import MemoryProfiler
from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.plugins.compression.consolidator import ModelConsolidator
from goldcrest.plugins.analysis.memory_inference import MemoryInferenceProfiler


@dataclass
class MemoryProfileResult:
    """Single memory profiling result."""
    model: str
    method: str
    compression_ratio: float
    # Memory metrics
    model_memory_mb: float
    inference_peak_gpu_mb: float
    inference_peak_rss_mb: float
    # Parameter metrics
    original_params: int
    compressed_params: int
    param_ratio: float
    # Timing metrics
    avg_inference_time_ms: float
    compression_time_sec: float
    # Derived metrics
    memory_reduction_pct: float
    speedup_ratio: float
    timestamp: str


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def random_ids_inputs(model, batch_size: int, seq_len: int, device: str) -> Dict[str, torch.Tensor]:
    """Generate random token IDs for profiling."""
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 32000)
    input_ids = torch.randint(low=0, high=int(vocab_size), size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def profile_model_memory(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    device: str,
    warmup: int = 3,
    repeats: int = 10,
) -> Dict[str, float]:
    """Profile model memory and inference time."""
    model.eval()

    # Force garbage collection and clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Get initial GPU memory
    initial_gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    # Model memory (parameters + buffers)
    model_gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    model_memory_mb = model_gpu_mb - initial_gpu_mb

    # Prepare inputs
    inputs = random_ids_inputs(model, batch_size, seq_len, device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Reset peak stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Profile inference
    inference_times = []
    peak_gpu_allocated = 0.0
    peak_rss = 0.0

    profiler = MemoryProfiler()

    with profiler.profile_execution("inference_profiling", sync_gpu=True):
        for _ in range(repeats):
            if torch.cuda.is_available():
                start_mem = torch.cuda.memory_allocated()

            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)  # ms

            if torch.cuda.is_available():
                peak_gpu_allocated = max(peak_gpu_allocated, torch.cuda.max_memory_allocated() / 1024 / 1024)

    # Get profile results
    prof = profiler.get_profile("inference_profiling")
    if prof and prof.peak_memory:
        peak_gpu_allocated = max(peak_gpu_allocated, prof.peak_memory.gpu_allocated_mb)
        peak_rss = prof.peak_memory.rss_mb

    avg_inference_time_ms = sum(inference_times) / len(inference_times) if inference_times else 0.0

    return {
        "model_memory_mb": model_memory_mb,
        "inference_peak_gpu_mb": peak_gpu_allocated,
        "inference_peak_rss_mb": peak_rss,
        "avg_inference_time_ms": avg_inference_time_ms,
    }


def compress_model(
    model: torch.nn.Module,
    method: str,
    target_ratio: float,
    ctx: PipelineContext,
) -> Tuple[float, Dict[str, Any]]:
    """Compress model using specified method."""
    original_params = params_count(model)

    # Find representative layer dimensions
    sample_m, sample_n = 512, 512
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    # Estimate rank
    if method == "svd":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    elif method == "tucker":
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.7))
    elif method == "cp":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n)))
    else:  # tensor_train
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.5))

    rank = min(rank, min(sample_m, sample_n))

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    consolidator_kwargs = {
        "compression_method": method,
        "target_modules": target_modules,
    }

    if method == "svd":
        consolidator_kwargs.update({"rank": rank, "svd_backend": "torch"})
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
    }


def run_memory_profiling(
    model_name: str,
    methods: List[str],
    ratios: List[float],
    batch_size: int,
    seq_len: int,
    warmup: int,
    repeats: int,
    workspace: Path,
    hf_token: Optional[str] = None,
) -> List[MemoryProfileResult]:
    """Run memory profiling for all configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().isoformat()
    results: List[MemoryProfileResult] = []

    print(f"\n{'='*60}")
    print(f"Memory Profiling")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load tokenizer (for metadata)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )

    # === BASELINE PROFILING ===
    print(f"\n[Baseline] Loading and profiling...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
    )
    baseline_model.to(device).eval()
    baseline_params = params_count(baseline_model)

    baseline_profile = profile_model_memory(
        baseline_model, batch_size, seq_len, device, warmup, repeats
    )

    print(f"  Model memory: {baseline_profile['model_memory_mb']:.2f} MB")
    print(f"  Inference peak GPU: {baseline_profile['inference_peak_gpu_mb']:.2f} MB")
    print(f"  Avg inference time: {baseline_profile['avg_inference_time_ms']:.2f} ms")

    baseline_result = MemoryProfileResult(
        model=model_name,
        method="baseline",
        compression_ratio=1.0,
        model_memory_mb=baseline_profile["model_memory_mb"],
        inference_peak_gpu_mb=baseline_profile["inference_peak_gpu_mb"],
        inference_peak_rss_mb=baseline_profile["inference_peak_rss_mb"],
        original_params=baseline_params,
        compressed_params=baseline_params,
        param_ratio=1.0,
        avg_inference_time_ms=baseline_profile["avg_inference_time_ms"],
        compression_time_sec=0.0,
        memory_reduction_pct=0.0,
        speedup_ratio=1.0,
        timestamp=timestamp,
    )
    results.append(baseline_result)

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === COMPRESSED MODEL PROFILING ===
    for method in methods:
        for ratio in ratios:
            print(f"\n[{method.upper()}] Ratio: {ratio}")

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
            )
            model.to(device).eval()

            method_workspace = workspace / f"{method}_{ratio}"
            method_workspace.mkdir(parents=True, exist_ok=True)
            ctx = PipelineContext(config={}, workspace_dir=method_workspace)
            ctx.state.model = model
            ctx.state.tokenizer = tokenizer

            try:
                # Compress
                compression_time, comp_info = compress_model(model, method, ratio, ctx)
                print(f"  Compressed in {compression_time:.2f}s")
                print(f"  Actual ratio: {comp_info['actual_ratio']:.4f}")

                # Profile
                compressed_profile = profile_model_memory(
                    model, batch_size, seq_len, device, warmup, repeats
                )

                # Calculate derived metrics
                memory_reduction_pct = 100.0 * (
                    1.0 - compressed_profile["inference_peak_gpu_mb"] / baseline_result.inference_peak_gpu_mb
                ) if baseline_result.inference_peak_gpu_mb > 0 else 0.0

                speedup_ratio = (
                    baseline_result.avg_inference_time_ms / compressed_profile["avg_inference_time_ms"]
                ) if compressed_profile["avg_inference_time_ms"] > 0 else 1.0

                print(f"  Model memory: {compressed_profile['model_memory_mb']:.2f} MB")
                print(f"  Inference peak GPU: {compressed_profile['inference_peak_gpu_mb']:.2f} MB")
                print(f"  Memory reduction: {memory_reduction_pct:.1f}%")
                print(f"  Avg inference time: {compressed_profile['avg_inference_time_ms']:.2f} ms")
                print(f"  Speedup: {speedup_ratio:.2f}x")

                results.append(MemoryProfileResult(
                    model=model_name,
                    method=method,
                    compression_ratio=ratio,
                    model_memory_mb=compressed_profile["model_memory_mb"],
                    inference_peak_gpu_mb=compressed_profile["inference_peak_gpu_mb"],
                    inference_peak_rss_mb=compressed_profile["inference_peak_rss_mb"],
                    original_params=comp_info["original_params"],
                    compressed_params=comp_info["compressed_params"],
                    param_ratio=comp_info["actual_ratio"],
                    avg_inference_time_ms=compressed_profile["avg_inference_time_ms"],
                    compression_time_sec=compression_time,
                    memory_reduction_pct=memory_reduction_pct,
                    speedup_ratio=speedup_ratio,
                    timestamp=timestamp,
                ))

                # Save compressed model
                save_dir = method_workspace / "compressed"
                save_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "type": f"memory_profile_{method}",
                    "base_model": model_name,
                    "method": method,
                    "compression_ratio": comp_info["actual_ratio"],
                }
                save_compressed_to_safetensors(model, manifest, save_dir)

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(MemoryProfileResult(
                    model=model_name,
                    method=method,
                    compression_ratio=ratio,
                    model_memory_mb=0.0,
                    inference_peak_gpu_mb=0.0,
                    inference_peak_rss_mb=0.0,
                    original_params=0,
                    compressed_params=0,
                    param_ratio=0.0,
                    avg_inference_time_ms=0.0,
                    compression_time_sec=0.0,
                    memory_reduction_pct=0.0,
                    speedup_ratio=0.0,
                    timestamp=timestamp,
                ))

            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def save_results(results: List[MemoryProfileResult], output_dir: Path):
    """Save results to CSV, JSON, and markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "memory_profile_results.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # Save to JSON
    json_path = output_dir / "memory_profile_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Generate markdown summary
    summary_path = output_dir / "memory_profile_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Memory Profiling Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Find baseline
        baseline = next((r for r in results if r.method == "baseline"), None)
        if baseline:
            f.write("## Baseline\n\n")
            f.write(f"- Model: {baseline.model}\n")
            f.write(f"- Parameters: {baseline.original_params:,}\n")
            f.write(f"- Inference Peak GPU: {baseline.inference_peak_gpu_mb:.2f} MB\n")
            f.write(f"- Avg Inference Time: {baseline.avg_inference_time_ms:.2f} ms\n\n")

        # Compression results table
        f.write("## Compression Results\n\n")
        f.write("| Method | Target | Actual | Peak GPU (MB) | Memory Red. | Inference (ms) | Speedup |\n")
        f.write("|--------|--------|--------|---------------|-------------|----------------|--------|\n")

        for r in sorted(results, key=lambda x: (x.method, x.compression_ratio)):
            if r.method == "baseline":
                continue
            f.write(f"| {r.method} | {r.compression_ratio:.2f} | {r.param_ratio:.4f} | ")
            f.write(f"{r.inference_peak_gpu_mb:.1f} | {r.memory_reduction_pct:.1f}% | ")
            f.write(f"{r.avg_inference_time_ms:.2f} | {r.speedup_ratio:.2f}x |\n")

        f.write("\n")

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Profiling for Compressed Models")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--methods", default="svd,tucker", help="Compression methods")
    parser.add_argument("--ratios", default="0.3,0.5,0.7", help="Compression ratios")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Profiling repeats")
    parser.add_argument("--workspace", default="logs/memory_profiling", help="Workspace directory")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    methods = args.methods.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    print("Memory Profiling for Compressed Models")
    print("======================================")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print(f"Workspace: {workspace}")

    results = run_memory_profiling(
        model_name=model_name,
        methods=methods,
        ratios=ratios,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        repeats=args.repeats,
        workspace=workspace,
        hf_token=hf_token,
    )

    save_results(results, workspace)

    print(f"\nCompleted {len(results)} memory profiles.")


if __name__ == "__main__":
    main()
