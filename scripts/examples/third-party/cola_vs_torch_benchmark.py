#!/usr/bin/env python3
"""
§1.1 CoLA vs Torch SVD Backend Comparison.

Compares runtime, memory, and accuracy of CoLA vs PyTorch SVD backends
on large LLM weight matrices.

Evidence from CoLA Paper (NeurIPS 2023):
> "CoLA's iterative algorithms benefit most from hardware accelerators as
>  the main bottleneck are the MVMs executed at each iteration, which can
>  easily be parallelized on GPUs" — Section 4, Figure 3

Expected Results:
- Runtime ratio (CoLA GPU / PyTorch CPU) < 0.5 for matrices > 4096×4096
- Memory reduction ~30% with CoLA iterative methods
- Frobenius reconstruction error within 1% of baseline

Usage:
  HF_TOKEN=... python scripts/examples/third-party/cola_vs_torch_benchmark.py [--model name]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.plugins.compression.svd_backend import build_svd_backend, ColaSVDBackend, TorchSVDBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoLA vs Torch SVD Backend Benchmark")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Model to extract weight matrices from",
    )
    parser.add_argument(
        "--ranks",
        default="32,64,128",
        help="Comma-separated list of SVD ranks to test",
    )
    parser.add_argument(
        "--layers",
        default="0,4,8",
        help="Comma-separated list of layer indices to test",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/cola_vs_torch",
        help="Output directory for results",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=5,
        help="Number of timed iterations",
    )
    return parser.parse_args()


def get_peak_memory_mb() -> float:
    """Get peak GPU memory in MB if CUDA available, else 0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_memory_stats() -> None:
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_svd_backend(
    backend_name: str,
    matrix: torch.Tensor,
    k: int,
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> Dict:
    """Benchmark a single SVD backend on a matrix."""
    config = {}
    if backend_name == "cola":
        config = {"algorithm": "lanczos", "max_iters": 100}

    backend = build_svd_backend(backend_name, config)

    # Warmup iterations
    for _ in range(warmup_iters):
        reset_memory_stats()
        _ = backend.compute_svd(matrix, k=k)

    # Timed iterations
    times: List[float] = []
    peak_mems: List[float] = []
    frob_errors: List[float] = []

    for _ in range(bench_iters):
        reset_memory_stats()
        start = time.perf_counter()
        U, S, Vh = backend.compute_svd(matrix, k=k)
        elapsed = time.perf_counter() - start
        peak_mem = get_peak_memory_mb()

        # Reconstruction error
        recon = U @ torch.diag(S) @ Vh
        frob_error = torch.linalg.norm(matrix - recon, ord="fro").item()
        rel_error = frob_error / torch.linalg.norm(matrix, ord="fro").item()

        times.append(elapsed)
        peak_mems.append(peak_mem)
        frob_errors.append(rel_error)

    return {
        "backend": backend_name,
        "rank": k,
        "matrix_shape": list(matrix.shape),
        "time_mean": sum(times) / len(times),
        "time_std": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "time_min": min(times),
        "peak_memory_mb": max(peak_mems),
        "frob_error_rel_mean": sum(frob_errors) / len(frob_errors),
    }


def extract_weight_matrices(model, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
    """Extract weight matrices from specified layers."""
    matrices = {}

    for idx in layer_indices:
        try:
            layer = model.model.layers[idx]
        except (AttributeError, IndexError):
            print(f"  [WARNING] Layer {idx} not found, skipping")
            continue

        # Extract projection matrices
        projections = [
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
        ]

        # Add MLP projections if available
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                projections.append(("gate_proj", mlp.gate_proj))
            if hasattr(mlp, "up_proj"):
                projections.append(("up_proj", mlp.up_proj))
            if hasattr(mlp, "down_proj"):
                projections.append(("down_proj", mlp.down_proj))

        for proj_name, proj in projections:
            name = f"layer{idx}.{proj_name}"
            if hasattr(proj, "weight"):
                matrices[name] = proj.weight.data.clone()

    return matrices


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    ranks = [int(r) for r in args.ranks.split(",")]
    layer_indices = [int(i) for i in args.layers.split(",")]

    print(f"[cola_vs_torch_benchmark] Loading model: {args.model}")
    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        token=token,
    )
    model.to(device).eval()

    print(f"[cola_vs_torch_benchmark] Device: {device}")
    print(f"[cola_vs_torch_benchmark] Extracting weights from layers: {layer_indices}")

    matrices = extract_weight_matrices(model, layer_indices)
    print(f"[cola_vs_torch_benchmark] Extracted {len(matrices)} weight matrices")

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []
    comparisons = []

    for matrix_name, matrix in matrices.items():
        print(f"\n[cola_vs_torch_benchmark] Testing: {matrix_name} {tuple(matrix.shape)}")

        for k in ranks:
            if k >= min(matrix.shape):
                print(f"  Skipping rank {k} (>= min dimension {min(matrix.shape)})")
                continue

            print(f"  Rank {k}:")

            # Benchmark Torch
            torch_result = benchmark_svd_backend(
                "torch", matrix, k, args.warmup_iters, args.bench_iters
            )
            torch_result["matrix_name"] = matrix_name
            results.append(torch_result)
            print(f"    Torch:  {torch_result['time_mean']:.4f}s ± {torch_result['time_std']:.4f}s")

            # Benchmark CoLA
            try:
                cola_result = benchmark_svd_backend(
                    "cola", matrix, k, args.warmup_iters, args.bench_iters
                )
                cola_result["matrix_name"] = matrix_name
                results.append(cola_result)
                print(f"    CoLA:   {cola_result['time_mean']:.4f}s ± {cola_result['time_std']:.4f}s")

                # Compute comparison
                speedup = torch_result["time_mean"] / cola_result["time_mean"]
                mem_ratio = cola_result["peak_memory_mb"] / max(torch_result["peak_memory_mb"], 1e-6)
                error_diff = abs(cola_result["frob_error_rel_mean"] - torch_result["frob_error_rel_mean"])

                comparison = {
                    "matrix_name": matrix_name,
                    "matrix_shape": torch_result["matrix_shape"],
                    "rank": k,
                    "torch_time": torch_result["time_mean"],
                    "cola_time": cola_result["time_mean"],
                    "speedup": speedup,
                    "torch_memory_mb": torch_result["peak_memory_mb"],
                    "cola_memory_mb": cola_result["peak_memory_mb"],
                    "memory_ratio": mem_ratio,
                    "torch_frob_error": torch_result["frob_error_rel_mean"],
                    "cola_frob_error": cola_result["frob_error_rel_mean"],
                    "error_diff": error_diff,
                }
                comparisons.append(comparison)

                print(f"    Speedup: {speedup:.2f}x, Memory ratio: {mem_ratio:.2f}, Error diff: {error_diff:.2e}")

            except ImportError as e:
                print(f"    CoLA:   SKIPPED (not installed: {e})")
                cola_result = {
                    "backend": "cola",
                    "matrix_name": matrix_name,
                    "rank": k,
                    "error": str(e),
                }
                results.append(cola_result)

    # Summary
    print(f"\n{'=' * 70}")
    print("[cola_vs_torch_benchmark] SUMMARY")
    print(f"{'=' * 70}")

    if comparisons:
        avg_speedup = sum(c["speedup"] for c in comparisons) / len(comparisons)
        avg_mem_ratio = sum(c["memory_ratio"] for c in comparisons) / len(comparisons)
        avg_error_diff = sum(c["error_diff"] for c in comparisons) / len(comparisons)

        print(f"  Configurations tested: {len(comparisons)}")
        print(f"  Average speedup (CoLA vs Torch): {avg_speedup:.2f}x")
        print(f"  Average memory ratio: {avg_mem_ratio:.2f}")
        print(f"  Average reconstruction error difference: {avg_error_diff:.2e}")

        print("\n  Per-configuration results:")
        print(f"  {'Matrix':<30} {'Shape':<20} {'Rank':<6} {'Speedup':<10} {'Mem Ratio':<10}")
        print(f"  {'-' * 76}")
        for c in comparisons:
            shape_str = f"{c['matrix_shape'][0]}x{c['matrix_shape'][1]}"
            print(f"  {c['matrix_name']:<30} {shape_str:<20} {c['rank']:<6} {c['speedup']:<10.2f} {c['memory_ratio']:<10.2f}")
    else:
        print("  No valid comparisons (CoLA may not be installed)")

    # Save results
    report = {
        "model": args.model,
        "ranks_tested": ranks,
        "layers_tested": layer_indices,
        "device": device,
        "results": results,
        "comparisons": comparisons,
    }

    report_path = workspace / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[cola_vs_torch_benchmark] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
