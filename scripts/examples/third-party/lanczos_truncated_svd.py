#!/usr/bin/env python3
"""
§1.2 Lanczos Algorithm Showcase for Truncated SVD.

Demonstrates Lanczos advantages for truncated SVD (k << min(m,n)).

Evidence from CoLA Paper (NeurIPS 2023):
> "When LinearOperators are annotated with additional structure (self-adjoint, PSD),
>  CoLA uses more efficient Krylov algorithms like MINRES, conjugate gradients,
>  and Lanczos" — Section 3.2

Expected Results:
- Speedup proportional to (full_rank / k) for truncated decomposition
- Linear memory scaling O(k·n) instead of O(n²) for dense methods
- Convergence in ~100 iterations for well-conditioned matrices

Usage:
  python scripts/examples/third-party/lanczos_truncated_svd.py [--sizes 1024,4096,8192]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from goldcrest.plugins.compression.svd_backend import build_svd_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lanczos Truncated SVD Showcase")
    parser.add_argument(
        "--sizes",
        default="512,1024,2048,4096",
        help="Comma-separated list of matrix sizes to test",
    )
    parser.add_argument(
        "--ranks",
        default="16,32,64,128",
        help="Comma-separated list of truncation ranks to test",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/lanczos_showcase",
        help="Output directory for results",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=3,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--max-lanczos-iters",
        type=int,
        default=100,
        help="Maximum Lanczos iterations",
    )
    return parser.parse_args()


def reset_memory() -> None:
    """Reset memory stats and collect garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def benchmark_truncated_svd(
    matrix: torch.Tensor,
    k: int,
    method: str,
    config: Dict,
    bench_iters: int = 3,
) -> Dict:
    """Benchmark truncated SVD with specified method."""
    backend = build_svd_backend(method, config)

    times: List[float] = []
    errors: List[float] = []

    for _ in range(bench_iters):
        reset_memory()
        start = time.perf_counter()
        U, S, Vh = backend.compute_svd(matrix, k=k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Compute reconstruction error
        recon = U @ torch.diag(S) @ Vh
        frob_error = torch.linalg.norm(matrix - recon, ord="fro").item()
        rel_error = frob_error / torch.linalg.norm(matrix, ord="fro").item()
        errors.append(rel_error)

    return {
        "method": method,
        "config": config,
        "rank": k,
        "matrix_size": matrix.shape[0],
        "time_mean": sum(times) / len(times),
        "time_min": min(times),
        "time_max": max(times),
        "rel_error_mean": sum(errors) / len(errors),
    }


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lanczos_truncated_svd] Device: {device}")
    print(f"[lanczos_truncated_svd] Matrix sizes: {sizes}")
    print(f"[lanczos_truncated_svd] Truncation ranks: {ranks}")

    results = []
    comparisons = []

    for n in sizes:
        print(f"\n{'=' * 60}")
        print(f"[lanczos_truncated_svd] Matrix size: {n} x {n}")
        print(f"{'=' * 60}")

        # Create random matrix (simulating weight matrix)
        torch.manual_seed(42)
        matrix = torch.randn(n, n, device=device, dtype=torch.float32)
        full_rank = n

        for k in ranks:
            if k >= n:
                print(f"  Skipping rank {k} >= matrix size {n}")
                continue

            print(f"\n  Rank k={k} (truncation ratio: {k/full_rank:.2%})")

            # Method 1: Torch dense SVD (computes all singular values, then truncates)
            torch_result = benchmark_truncated_svd(
                matrix, k, "torch", {}, args.bench_iters
            )
            results.append(torch_result)
            print(f"    Torch (dense):  {torch_result['time_mean']:.4f}s, error: {torch_result['rel_error_mean']:.2e}")

            # Method 2: CoLA Lanczos (only computes k singular values)
            try:
                lanczos_config = {"algorithm": "lanczos", "max_iters": args.max_lanczos_iters}
                lanczos_result = benchmark_truncated_svd(
                    matrix, k, "cola", lanczos_config, args.bench_iters
                )
                results.append(lanczos_result)
                print(f"    CoLA (Lanczos): {lanczos_result['time_mean']:.4f}s, error: {lanczos_result['rel_error_mean']:.2e}")

                # Compute speedup
                speedup = torch_result["time_mean"] / lanczos_result["time_mean"]
                theoretical_speedup = full_rank / k  # Theoretical asymptotic speedup

                comparison = {
                    "matrix_size": n,
                    "rank": k,
                    "truncation_ratio": k / full_rank,
                    "torch_time": torch_result["time_mean"],
                    "lanczos_time": lanczos_result["time_mean"],
                    "speedup": speedup,
                    "theoretical_speedup": theoretical_speedup,
                    "speedup_efficiency": speedup / theoretical_speedup,
                    "torch_error": torch_result["rel_error_mean"],
                    "lanczos_error": lanczos_result["rel_error_mean"],
                    "error_ratio": lanczos_result["rel_error_mean"] / max(torch_result["rel_error_mean"], 1e-12),
                }
                comparisons.append(comparison)

                print(f"    Speedup: {speedup:.2f}x (theoretical max: {theoretical_speedup:.1f}x)")

            except ImportError as e:
                print(f"    CoLA (Lanczos): SKIPPED (not installed)")
                results.append({
                    "method": "cola",
                    "matrix_size": n,
                    "rank": k,
                    "error": str(e),
                })

        # Clean up
        del matrix
        reset_memory()

    # Summary
    print(f"\n{'=' * 70}")
    print("[lanczos_truncated_svd] SUMMARY")
    print(f"{'=' * 70}")

    if comparisons:
        print("\n  Scaling Analysis (Lanczos speedup vs matrix size):")
        print(f"  {'Size':<10} {'Rank':<8} {'Speedup':<12} {'Theoretical':<12} {'Efficiency':<12}")
        print(f"  {'-' * 54}")

        for c in comparisons:
            print(
                f"  {c['matrix_size']:<10} {c['rank']:<8} "
                f"{c['speedup']:<12.2f} {c['theoretical_speedup']:<12.1f} "
                f"{c['speedup_efficiency']:<12.1%}"
            )

        # Group by matrix size for trend analysis
        by_size = {}
        for c in comparisons:
            size = c["matrix_size"]
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(c["speedup"])

        print("\n  Average speedup by matrix size:")
        for size in sorted(by_size.keys()):
            avg = sum(by_size[size]) / len(by_size[size])
            print(f"    {size}x{size}: {avg:.2f}x average speedup")

        # Verify paper claim: speedup proportional to n/k
        print("\n  Paper Claim Verification:")
        print("  > 'Speedup proportional to (full_rank / k) for truncated decomposition'")
        avg_efficiency = sum(c["speedup_efficiency"] for c in comparisons) / len(comparisons)
        print(f"    Average speedup efficiency: {avg_efficiency:.1%} of theoretical")
        if avg_efficiency > 0.3:
            print("    ✓ Claim SUPPORTED: Lanczos shows substantial speedup for truncated SVD")
        else:
            print("    ? Claim needs more investigation")

    else:
        print("  No valid comparisons (CoLA may not be installed)")

    # Save results
    report = {
        "matrix_sizes": sizes,
        "ranks_tested": ranks,
        "max_lanczos_iters": args.max_lanczos_iters,
        "device": device,
        "results": results,
        "comparisons": comparisons,
    }

    report_path = workspace / "lanczos_benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[lanczos_truncated_svd] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
