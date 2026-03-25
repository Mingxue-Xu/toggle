#!/usr/bin/env python3
"""
§1.3 LOBPCG for Ill-Conditioned Layers.

Shows LOBPCG superiority for problematic layers with high condition number.

Evidence from CoLA Paper (NeurIPS 2023):
> "sk(A) denotes sklearn's LOBPCG with an algebraic multi-graph preconditioner
>  (PyAMG). CoLA(B) denotes their LOBPCG implementation" — Section 5

> "These algorithms often provide significant computational speedups for
>  structured matrices that admit sub-quadratic MVMs" — Section 3.1

Expected Results:
- LOBPCG converges where Lanczos struggles (condition number > 10^6)
- Better handling of clustered eigenvalues in early/late transformer layers

Usage:
  python scripts/examples/third-party/lobpcg_ill_conditioned.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from goldcrest.plugins.compression.svd_backend import build_svd_backend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOBPCG for Ill-Conditioned Matrices")
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Matrix size (n x n)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Target SVD rank",
    )
    parser.add_argument(
        "--condition-numbers",
        default="1e2,1e4,1e6,1e8",
        help="Comma-separated condition numbers to test",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/lobpcg_ill_conditioned",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum iterations for iterative methods",
    )
    return parser.parse_args()


def create_ill_conditioned_matrix(
    n: int,
    condition_number: float,
    seed: int = 42,
) -> torch.Tensor:
    """Create a matrix with specified condition number.

    Uses SVD to construct: A = U @ diag(s) @ Vh
    where s ranges from 1 to 1/condition_number.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create random orthogonal matrices via QR
    Q1, _ = torch.linalg.qr(torch.randn(n, n))
    Q2, _ = torch.linalg.qr(torch.randn(n, n))

    # Create singular values with specified condition number
    # s_max / s_min = condition_number
    s_max = 1.0
    s_min = 1.0 / condition_number
    s = torch.logspace(np.log10(s_max), np.log10(s_min), n)

    # Construct matrix
    A = Q1 @ torch.diag(s) @ Q2.T

    return A


def compute_condition_number(matrix: torch.Tensor) -> float:
    """Compute condition number of a matrix."""
    s = torch.linalg.svdvals(matrix)
    return (s[0] / s[-1]).item()


def benchmark_algorithm(
    matrix: torch.Tensor,
    k: int,
    method: str,
    algorithm: str,
    max_iters: int,
) -> Dict:
    """Benchmark an SVD algorithm on an ill-conditioned matrix."""
    config = {"algorithm": algorithm, "max_iters": max_iters}
    backend = build_svd_backend(method, config)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    converged = True
    error_msg = None

    try:
        start = time.perf_counter()
        U, S, Vh = backend.compute_svd(matrix, k=k)
        elapsed = time.perf_counter() - start

        # Compute reconstruction error
        recon = U @ torch.diag(S) @ Vh
        frob_error = torch.linalg.norm(matrix - recon, ord="fro").item()
        rel_error = frob_error / torch.linalg.norm(matrix, ord="fro").item()

        # Compare with ground truth (torch dense SVD)
        torch_backend = build_svd_backend("torch", {})
        U_gt, S_gt, Vh_gt = torch_backend.compute_svd(matrix, k=k)

        # Subspace angle between computed and ground truth
        # Using Frobenius norm of difference between projected matrices
        proj_computed = U @ U.T
        proj_gt = U_gt @ U_gt.T
        subspace_error = torch.linalg.norm(proj_computed - proj_gt, ord="fro").item()

        # Singular value accuracy
        sv_error = torch.linalg.norm(S - S_gt).item() / torch.linalg.norm(S_gt).item()

    except Exception as e:
        elapsed = float("nan")
        rel_error = float("nan")
        subspace_error = float("nan")
        sv_error = float("nan")
        converged = False
        error_msg = str(e)

    return {
        "method": method,
        "algorithm": algorithm,
        "converged": converged,
        "time": elapsed,
        "rel_reconstruction_error": rel_error,
        "subspace_error": subspace_error,
        "singular_value_error": sv_error,
        "error_msg": error_msg,
    }


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    condition_numbers = [float(c) for c in args.condition_numbers.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lobpcg_ill_conditioned] Device: {device}")
    print(f"[lobpcg_ill_conditioned] Matrix size: {args.size} x {args.size}")
    print(f"[lobpcg_ill_conditioned] Target rank: {args.rank}")
    print(f"[lobpcg_ill_conditioned] Condition numbers: {condition_numbers}")

    results = []
    comparisons = []

    # Test Torch as baseline
    algorithms_to_test = [
        ("torch", "dense"),
    ]

    # Add CoLA algorithms if available
    try:
        build_svd_backend("cola", {"algorithm": "lanczos"})
        algorithms_to_test.append(("cola", "lanczos"))
        algorithms_to_test.append(("cola", "lobpcg"))
        cola_available = True
    except ImportError:
        print("[lobpcg_ill_conditioned] CoLA not available, testing Torch only")
        cola_available = False

    for cond in condition_numbers:
        print(f"\n{'=' * 60}")
        print(f"[lobpcg_ill_conditioned] Condition number: {cond:.0e}")
        print(f"{'=' * 60}")

        # Create ill-conditioned matrix
        matrix = create_ill_conditioned_matrix(args.size, cond)
        actual_cond = compute_condition_number(matrix)
        print(f"  Actual condition number: {actual_cond:.2e}")

        matrix = matrix.to(device)

        comparison = {
            "condition_number": cond,
            "actual_condition_number": actual_cond,
        }

        for method, algorithm in algorithms_to_test:
            print(f"\n  Testing: {method}/{algorithm}")

            result = benchmark_algorithm(
                matrix, args.rank, method, algorithm, args.max_iters
            )
            result["condition_number"] = cond
            result["actual_condition_number"] = actual_cond
            result["matrix_size"] = args.size
            result["rank"] = args.rank
            results.append(result)

            if result["converged"]:
                print(f"    Time: {result['time']:.4f}s")
                print(f"    Reconstruction error: {result['rel_reconstruction_error']:.2e}")
                print(f"    Subspace error: {result['subspace_error']:.2e}")
                print(f"    Singular value error: {result['singular_value_error']:.2e}")
                comparison[f"{method}_{algorithm}_time"] = result["time"]
                comparison[f"{method}_{algorithm}_error"] = result["rel_reconstruction_error"]
                comparison[f"{method}_{algorithm}_converged"] = True
            else:
                print(f"    FAILED: {result['error_msg']}")
                comparison[f"{method}_{algorithm}_converged"] = False

        comparisons.append(comparison)

        # Clean up
        del matrix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("[lobpcg_ill_conditioned] SUMMARY")
    print(f"{'=' * 70}")

    if cola_available:
        print("\n  Convergence Analysis by Condition Number:")
        print(f"  {'Cond #':<12} {'Torch':<10} {'Lanczos':<10} {'LOBPCG':<10}")
        print(f"  {'-' * 42}")

        for c in comparisons:
            cond_str = f"{c['condition_number']:.0e}"
            torch_status = "✓" if c.get("torch_dense_converged", False) else "✗"
            lanczos_status = "✓" if c.get("cola_lanczos_converged", False) else "✗"
            lobpcg_status = "✓" if c.get("cola_lobpcg_converged", False) else "✗"
            print(f"  {cond_str:<12} {torch_status:<10} {lanczos_status:<10} {lobpcg_status:<10}")

        print("\n  Error Analysis (where converged):")
        print(f"  {'Cond #':<12} {'Lanczos Err':<15} {'LOBPCG Err':<15} {'Winner':<10}")
        print(f"  {'-' * 52}")

        for c in comparisons:
            cond_str = f"{c['condition_number']:.0e}"
            lanczos_err = c.get("cola_lanczos_error", float("nan"))
            lobpcg_err = c.get("cola_lobpcg_error", float("nan"))

            if np.isnan(lanczos_err) and np.isnan(lobpcg_err):
                winner = "N/A"
            elif np.isnan(lanczos_err):
                winner = "LOBPCG"
            elif np.isnan(lobpcg_err):
                winner = "Lanczos"
            elif lobpcg_err < lanczos_err:
                winner = "LOBPCG"
            else:
                winner = "Lanczos"

            lanczos_str = f"{lanczos_err:.2e}" if not np.isnan(lanczos_err) else "FAIL"
            lobpcg_str = f"{lobpcg_err:.2e}" if not np.isnan(lobpcg_err) else "FAIL"

            print(f"  {cond_str:<12} {lanczos_str:<15} {lobpcg_str:<15} {winner:<10}")

        # Paper claim verification
        print("\n  Paper Claim Verification:")
        print("  > 'LOBPCG converges where Lanczos struggles (condition number > 10^6)'")

        high_cond_cases = [c for c in comparisons if c["condition_number"] >= 1e6]
        if high_cond_cases:
            lobpcg_wins = sum(
                1 for c in high_cond_cases
                if c.get("cola_lobpcg_converged", False) and not c.get("cola_lanczos_converged", True)
            )
            lobpcg_better = sum(
                1 for c in high_cond_cases
                if c.get("cola_lobpcg_converged", False) and c.get("cola_lanczos_converged", False)
                and c.get("cola_lobpcg_error", float("inf")) < c.get("cola_lanczos_error", float("inf"))
            )

            if lobpcg_wins > 0:
                print(f"    ✓ LOBPCG converged in {lobpcg_wins} case(s) where Lanczos failed")
            if lobpcg_better > 0:
                print(f"    ✓ LOBPCG had lower error in {lobpcg_better} high-condition case(s)")
            if lobpcg_wins == 0 and lobpcg_better == 0:
                print("    ? No clear LOBPCG advantage observed in this test")
        else:
            print("    ? No high condition number cases tested")
    else:
        print("  CoLA not available - cannot compare Lanczos vs LOBPCG")

    # Save results
    report = {
        "matrix_size": args.size,
        "rank": args.rank,
        "max_iters": args.max_iters,
        "condition_numbers_tested": condition_numbers,
        "device": device,
        "cola_available": cola_available,
        "results": results,
        "comparisons": comparisons,
    }

    report_path = workspace / "lobpcg_benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[lobpcg_ill_conditioned] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
