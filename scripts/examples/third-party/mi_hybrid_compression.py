#!/usr/bin/env python3
"""
§3.2 Mutual Information Hybrid Compression.

Uses MI metric to guide compression method selection (SVD vs Tucker).

Evidence from papers:
- CoLA (NeurIPS 2023): "Significant speedups for compositional structure (Kronecker, block diagonal)"
    → supports Tucker for structured layers
- Layer by Layer (ICML 2025): "All metrics exhibit strong relationships with downstream performance"
    → MI guides method selection

Expected Results:
- MI Hybrid achieves best quality/speed tradeoff
- High MI layers → Tucker (exploits structure)
- Low MI layers → SVD with CoLA (faster)

Usage:
  HF_TOKEN=... python scripts/examples/third-party/mi_hybrid_compression.py [--model name]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.plugins.compression.svd_backend import build_svd_backend
from src.plugins.analysis.metric_utils import ExternalMetricsBackend, BasicMetricsBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI Hybrid Compression")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Model to compress",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/mi_hybrid_compression",
        help="Output directory",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.7,
        help="Target compression ratio",
    )
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=0.5,
        help="MI threshold for method selection (normalized)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Number of calibration samples",
    )
    return parser.parse_args()


def count_params(model) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def collect_layer_statistics(
    model,
    tokenizer,
    n_samples: int = 32,
) -> Dict[int, Dict[str, float]]:
    """Collect statistics that can indicate mode interaction (proxy for MI)."""
    print("\n[MI Hybrid] Collecting layer statistics...")

    torch.manual_seed(42)
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    layer_stats = {}
    hooks = []

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unknown model architecture")

    layer_activations = {}

    def make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if idx not in layer_activations:
                layer_activations[idx] = []
            # Store full activation for analysis
            layer_activations[idx].append(out.detach().cpu())
        return hook

    for idx, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids = torch.randint(0, min(vocab_size, 32000), (1, 32), device=device)
            try:
                _ = model(input_ids)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    # Compute statistics as MI proxy
    for layer_idx, acts in layer_activations.items():
        combined = torch.cat(acts, dim=0)  # (samples, seq_len, hidden)

        # Flatten to (samples * seq_len, hidden)
        flat = combined.view(-1, combined.shape[-1]).float()

        # Compute SVD to analyze mode structure
        try:
            # Covariance matrix eigenvalues as proxy for mode structure
            cov = flat.T @ flat / flat.shape[0]
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.clamp(min=1e-10)

            # Entropy of normalized eigenvalues (higher = more modes interacting)
            normalized = eigenvalues / eigenvalues.sum()
            entropy = -torch.sum(normalized * torch.log(normalized + 1e-10)).item()

            # Effective rank
            effective_rank = torch.exp(torch.tensor(entropy)).item()

            # Spectral ratio (top eigenvalue / sum) - lower = more modes
            spectral_ratio = (eigenvalues[-1] / eigenvalues.sum()).item()

            # MI proxy: combine entropy and inverse spectral ratio
            mi_proxy = entropy * (1 - spectral_ratio)

        except Exception:
            entropy = 1.0
            effective_rank = 1.0
            spectral_ratio = 1.0
            mi_proxy = 0.5

        layer_stats[layer_idx] = {
            "entropy": entropy,
            "effective_rank": effective_rank,
            "spectral_ratio": spectral_ratio,
            "mi_proxy": mi_proxy,
        }

    print(f"  Computed statistics for {len(layer_stats)} layers")
    return layer_stats


def normalize_mi_values(layer_stats: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """Normalize MI proxy values to [0, 1]."""
    mi_values = [s["mi_proxy"] for s in layer_stats.values()]
    mi_min = min(mi_values)
    mi_max = max(mi_values)

    if mi_max - mi_min < 1e-10:
        return {i: 0.5 for i in layer_stats.keys()}

    return {
        i: (layer_stats[i]["mi_proxy"] - mi_min) / (mi_max - mi_min)
        for i in layer_stats.keys()
    }


def compress_layer_svd(
    weight: torch.Tensor,
    rank: int,
    backend_name: str = "torch",
) -> Tuple[torch.Tensor, float]:
    """Compress using SVD."""
    config = {}
    if backend_name == "cola":
        config = {"algorithm": "lanczos", "max_iters": 100}

    backend = build_svd_backend(backend_name, config)

    t_start = time.perf_counter()
    U, S, Vh = backend.compute_svd(weight, k=rank)
    new_weight = U @ torch.diag(S) @ Vh
    elapsed = time.perf_counter() - t_start

    return new_weight, elapsed


def compress_layer_tucker(
    weight: torch.Tensor,
    ranks: Tuple[int, int],
) -> Tuple[torch.Tensor, float]:
    """Compress using Tucker decomposition (simplified 2D version)."""
    # For 2D weights, Tucker is essentially double-sided SVD
    # W ≈ U @ core @ V^T where U is m×r1, core is r1×r2, V is n×r2

    t_start = time.perf_counter()

    # Left SVD
    U_full, S1, _ = torch.linalg.svd(weight, full_matrices=False)
    U = U_full[:, :ranks[0]]

    # Project and right SVD
    projected = U.T @ weight
    _, S2, Vh_full = torch.linalg.svd(projected, full_matrices=False)
    Vh = Vh_full[:ranks[1], :]

    # Core
    core = U.T @ weight @ Vh.T

    # Reconstruct
    new_weight = U @ core @ Vh

    elapsed = time.perf_counter() - t_start

    return new_weight, elapsed


def evaluate_perplexity(
    model,
    tokenizer,
    n_samples: int = 50,
) -> float:
    """Evaluate perplexity."""
    torch.manual_seed(123)
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids = torch.randint(0, min(vocab_size, 32000), (1, 64), device=device)
            try:
                outputs = model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * 64
                total_tokens += 64
            except Exception:
                pass

    if total_tokens == 0:
        return float("inf")

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def compress_model(
    model,
    strategy: str,
    layer_stats: Dict[int, Dict[str, float]],
    target_ratio: float,
    mi_threshold: float,
    svd_backend: str = "torch",
) -> Dict[str, Any]:
    """Compress model with specified strategy."""
    layers = model.model.layers
    n_layers = len(layers)

    normalized_mi = normalize_mi_values(layer_stats)

    results = {
        "strategy": strategy,
        "layers": [],
        "total_time": 0,
        "svd_count": 0,
        "tucker_count": 0,
    }

    base_rank = 64

    for layer_idx, layer in enumerate(layers):
        # Only compress q_proj for simplicity
        if not hasattr(layer.self_attn, "q_proj"):
            continue

        q_proj = layer.self_attn.q_proj
        weight = q_proj.weight.data.float()
        original_shape = weight.shape

        mi_value = normalized_mi.get(layer_idx, 0.5)

        # Determine method and rank
        if strategy == "uniform_svd":
            method = "svd"
            rank = base_rank
        elif strategy == "uniform_tucker":
            method = "tucker"
            rank = base_rank
        elif strategy == "mi_hybrid":
            if mi_value > mi_threshold:
                method = "tucker"
            else:
                method = "svd"
            rank = base_rank
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply compression
        if rank >= min(original_shape):
            continue

        try:
            if method == "svd":
                new_weight, elapsed = compress_layer_svd(weight, rank, svd_backend)
                results["svd_count"] += 1
            else:  # tucker
                tucker_ranks = (min(rank, original_shape[0]), min(rank, original_shape[1]))
                new_weight, elapsed = compress_layer_tucker(weight, tucker_ranks)
                results["tucker_count"] += 1

            q_proj.weight.data = new_weight.to(q_proj.weight.dtype)
            results["total_time"] += elapsed

            results["layers"].append({
                "layer_idx": layer_idx,
                "method": method,
                "mi_value": mi_value,
                "time": elapsed,
            })

        except Exception as e:
            print(f"  [WARNING] Layer {layer_idx} failed: {e}")

    return results


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[mi_hybrid_compression] MI-Guided Hybrid Compression")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  MI threshold: {args.mi_threshold}")
    print(f"  Target ratio: {args.target_ratio}")

    # Strategies to compare
    strategies = [
        ("uniform_svd", "torch"),
        ("uniform_tucker", "torch"),
        ("mi_hybrid", "torch"),
    ]

    # Try adding CoLA
    try:
        build_svd_backend("cola", {"algorithm": "lanczos"})
        strategies.append(("mi_hybrid_cola", "cola"))
        cola_available = True
        print("  CoLA SVD: available")
    except ImportError:
        cola_available = False
        print("  CoLA SVD: not available")

    all_results = {}

    for strategy_name, svd_backend in strategies:
        print(f"\n{'=' * 60}")
        print(f"[mi_hybrid_compression] Strategy: {strategy_name}")
        print(f"{'=' * 60}")

        # Reload model fresh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            token=token,
        )
        model.to(device).eval()

        original_params = count_params(model)
        print(f"  Original params: {original_params:,}")

        # Collect statistics
        layer_stats = collect_layer_statistics(model, tokenizer, args.n_samples)

        # Compress
        actual_strategy = strategy_name.replace("_cola", "")
        compression_results = compress_model(
            model, actual_strategy, layer_stats,
            args.target_ratio, args.mi_threshold, svd_backend
        )

        compressed_params = count_params(model)
        actual_ratio = compressed_params / original_params

        print(f"  Compressed params: {compressed_params:,}")
        print(f"  Actual ratio: {actual_ratio:.4f}")
        print(f"  Compression time: {compression_results['total_time']:.2f}s")
        print(f"  Methods used: SVD={compression_results['svd_count']}, Tucker={compression_results['tucker_count']}")

        # Evaluate
        print("  Evaluating perplexity...")
        perplexity = evaluate_perplexity(model, tokenizer)
        print(f"  Perplexity: {perplexity:.2f}")

        all_results[strategy_name] = {
            "svd_backend": svd_backend,
            "original_params": original_params,
            "compressed_params": compressed_params,
            "actual_ratio": actual_ratio,
            "compression_time": compression_results["total_time"],
            "svd_count": compression_results["svd_count"],
            "tucker_count": compression_results["tucker_count"],
            "perplexity": perplexity,
            "layer_stats": layer_stats,
        }

        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("[mi_hybrid_compression] SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Strategy':<20} {'Time':<10} {'PPL':<10} {'Methods':<20}")
    print(f"  {'-' * 60}")

    for strategy_name, result in all_results.items():
        time_str = f"{result['compression_time']:.2f}s"
        ppl_str = f"{result['perplexity']:.2f}"
        methods_str = f"SVD:{result['svd_count']}, Tucker:{result['tucker_count']}"
        print(f"  {strategy_name:<20} {time_str:<10} {ppl_str:<10} {methods_str:<20}")

    # Identify best
    best_ppl = min(r["perplexity"] for r in all_results.values())
    best_strategy = [s for s, r in all_results.items() if r["perplexity"] == best_ppl][0]

    print(f"\n  Best perplexity: {best_strategy} ({best_ppl:.2f})")

    # Paper claim verification
    print(f"\n{'=' * 70}")
    print("[mi_hybrid_compression] PAPER CLAIM VERIFICATION")
    print(f"{'=' * 70}")

    print("\n  Claim: 'MI Hybrid achieves best quality/speed tradeoff'")

    uniform_svd_ppl = all_results.get("uniform_svd", {}).get("perplexity", float("inf"))
    uniform_tucker_ppl = all_results.get("uniform_tucker", {}).get("perplexity", float("inf"))
    hybrid_ppl = all_results.get("mi_hybrid", {}).get("perplexity", float("inf"))

    uniform_svd_time = all_results.get("uniform_svd", {}).get("compression_time", float("inf"))
    uniform_tucker_time = all_results.get("uniform_tucker", {}).get("compression_time", float("inf"))
    hybrid_time = all_results.get("mi_hybrid", {}).get("compression_time", float("inf"))

    # Check if hybrid is competitive
    if hybrid_ppl <= min(uniform_svd_ppl, uniform_tucker_ppl) * 1.05:
        print(f"    ✓ MI Hybrid perplexity ({hybrid_ppl:.2f}) competitive with best uniform ({min(uniform_svd_ppl, uniform_tucker_ppl):.2f})")
    else:
        print(f"    ? MI Hybrid perplexity ({hybrid_ppl:.2f}) not better than uniform methods")

    if hybrid_time < uniform_tucker_time:
        print(f"    ✓ MI Hybrid ({hybrid_time:.2f}s) faster than uniform Tucker ({uniform_tucker_time:.2f}s)")
    else:
        print(f"    ? MI Hybrid ({hybrid_time:.2f}s) not faster than Tucker ({uniform_tucker_time:.2f}s)")

    if cola_available:
        hybrid_cola_ppl = all_results.get("mi_hybrid_cola", {}).get("perplexity", float("inf"))
        hybrid_cola_time = all_results.get("mi_hybrid_cola", {}).get("compression_time", float("inf"))
        print(f"\n    With CoLA: PPL={hybrid_cola_ppl:.2f}, Time={hybrid_cola_time:.2f}s")

    # Save report
    report = {
        "model": args.model,
        "device": device,
        "mi_threshold": args.mi_threshold,
        "target_ratio": args.target_ratio,
        "cola_available": cola_available,
        "results": all_results,
    }

    report_path = workspace / "mi_hybrid_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[mi_hybrid_compression] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
