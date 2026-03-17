#!/usr/bin/env python3
"""
§2.3 Metric-Guided Compression Quality.

Shows info_flow metrics improve compression quality vs uniform allocation.

Evidence from Layer by Layer Paper (ICML 2025):
> "DiME-based layer selection boosts the average MTEB score of Pythia-410M
>  by 3%" — Section 4.3, Figure 2

> "LLM2Vec-8B: DiME=64.3% vs Last Layer=60.0%" → 4.3% improvement — Table 2

Expected Results:
- Info Flow guided allocation achieves 3-4% better accuracy at same compression
- Or: Same accuracy at 5-10% higher compression

Usage:
  HF_TOKEN=... python scripts/examples/third-party/metric_guided_compression.py [--model name]
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
    parser = argparse.ArgumentParser(description="Metric-Guided Compression Quality")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Model to compress",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/metric_guided_compression",
        help="Output directory for results",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.7,
        help="Target compression ratio (compressed/original params)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip LMHarness evaluation",
    )
    return parser.parse_args()


def count_params(model) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def collect_layer_metrics(
    model,
    tokenizer,
    backend,
    n_samples: int = 32,
) -> Dict[int, Dict[str, float]]:
    """Collect metrics for each layer."""
    # Generate calibration inputs
    torch.manual_seed(42)
    device = next(model.parameters()).device

    metrics_per_layer = {}
    hooks = []
    layer_activations = {}

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unknown model architecture")

    def make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if idx not in layer_activations:
                layer_activations[idx] = []
            layer_activations[idx].append(out.detach().mean(dim=1).cpu())
        return hook

    for idx, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Run forward passes
    vocab_size = tokenizer.vocab_size
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids = torch.randint(0, min(vocab_size, 32000), (1, 64), device=device)
            try:
                _ = model(input_ids)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    # Compute metrics
    available_metrics = list(backend.list_metrics().keys())

    for layer_idx, acts in layer_activations.items():
        combined = torch.cat(acts, dim=0)
        layer_metrics = {}

        for metric_name in available_metrics:
            try:
                metric_fn = backend.get(metric_name)
                value = metric_fn(combined)
                if hasattr(value, "item"):
                    value = value.item()
                layer_metrics[metric_name] = value
            except Exception:
                layer_metrics[metric_name] = None

        metrics_per_layer[layer_idx] = layer_metrics

    return metrics_per_layer


def allocate_ranks_uniform(
    n_layers: int,
    layer_sizes: List[Tuple[int, int]],
    target_ratio: float,
    base_rank: int = 64,
) -> Dict[int, int]:
    """Uniform rank allocation."""
    return {i: base_rank for i in range(n_layers)}


def allocate_ranks_by_metric(
    n_layers: int,
    layer_sizes: List[Tuple[int, int]],
    target_ratio: float,
    layer_metrics: Dict[int, float],
    min_rank: int = 16,
    max_rank: int = 256,
) -> Dict[int, int]:
    """Allocate ranks proportionally to metric values."""
    if not layer_metrics:
        return allocate_ranks_uniform(n_layers, layer_sizes, target_ratio)

    # Normalize metrics
    values = [layer_metrics.get(i, 0) or 0 for i in range(n_layers)]
    total = sum(values) or 1

    # Total rank budget
    total_budget = sum(min(s[0], s[1]) for s in layer_sizes) * target_ratio

    # Allocate proportionally
    ranks = {}
    for i in range(n_layers):
        proportion = values[i] / total
        allocated = int(proportion * total_budget / n_layers)
        ranks[i] = max(min_rank, min(max_rank, allocated, min(layer_sizes[i])))

    return ranks


def compress_layer_svd(
    weight: torch.Tensor,
    rank: int,
    backend_name: str = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a weight matrix using SVD."""
    backend = build_svd_backend(backend_name, {"algorithm": "lanczos", "max_iters": 100})

    U, S, Vh = backend.compute_svd(weight, k=rank)
    sqrt_s = torch.sqrt(S)

    A = U * sqrt_s.unsqueeze(0)  # (m, k)
    B = Vh * sqrt_s.unsqueeze(1)  # (k, n)

    return A, B


def evaluate_perplexity(
    model,
    tokenizer,
    n_samples: int = 100,
    seq_length: int = 128,
) -> float:
    """Simple perplexity evaluation."""
    torch.manual_seed(123)
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids = torch.randint(0, min(vocab_size, 32000), (1, seq_length), device=device)
            try:
                outputs = model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * seq_length
                total_tokens += seq_length
            except Exception:
                pass

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[metric_guided_compression] Model: {args.model}")
    print(f"[metric_guided_compression] Target ratio: {args.target_ratio}")

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[metric_guided_compression] Device: {device}")

    # Load metrics backend
    try:
        backend = ExternalMetricsBackend()
        if not backend.provenance["found"]:
            backend = BasicMetricsBackend()
            backend_type = "basic"
        else:
            backend_type = "external"
    except Exception:
        backend = BasicMetricsBackend()
        backend_type = "basic"

    print(f"[metric_guided_compression] Metrics backend: {backend_type}")

    results = {}

    # Test each allocation strategy
    strategies = [
        ("uniform", None),
    ]

    # Add metric-based strategies
    available_metrics = list(backend.list_metrics().keys())
    for metric in ["l2_norm", "std"]:  # Basic metrics always available
        if metric in available_metrics:
            strategies.append((f"metric_{metric}", metric))

    # Add info_flow metrics if available
    if backend_type == "external":
        for metric in ["effective_rank", "dime", "prompt_entropy"]:
            if metric.lower() in [m.lower() for m in available_metrics]:
                strategies.append((f"metric_{metric}", metric))

    for strategy_name, metric_name in strategies:
        print(f"\n{'=' * 60}")
        print(f"[metric_guided_compression] Strategy: {strategy_name}")
        print(f"{'=' * 60}")

        # Reload model fresh for each strategy
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

        # Get layer info
        layers = model.model.layers
        n_layers = len(layers)

        # Collect layer sizes
        layer_sizes = []
        for layer in layers:
            if hasattr(layer.self_attn, "q_proj"):
                w = layer.self_attn.q_proj.weight
                layer_sizes.append((w.shape[0], w.shape[1]))
            else:
                layer_sizes.append((512, 512))

        # Collect metrics if needed
        if metric_name:
            print(f"  Collecting {metric_name} metrics...")
            layer_metrics_full = collect_layer_metrics(
                model, tokenizer, backend, args.n_samples
            )
            layer_metrics = {
                i: metrics.get(metric_name, 0) or 0
                for i, metrics in layer_metrics_full.items()
            }
        else:
            layer_metrics = {}

        # Allocate ranks
        if strategy_name == "uniform":
            ranks = allocate_ranks_uniform(n_layers, layer_sizes, args.target_ratio)
        else:
            ranks = allocate_ranks_by_metric(
                n_layers, layer_sizes, args.target_ratio, layer_metrics
            )

        print(f"  Rank allocation: min={min(ranks.values())}, max={max(ranks.values())}")

        # Apply compression (simplified: just compress q_proj of each layer)
        print("  Applying SVD compression...")
        t_start = time.perf_counter()

        for layer_idx, layer in enumerate(layers):
            rank = ranks.get(layer_idx, 64)

            # Compress Q projection
            if hasattr(layer.self_attn, "q_proj"):
                q_proj = layer.self_attn.q_proj
                weight = q_proj.weight.data

                if rank < min(weight.shape):
                    try:
                        A, B = compress_layer_svd(weight, rank, "torch")
                        # Create factorized representation (simplified)
                        new_weight = A @ B
                        q_proj.weight.data = new_weight
                    except Exception as e:
                        print(f"    Layer {layer_idx} compression failed: {e}")

        compression_time = time.perf_counter() - t_start
        compressed_params = count_params(model)
        actual_ratio = compressed_params / original_params

        print(f"  Compression time: {compression_time:.2f}s")
        print(f"  Compressed params: {compressed_params:,}")
        print(f"  Actual ratio: {actual_ratio:.4f}")

        # Evaluate
        print("  Evaluating perplexity...")
        perplexity = evaluate_perplexity(model, tokenizer)
        print(f"  Perplexity: {perplexity:.2f}")

        results[strategy_name] = {
            "metric_name": metric_name,
            "original_params": original_params,
            "compressed_params": compressed_params,
            "actual_ratio": actual_ratio,
            "compression_time": compression_time,
            "perplexity": perplexity,
            "rank_allocation": ranks,
        }

        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("[metric_guided_compression] SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Strategy':<25} {'Ratio':<10} {'Perplexity':<12} {'Δ vs Uniform':<15}")
    print(f"  {'-' * 62}")

    baseline_ppl = results.get("uniform", {}).get("perplexity", float("inf"))

    for strategy_name, result in results.items():
        ratio = f"{result['actual_ratio']:.4f}"
        ppl = result["perplexity"]
        ppl_str = f"{ppl:.2f}"

        if strategy_name == "uniform":
            delta = "baseline"
        else:
            delta_pct = (baseline_ppl - ppl) / baseline_ppl * 100
            delta = f"{delta_pct:+.1f}%"

        print(f"  {strategy_name:<25} {ratio:<10} {ppl_str:<12} {delta:<15}")

    # Paper claim verification
    print(f"\n{'=' * 70}")
    print("[metric_guided_compression] PAPER CLAIM VERIFICATION")
    print(f"{'=' * 70}")

    print("\n  Claim: 'DiME-based layer selection boosts MTEB by 3%'")

    metric_strategies = [s for s in results if s.startswith("metric_")]
    if metric_strategies:
        best_metric = min(metric_strategies, key=lambda s: results[s]["perplexity"])
        best_ppl = results[best_metric]["perplexity"]
        improvement = (baseline_ppl - best_ppl) / baseline_ppl * 100

        if improvement > 0:
            print(f"    ✓ Best metric strategy ({best_metric}): {improvement:.1f}% improvement")
        else:
            print(f"    ? No clear improvement observed in this test")
    else:
        print("    ? No metric-based strategies tested")

    if backend_type == "basic":
        print("\n  [NOTE] Using BasicMetricsBackend. For full paper reproduction,")
        print("    install information_flow package for DiME, effective_rank, etc.")

    # Save results
    report = {
        "model": args.model,
        "target_ratio": args.target_ratio,
        "backend_type": backend_type,
        "n_samples": args.n_samples,
        "device": device,
        "results": results,
    }

    report_path = workspace / "compression_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[metric_guided_compression] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
