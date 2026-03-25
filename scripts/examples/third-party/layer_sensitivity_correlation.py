#!/usr/bin/env python3
"""
§2.4 Layer Sensitivity Correlation.

Verifies info_flow metrics correlate with compression sensitivity.

Evidence from Layer by Layer Paper (ICML 2025):
> "All metrics exhibit strong relationships with downstream performance.
>  Among them, curvature, DiME, and InfoNCE stand out with particularly
>  high correlations" — Section 4.2

Expected Results:
- DiME: High correlation with compression sensitivity
- Curvature: High correlation with compression sensitivity
- InfoNCE: High correlation with compression sensitivity
- Basic metrics (L2 norm): Lower correlation

Usage:
  HF_TOKEN=... python scripts/examples/third-party/layer_sensitivity_correlation.py [--model name]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.plugins.compression.svd_backend import build_svd_backend
from goldcrest.plugins.analysis.metric_utils import ExternalMetricsBackend, BasicMetricsBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer Sensitivity Correlation")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Model to analyze",
    )
    parser.add_argument(
        "--workspace",
        default="logs/third-party/layer_sensitivity_correlation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--compression-rank",
        type=int,
        default=32,
        help="SVD rank for compression sensitivity test",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Number of calibration samples",
    )
    return parser.parse_args()


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")

    x_arr = np.array(x)
    y_arr = np.array(y)

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sqrt(np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2))

    if denominator < 1e-12:
        return float("nan")

    return numerator / denominator


def collect_layer_metrics_and_weights(
    model,
    tokenizer,
    backend,
    n_samples: int = 32,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, torch.Tensor]]:
    """Collect metrics and weights for each layer."""
    torch.manual_seed(42)
    device = next(model.parameters()).device

    layer_activations = {}
    hooks = []

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
    metrics_per_layer = {}

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

    # Collect weights
    weights_per_layer = {}
    for idx, layer in enumerate(layers):
        if hasattr(layer.self_attn, "q_proj"):
            weights_per_layer[idx] = layer.self_attn.q_proj.weight.data.clone().cpu()

    return metrics_per_layer, weights_per_layer


def measure_compression_sensitivity(
    model,
    tokenizer,
    layer_idx: int,
    rank: int,
    n_eval_samples: int = 50,
) -> float:
    """Measure perplexity change when compressing a single layer."""
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Get baseline perplexity
    def eval_perplexity():
        torch.manual_seed(123)
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for _ in range(n_eval_samples):
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

    baseline_ppl = eval_perplexity()

    # Compress layer
    layers = model.model.layers
    layer = layers[layer_idx]
    q_proj = layer.self_attn.q_proj
    original_weight = q_proj.weight.data.clone()

    try:
        backend = build_svd_backend("torch", {})
        U, S, Vh = backend.compute_svd(original_weight.float(), k=rank)
        compressed_weight = U @ torch.diag(S) @ Vh
        q_proj.weight.data = compressed_weight.to(original_weight.dtype)

        compressed_ppl = eval_perplexity()

        # Restore original weight
        q_proj.weight.data = original_weight

        sensitivity = (compressed_ppl - baseline_ppl) / baseline_ppl
        return sensitivity

    except Exception as e:
        q_proj.weight.data = original_weight
        return float("nan")


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[layer_sensitivity_correlation] Model: {args.model}")
    print(f"[layer_sensitivity_correlation] Compression rank: {args.compression_rank}")

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[layer_sensitivity_correlation] Device: {device}")

    # Load model
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

    # Get number of layers
    n_layers = len(model.model.layers)
    print(f"[layer_sensitivity_correlation] Total layers: {n_layers}")

    # Load metrics backends
    backends = {"basic": BasicMetricsBackend()}

    try:
        ext_backend = ExternalMetricsBackend()
        if ext_backend.provenance["found"]:
            backends["external"] = ext_backend
            print(f"[layer_sensitivity_correlation] External metrics available")
    except Exception:
        pass

    # Collect metrics
    print("\n[layer_sensitivity_correlation] Collecting layer metrics...")
    all_metrics = {}
    for name, backend in backends.items():
        metrics_per_layer, weights = collect_layer_metrics_and_weights(
            model, tokenizer, backend, args.n_samples
        )
        all_metrics[name] = metrics_per_layer

    # Measure compression sensitivity for each layer
    print("\n[layer_sensitivity_correlation] Measuring compression sensitivity...")
    sensitivities = {}

    for layer_idx in range(n_layers):
        print(f"  Layer {layer_idx}/{n_layers}...", end=" ", flush=True)
        sensitivity = measure_compression_sensitivity(
            model, tokenizer, layer_idx, args.compression_rank
        )
        sensitivities[layer_idx] = sensitivity
        print(f"sensitivity={sensitivity:.4f}")

    # Compute correlations
    print("\n[layer_sensitivity_correlation] Computing correlations...")

    sensitivity_values = [sensitivities[i] for i in range(n_layers)]
    correlations = {}

    for backend_name, metrics_per_layer in all_metrics.items():
        backend_corrs = {}

        # Get available metrics
        sample_metrics = metrics_per_layer.get(0, {})

        for metric_name in sample_metrics.keys():
            metric_values = []
            valid_sensitivities = []

            for i in range(n_layers):
                val = metrics_per_layer.get(i, {}).get(metric_name)
                sens = sensitivities[i]

                if val is not None and not np.isnan(val) and not np.isnan(sens):
                    metric_values.append(val)
                    valid_sensitivities.append(sens)

            if len(metric_values) >= 3:
                corr = pearson_correlation(metric_values, valid_sensitivities)
                backend_corrs[metric_name] = {
                    "correlation": corr,
                    "n_samples": len(metric_values),
                }

        correlations[backend_name] = backend_corrs

    # Summary
    print(f"\n{'=' * 70}")
    print("[layer_sensitivity_correlation] CORRELATION RESULTS")
    print(f"{'=' * 70}")

    for backend_name, backend_corrs in correlations.items():
        print(f"\n  Backend: {backend_name}")
        print(f"  {'Metric':<25} {'Correlation':<15} {'|r|':<10}")
        print(f"  {'-' * 50}")

        # Sort by absolute correlation
        sorted_metrics = sorted(
            backend_corrs.items(),
            key=lambda x: abs(x[1]["correlation"]) if not np.isnan(x[1]["correlation"]) else 0,
            reverse=True,
        )

        for metric_name, result in sorted_metrics:
            corr = result["correlation"]
            if np.isnan(corr):
                corr_str = "N/A"
                abs_str = "N/A"
            else:
                corr_str = f"{corr:+.4f}"
                abs_str = f"{abs(corr):.4f}"

            print(f"  {metric_name:<25} {corr_str:<15} {abs_str:<10}")

    # Paper claim verification
    print(f"\n{'=' * 70}")
    print("[layer_sensitivity_correlation] PAPER CLAIM VERIFICATION")
    print(f"{'=' * 70}")

    print("\n  Claim: 'curvature, DiME, and InfoNCE stand out with high correlations'")

    if "external" in correlations:
        ext_corrs = correlations["external"]
        target_metrics = ["curvature", "dime", "infonce"]

        high_corr_found = []
        for metric in target_metrics:
            # Case-insensitive lookup
            for m, result in ext_corrs.items():
                if metric.lower() in m.lower():
                    corr = result["correlation"]
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        high_corr_found.append((m, corr))

        if high_corr_found:
            print(f"    ✓ High correlations found:")
            for m, c in high_corr_found:
                print(f"      {m}: r={c:+.4f}")
        else:
            print("    ? Target metrics not showing high correlation in this test")
    else:
        print("    ? External metrics not available")
        print("      Install information_flow package for curvature, DiME, InfoNCE")

    # Compare basic vs advanced metrics
    if "basic" in correlations:
        basic_corrs = correlations["basic"]
        max_basic = max(
            (abs(r["correlation"]) for r in basic_corrs.values() if not np.isnan(r["correlation"])),
            default=0,
        )

        print(f"\n  Basic metrics max |r|: {max_basic:.4f}")

        if "external" in correlations:
            ext_corrs = correlations["external"]
            max_ext = max(
                (abs(r["correlation"]) for r in ext_corrs.values() if not np.isnan(r["correlation"])),
                default=0,
            )
            print(f"  External metrics max |r|: {max_ext:.4f}")

            if max_ext > max_basic:
                print("    ✓ External (info_flow) metrics show stronger correlations")
            else:
                print("    ? Basic metrics performed comparably in this test")

    # Save results
    report = {
        "model": args.model,
        "compression_rank": args.compression_rank,
        "n_layers": n_layers,
        "n_samples": args.n_samples,
        "device": device,
        "sensitivities": sensitivities,
        "correlations": correlations,
    }

    report_path = workspace / "correlation_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[layer_sensitivity_correlation] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
