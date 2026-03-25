#!/usr/bin/env python3
"""
§2.1 External Metrics Loading Verification.

Verifies that ExternalMetricsBackend correctly loads information_flow metrics.

Evidence from Layer by Layer Paper (ICML 2025):
> "We propose a unified framework of representation quality metrics based on
>  information theory, geometry, and invariance to input perturbations" — Abstract

Expected Metrics (7 total):
- Information-Theoretic: prompt_entropy, dataset_entropy, effective_rank
- Geometric: curvature
- Invariance: infonce, lidar, dime

Usage:
  python scripts/examples/third-party/info_flow_metrics_loading.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldcrest.plugins.analysis.metric_utils import ExternalMetricsBackend, BasicMetricsBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External Metrics Loading Verification")
    parser.add_argument(
        "--workspace",
        default="logs/third-party/info_flow_metrics",
        help="Output directory for results",
    )
    parser.add_argument(
        "--module-path",
        default="information_flow.experiments.utils.metrics.metric_functions",
        help="Module path for external metrics",
    )
    return parser.parse_args()


def test_basic_backend() -> Dict[str, Any]:
    """Test BasicMetricsBackend as baseline."""
    print("\n[info_flow_metrics_loading] Testing BasicMetricsBackend")
    print("=" * 60)

    backend = BasicMetricsBackend()
    metrics = backend.list_metrics()

    result = {
        "backend": "BasicMetricsBackend",
        "success": True,
        "metrics_count": len(metrics),
        "metrics_available": list(metrics.keys()),
    }

    print(f"  Metrics available: {len(metrics)}")
    for name in sorted(metrics.keys()):
        print(f"    - {name}")

    # Test each metric with a sample tensor
    import torch
    sample = torch.randn(100, 100)

    print("\n  Testing metric computation:")
    for name, fn in metrics.items():
        try:
            value = fn(sample)
            print(f"    {name}: {value:.4f}")
            result[f"test_{name}"] = True
        except Exception as e:
            print(f"    {name}: FAILED ({e})")
            result[f"test_{name}"] = False
            result["success"] = False

    return result


def test_external_backend(module_path: str) -> Dict[str, Any]:
    """Test ExternalMetricsBackend with information_flow metrics."""
    print(f"\n[info_flow_metrics_loading] Testing ExternalMetricsBackend")
    print(f"  Module path: {module_path}")
    print("=" * 60)

    backend = ExternalMetricsBackend(module_path=module_path)
    provenance = backend.provenance

    result = {
        "backend": "ExternalMetricsBackend",
        "module_path": module_path,
        "provenance": provenance,
    }

    if not provenance["found"]:
        print("  [WARNING] External metrics module not found!")
        print("  Checking alternative loading methods...")

        # Try environment variable
        import os
        env_module = os.environ.get("INFO_FLOW_METRICS_MODULE")
        if env_module:
            print(f"  Found INFO_FLOW_METRICS_MODULE={env_module}")

        env_file = os.environ.get("INFO_FLOW_METRICS_FILE")
        if env_file:
            print(f"  Found INFO_FLOW_METRICS_FILE={env_file}")

        result["success"] = False
        result["error"] = "Module not found. Install information_flow package or set environment variables."
        return result

    print(f"  Module found: {provenance['found']}")
    print(f"  Load method: {provenance['load_method']}")
    print(f"  Source path: {provenance['module_path']}")

    metrics = backend.list_metrics()
    result["metrics_count"] = len(metrics)
    result["metrics_available"] = list(metrics.keys())

    print(f"\n  Metrics available: {len(metrics)}")
    for name in sorted(metrics.keys()):
        print(f"    - {name}")

    # Expected metrics from paper
    expected_metrics = [
        "prompt_entropy",
        "dataset_entropy",
        "effective_rank",
        "curvature",
        "infonce",
        "lidar",
        "dime",
    ]

    print("\n  Checking expected metrics from paper:")
    missing = []
    for expected in expected_metrics:
        # Check case-insensitive
        found = any(expected.lower() == m.lower() for m in metrics.keys())
        status = "✓" if found else "✗"
        print(f"    {status} {expected}")
        if not found:
            missing.append(expected)

    result["expected_metrics"] = expected_metrics
    result["missing_metrics"] = missing
    result["all_expected_found"] = len(missing) == 0

    # Test metric computation with sample tensor
    import torch
    sample = torch.randn(100, 100)

    print("\n  Testing metric computation:")
    computation_results = {}
    for name, fn in list(metrics.items())[:5]:  # Test first 5
        try:
            value = fn(sample)
            if hasattr(value, "item"):
                value = value.item()
            elif hasattr(value, "__float__"):
                value = float(value)
            print(f"    {name}: {value}")
            computation_results[name] = {"success": True, "value": value}
        except Exception as e:
            print(f"    {name}: FAILED ({e})")
            computation_results[name] = {"success": False, "error": str(e)}

    result["computation_tests"] = computation_results
    result["success"] = provenance["found"] and result["all_expected_found"]

    return result


def compare_backends(basic_result: Dict, external_result: Dict) -> Dict[str, Any]:
    """Compare BasicMetricsBackend and ExternalMetricsBackend."""
    print("\n[info_flow_metrics_loading] Backend Comparison")
    print("=" * 60)

    comparison = {
        "basic_metrics_count": basic_result["metrics_count"],
        "external_metrics_count": external_result.get("metrics_count", 0),
    }

    print(f"  BasicMetricsBackend:    {basic_result['metrics_count']} metrics")
    print(f"  ExternalMetricsBackend: {external_result.get('metrics_count', 0)} metrics")

    # Unique metrics in each
    basic_set = set(basic_result["metrics_available"])
    external_set = set(external_result.get("metrics_available", []))

    comparison["basic_only"] = list(basic_set - external_set)
    comparison["external_only"] = list(external_set - basic_set)
    comparison["common"] = list(basic_set & external_set)

    print(f"\n  Metrics only in Basic: {comparison['basic_only']}")
    print(f"  Metrics only in External: {comparison['external_only']}")
    print(f"  Common metrics: {comparison['common']}")

    # Feature comparison
    print("\n  Feature Comparison:")
    print(f"  {'Feature':<30} {'Basic':<10} {'External':<10}")
    print(f"  {'-' * 50}")

    features = [
        ("Information-theoretic", False, "prompt_entropy" in external_result.get("metrics_available", [])),
        ("Geometric (curvature)", False, "curvature" in external_result.get("metrics_available", [])),
        ("Invariance (infonce, dime)", False, "infonce" in external_result.get("metrics_available", [])),
        ("Paper-validated", False, external_result.get("all_expected_found", False)),
    ]

    for name, basic_has, external_has in features:
        basic_str = "✓" if basic_has else "✗"
        external_str = "✓" if external_has else "✗"
        print(f"  {name:<30} {basic_str:<10} {external_str:<10}")

    comparison["features"] = [
        {"name": name, "basic": basic_has, "external": external_has}
        for name, basic_has, external_has in features
    ]

    return comparison


def main():
    args = _parse_args()

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)

    print("[info_flow_metrics_loading] External Metrics Loading Verification")
    print("=" * 70)

    # Test BasicMetricsBackend
    basic_result = test_basic_backend()

    # Test ExternalMetricsBackend
    external_result = test_external_backend(args.module_path)

    # Compare backends
    comparison = compare_backends(basic_result, external_result)

    # Summary
    print(f"\n{'=' * 70}")
    print("[info_flow_metrics_loading] SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  BasicMetricsBackend: {'SUCCESS' if basic_result['success'] else 'FAILED'}")
    print(f"  ExternalMetricsBackend: {'SUCCESS' if external_result.get('success', False) else 'NOT AVAILABLE'}")

    if external_result.get("success", False):
        print("\n  Paper Claim Verification:")
        print("  > 'We propose a unified framework of representation quality metrics'")
        print(f"     ✓ {external_result['metrics_count']} metrics loaded from information_flow")
        if external_result["all_expected_found"]:
            print("     ✓ All 7 expected metrics from paper found")
        else:
            print(f"     ✗ Missing metrics: {external_result['missing_metrics']}")
    else:
        print("\n  [INFO] To enable ExternalMetricsBackend:")
        print("    1. Install information_flow package: pip install information_flow")
        print("    2. Or clone: https://github.com/OFSkean/information_flow")
        print("    3. Or set: INFO_FLOW_METRICS_MODULE=<module.path>")

    # Save results
    report = {
        "basic_backend": basic_result,
        "external_backend": external_result,
        "comparison": comparison,
    }

    report_path = workspace / "metrics_loading_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[info_flow_metrics_loading] Results saved to: {report_path}")


if __name__ == "__main__":
    main()
