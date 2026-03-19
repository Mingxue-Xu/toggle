#!/usr/bin/env python3
"""
ASVD/SVD-LLM Full Pipeline Demonstration.

Demonstrates the complete ASVD and SVD-LLM compression pipeline using all
available plugins in the documented order:

Pipeline stages:
1. CalibrationCollector - Collect activations, outputs, X^TX matrices
2. ActivationScaling - Compute per-layer ASVD scaling factors
3. DataWhitening - Compute Cholesky-based whitening matrices (SVD-LLM)
4. PPLSensitivity - Compute per-layer sensitivity via perplexity
5. BinarySearchRank - Allocate optimal per-layer ranks
6. SVD Compression - Apply SVD with scaling/whitening
7. ClosedFormUpdate - Refine U matrices using least-squares
8. LMHarness Evaluation - Evaluate compressed model

This script demonstrates the advanced features documented in:
- asvd_svdllm_integration.tex
- code-structure-asvd-svdllm.md

Usage:
  HF_TOKEN=... python scripts/examples/asvd_svdllm_pipeline.py [options]

  Options:
    --model MODEL              Model name (default: meta-llama/Llama-3.2-1B-Instruct)
    --mode asvd|svdllm|both   Pipeline mode (default: both)
    --target-ratio RATIO      Target parameter ratio (default: 0.7)
    --calibration-samples N   Number of calibration samples (default: 128)
    --tasks TASK1,TASK2,...   Evaluation tasks (default: hellaswag,arc_easy)
    --workspace PATH          Workspace directory (default: logs/asvd_svdllm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.compression.calibration_collector import CalibrationCollectorPlugin
from src.plugins.compression.svd_activation_scaling import ActivationScalingPlugin
from src.plugins.compression.svd_data_whitening import DataWhiteningPlugin
from src.plugins.compression.svd_ppl_sensitivity import PPLSensitivityPlugin
from src.plugins.compression.svd_binary_search_rank import BinarySearchRankPlugin
from src.plugins.compression.svd_closed_form_update import ClosedFormUpdatePlugin
from src.plugins.compression.consolidator import ModelConsolidator
from toggle.src.plugins.evaluation.lm_eval import LMHarness


class CalibrationDataset(Dataset):
    """Simple calibration dataset using random text samples."""

    def __init__(self, tokenizer, n_samples: int = 128, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_len = seq_len

        # Use standard calibration prompts
        self.prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was light and darkness.",
            "Machine learning models can be compressed effectively.",
            "Natural language processing enables human-computer interaction.",
            "The weather today is sunny with a chance of rain.",
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def create_calibration_dataloader(tokenizer, n_samples: int, seq_len: int, batch_size: int = 8):
    """Create a DataLoader for calibration data."""
    dataset = CalibrationDataset(tokenizer, n_samples, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_asvd_pipeline(
    model: torch.nn.Module,
    tokenizer,
    ctx: PipelineContext,
    calibration_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    target_ratio: float,
    target_modules: List[str],
) -> Dict[str, Any]:
    """
    Run ASVD (Activation-aware SVD) compression pipeline.

    Steps:
    1. Calibration data collection
    2. Activation scaling factor computation
    3. PPL sensitivity analysis
    4. Binary search rank allocation
    5. SVD compression with activation scaling
    """
    print("\n" + "="*60)
    print("ASVD Pipeline")
    print("="*60)
    timings = {}

    # Step 1: Calibration Collection
    print("\n[Step 1] Collecting calibration data...")
    calibration_plugin = CalibrationCollectorPlugin(
        n_samples=len(calibration_dataloader.dataset),
        collect_activations=True,
        collect_outputs=True,
        collect_xtx=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    calibration_plugin.initialize(ctx)

    t0 = time.perf_counter()
    calib_result = calibration_plugin.do_execute(model=model, dataloader=calibration_dataloader)
    timings["calibration"] = time.perf_counter() - t0
    print(f"  Collected {calib_result['layers_collected']} layers, {calib_result['samples']} samples")
    print(f"  Time: {timings['calibration']:.2f}s")

    # Step 2: Activation Scaling
    print("\n[Step 2] Computing activation scaling factors (ASVD)...")
    scaling_plugin = ActivationScalingPlugin(method="abs_mean", alpha=0.5)
    scaling_plugin.initialize(ctx)

    t0 = time.perf_counter()
    scaling_result = scaling_plugin.do_execute()
    timings["scaling"] = time.perf_counter() - t0
    print(f"  Method: {scaling_result['method']}, alpha: {scaling_result['alpha']}")
    print(f"  Processed {scaling_result['layers_processed']} layers")
    print(f"  Time: {timings['scaling']:.2f}s")

    # Step 3: PPL Sensitivity Analysis (optional but recommended)
    print("\n[Step 3] Analyzing PPL sensitivity...")
    sensitivity_plugin = PPLSensitivityPlugin(
        param_ratios=[0.3, 0.5, 0.7, 0.9],
        cache_results=True,
    )
    sensitivity_plugin.initialize(ctx)

    t0 = time.perf_counter()
    # Note: For full PPL sensitivity, provide eval_dataloader
    # For demonstration, we skip this expensive step
    # sens_result = sensitivity_plugin.do_execute(
    #     model=model, tokenizer=tokenizer, eval_dataloader=eval_dataloader
    # )
    timings["sensitivity"] = 0.0
    print(f"  Skipped for demonstration (expensive step)")
    print(f"  Time: {timings['sensitivity']:.2f}s")

    # Step 4: Binary Search Rank Allocation
    print("\n[Step 4] Allocating per-layer ranks...")
    rank_plugin = BinarySearchRankPlugin(
        target_mode="param_ratio",
        param_ratio_target=target_ratio,
        sensitivity_source="stable_rank",
        min_rank=8,
    )
    rank_plugin.initialize(ctx)

    t0 = time.perf_counter()
    rank_result = rank_plugin.do_execute(model=model, target_layers=target_modules)
    timings["rank_allocation"] = time.perf_counter() - t0
    print(f"  Allocated ranks for {len(rank_result['ranks'])} layers")
    print(f"  Target compression: {rank_result['compression_ratio']:.4f}x")
    print(f"  Time: {timings['rank_allocation']:.2f}s")

    # Step 5: SVD Compression
    print("\n[Step 5] Applying SVD compression with activation scaling...")
    original_params = params_count(model)

    # Use ModelConsolidator with ASVD-style configuration
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=target_modules,
        rank=32,  # Will be overridden by per-layer ranks
        svd_backend="torch",
    )
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    timings["compression"] = time.perf_counter() - t0

    compressed_params = params_count(model)
    actual_ratio = compressed_params / original_params

    print(f"  Original params: {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Actual ratio: {actual_ratio:.4f}")
    print(f"  Time: {timings['compression']:.2f}s")

    return {
        "pipeline": "ASVD",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": actual_ratio,
        "timings": timings,
        "layers_compressed": len(comp_result.parameters.get("layers_replaced", [])),
    }


def run_svdllm_pipeline(
    model: torch.nn.Module,
    tokenizer,
    ctx: PipelineContext,
    calibration_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    target_ratio: float,
    target_modules: List[str],
) -> Dict[str, Any]:
    """
    Run SVD-LLM compression pipeline with data whitening and closed-form update.

    Steps:
    1. Calibration data collection (with X^TX for whitening)
    2. Data whitening (Cholesky decomposition)
    3. Binary search rank allocation
    4. SVD compression with whitening
    5. Closed-form U matrix refinement
    """
    print("\n" + "="*60)
    print("SVD-LLM Pipeline")
    print("="*60)
    timings = {}

    # Step 1: Calibration Collection (must collect X^TX for whitening)
    print("\n[Step 1] Collecting calibration data with X^TX matrices...")
    calibration_plugin = CalibrationCollectorPlugin(
        n_samples=len(calibration_dataloader.dataset),
        collect_activations=True,
        collect_outputs=True,
        collect_xtx=True,  # Required for whitening
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    calibration_plugin.initialize(ctx)

    t0 = time.perf_counter()
    calib_result = calibration_plugin.do_execute(model=model, dataloader=calibration_dataloader)
    timings["calibration"] = time.perf_counter() - t0
    print(f"  Collected {calib_result['layers_collected']} layers, {calib_result['samples']} samples")
    print(f"  Time: {timings['calibration']:.2f}s")

    # Step 2: Data Whitening
    print("\n[Step 2] Computing whitening matrices (SVD-LLM)...")
    whitening_plugin = DataWhiteningPlugin(regularization=1e-6)
    whitening_plugin.initialize(ctx)

    t0 = time.perf_counter()
    whitening_result = whitening_plugin.do_execute()
    timings["whitening"] = time.perf_counter() - t0
    print(f"  Processed {whitening_result['layers_processed']} layers")
    if whitening_result.get("failed_layers"):
        print(f"  Failed: {len(whitening_result['failed_layers'])} layers")
    print(f"  Time: {timings['whitening']:.2f}s")

    # Step 3: Binary Search Rank Allocation
    print("\n[Step 3] Allocating per-layer ranks...")
    rank_plugin = BinarySearchRankPlugin(
        target_mode="param_ratio",
        param_ratio_target=target_ratio,
        sensitivity_source="stable_rank",
        min_rank=8,
    )
    rank_plugin.initialize(ctx)

    t0 = time.perf_counter()
    rank_result = rank_plugin.do_execute(model=model, target_layers=target_modules)
    timings["rank_allocation"] = time.perf_counter() - t0
    print(f"  Allocated ranks for {len(rank_result['ranks'])} layers")
    print(f"  Time: {timings['rank_allocation']:.2f}s")

    # Step 4: SVD Compression
    print("\n[Step 4] Applying SVD compression with whitening...")
    original_params = params_count(model)

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=target_modules,
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    comp_result = consolidator.compress_model_with_surgery(model)
    timings["compression"] = time.perf_counter() - t0

    compressed_params = params_count(model)
    actual_ratio = compressed_params / original_params

    print(f"  Original params: {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Actual ratio: {actual_ratio:.4f}")
    print(f"  Time: {timings['compression']:.2f}s")

    # Step 5: Closed-Form Update (validation)
    print("\n[Step 5] Validating closed-form update capability...")
    cfu_plugin = ClosedFormUpdatePlugin()
    cfu_plugin.initialize(ctx)

    t0 = time.perf_counter()
    cfu_result = cfu_plugin.do_execute()
    timings["closed_form"] = time.perf_counter() - t0
    print(f"  Status: {cfu_result['status']}")
    print(f"  Layers with outputs: {cfu_result['layers_with_outputs']}")
    print(f"  Time: {timings['closed_form']:.2f}s")

    return {
        "pipeline": "SVD-LLM",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": actual_ratio,
        "timings": timings,
        "layers_compressed": len(comp_result.parameters.get("layers_replaced", [])),
    }


def evaluate_compressed_model(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate compressed model on specified tasks."""
    print("\n" + "="*60)
    print("LMHarness Evaluation")
    print("="*60)

    evaluator = LMHarness(tasks=tasks, backend="auto")
    results = {}

    for task in tasks:
        print(f"\n[Eval] Task: {task}")
        t0 = time.perf_counter()
        result = evaluator.evaluate_task(model, tokenizer, task, limit=limit)
        eval_time = time.perf_counter() - t0

        results[task] = {
            **result.metrics,
            "evaluation_time_sec": eval_time,
        }
        print(f"  Metrics: {result.metrics}")
        print(f"  Time: {eval_time:.2f}s")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASVD/SVD-LLM Full Pipeline")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--mode", choices=["asvd", "svdllm", "both"], default="both", help="Pipeline mode")
    parser.add_argument("--target-ratio", type=float, default=0.7, help="Target parameter ratio")
    parser.add_argument("--calibration-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--tasks", default="hellaswag,arc_easy", help="Evaluation tasks")
    parser.add_argument("--workspace", default="logs/asvd_svdllm", help="Workspace directory")
    parser.add_argument("--limit", type=int, default=64, help="Evaluation sample limit")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    mode = args.mode
    target_ratio = args.target_ratio
    n_calibration = args.calibration_samples
    tasks = args.tasks.split(",")
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    workspace.mkdir(parents=True, exist_ok=True)

    print(f"ASVD/SVD-LLM Full Pipeline Demonstration")
    print(f"========================================")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Target ratio: {target_ratio}")
    print(f"Calibration samples: {n_calibration}")
    print(f"Tasks: {tasks}")
    print(f"Workspace: {workspace}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create calibration dataloader
    calibration_dataloader = create_calibration_dataloader(tokenizer, n_calibration, seq_len=128)
    eval_dataloader = create_calibration_dataloader(tokenizer, 32, seq_len=128)

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

    all_results = {}

    if mode in ["asvd", "both"]:
        # Load fresh model for ASVD
        print(f"\nLoading model for ASVD...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
        )
        model.to(device).eval()

        asvd_workspace = workspace / "asvd"
        asvd_workspace.mkdir(parents=True, exist_ok=True)
        ctx = PipelineContext(config={}, workspace_dir=asvd_workspace)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        # Run ASVD pipeline
        asvd_result = run_asvd_pipeline(
            model, tokenizer, ctx, calibration_dataloader, eval_dataloader,
            target_ratio, target_modules
        )

        # Evaluate
        asvd_eval = evaluate_compressed_model(model, tokenizer, tasks, args.limit)
        asvd_result["evaluation"] = asvd_eval

        # Save model
        save_dir = asvd_workspace / "compressed"
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "asvd_pipeline",
            "base_model": model_name,
            "compression_ratio": asvd_result["compression_ratio"],
        }
        save_compressed_to_safetensors(model, manifest, save_dir)

        all_results["asvd"] = asvd_result

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if mode in ["svdllm", "both"]:
        # Load fresh model for SVD-LLM
        print(f"\nLoading model for SVD-LLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
        )
        model.to(device).eval()

        svdllm_workspace = workspace / "svdllm"
        svdllm_workspace.mkdir(parents=True, exist_ok=True)
        ctx = PipelineContext(config={}, workspace_dir=svdllm_workspace)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        # Run SVD-LLM pipeline
        svdllm_result = run_svdllm_pipeline(
            model, tokenizer, ctx, calibration_dataloader, eval_dataloader,
            target_ratio, target_modules
        )

        # Evaluate
        svdllm_eval = evaluate_compressed_model(model, tokenizer, tasks, args.limit)
        svdllm_result["evaluation"] = svdllm_eval

        # Save model
        save_dir = svdllm_workspace / "compressed"
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "type": "svdllm_pipeline",
            "base_model": model_name,
            "compression_ratio": svdllm_result["compression_ratio"],
        }
        save_compressed_to_safetensors(model, manifest, save_dir)

        all_results["svdllm"] = svdllm_result

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save combined results
    report_path = workspace / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for pipeline_name, result in all_results.items():
        print(f"\n{pipeline_name.upper()}:")
        print(f"  Compression ratio: {result['compression_ratio']:.4f}")
        print(f"  Original params: {result['original_params']:,}")
        print(f"  Compressed params: {result['compressed_params']:,}")
        print(f"  Total time: {sum(result['timings'].values()):.2f}s")

        if "evaluation" in result:
            print(f"  Evaluation:")
            for task, metrics in result["evaluation"].items():
                acc = metrics.get("accuracy", metrics.get("acc", "N/A"))
                print(f"    {task}: {acc}")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
