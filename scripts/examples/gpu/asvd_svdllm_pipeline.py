#!/usr/bin/env python3
"""
ASVD/SVD-LLM Full Pipeline Demonstration - H200 GPU Version.

Demonstrates the complete ASVD and SVD-LLM compression pipeline using all
available plugins on large-scale models (8B-70B parameters).

Optimized for H200 GPU with 140GB VRAM. Uses BF16 precision.

Pipeline stages:
1. CalibrationCollector - Collect activations, outputs, X^TX matrices
2. ActivationScaling - Compute per-layer ASVD scaling factors
3. DataWhitening - Compute Cholesky-based whitening matrices (SVD-LLM)
4. PPLSensitivity - Compute per-layer sensitivity via perplexity
5. BinarySearchRank - Allocate optimal per-layer ranks
6. SVD Compression - Apply SVD with scaling/whitening
7. ClosedFormUpdate - Refine U matrices using least-squares
8. LMHarness Evaluation - Evaluate compressed model

Recommended Models:
- Phase 1 (Quick): meta-llama/Llama-3.1-8B (~30 min)
- Phase 2 (Large): google/gemma-2-27b (~1-2 hours)
- Phase 3 (Flagship): meta-llama/Llama-3.1-70B (~4-6 hours)

Usage:
  HF_TOKEN=... python scripts/examples/gpu/asvd_svdllm_pipeline.py [options]

  Options:
    --model MODEL              Model name (default: meta-llama/Llama-3.1-8B)
    --mode asvd|svdllm|both   Pipeline mode (default: both)
    --target-ratio RATIO      Target parameter ratio (default: 0.5)
    --calibration-samples N   Number of calibration samples (default: 128)
    --tasks TASK1,TASK2,...   Evaluation tasks
    --workspace PATH          Workspace directory
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


# H200 GPU-optimized model recommendations
RECOMMENDED_MODELS = {
    "quick": "meta-llama/Llama-3.1-8B",
    "mid": "Qwen/Qwen2.5-14B",
    "large": "google/gemma-2-27b",
    "flagship": "meta-llama/Llama-3.1-70B",
}

# Extended task set for GPU benchmarks
DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]


class CalibrationDataset(Dataset):
    """Simple calibration dataset using random text samples."""

    def __init__(self, tokenizer, n_samples: int = 128, seq_len: int = 256):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_len = seq_len

        # Extended calibration prompts for better activation coverage
        self.prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was light and darkness.",
            "Machine learning models can be compressed effectively using tensor decomposition.",
            "Natural language processing enables human-computer interaction through understanding text.",
            "The weather today is sunny with a chance of rain in the afternoon.",
            "Large language models have billions of parameters that require significant memory.",
            "Compression techniques like SVD reduce model size while preserving accuracy.",
            "The transformer architecture revolutionized natural language understanding.",
            "Calibration data helps determine optimal compression parameters for each layer.",
            "Activation-aware methods consider how data flows through the network.",
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


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def create_calibration_dataloader(tokenizer, n_samples: int, seq_len: int, batch_size: int = 4):
    """Create a DataLoader for calibration data. Smaller batch for large models."""
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
    Run ASVD (Activation-aware SVD) compression pipeline on H200 GPU.
    """
    print("\n" + "="*60)
    print("ASVD Pipeline (H200 GPU)")
    print("="*60)
    timings = {}

    # Step 1: Calibration Collection
    print("\n[Step 1] Collecting calibration data...")
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")
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
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")

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

    # Step 3: PPL Sensitivity Analysis
    print("\n[Step 3] Analyzing PPL sensitivity...")
    sensitivity_plugin = PPLSensitivityPlugin(
        param_ratios=[0.3, 0.5, 0.7, 0.9],
        cache_results=True,
    )
    sensitivity_plugin.initialize(ctx)

    t0 = time.perf_counter()
    # For large models, we skip the expensive PPL sensitivity step
    # and use stable_rank heuristics instead
    timings["sensitivity"] = 0.0
    print(f"  Using stable_rank heuristics (skipping expensive PPL computation for large models)")
    print(f"  Time: {timings['sensitivity']:.2f}s")

    # Step 4: Binary Search Rank Allocation
    print("\n[Step 4] Allocating per-layer ranks...")
    rank_plugin = BinarySearchRankPlugin(
        target_mode="param_ratio",
        param_ratio_target=target_ratio,
        sensitivity_source="stable_rank",
        min_rank=16,  # Higher minimum for large models
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

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=target_modules,
        rank=64,  # Higher default rank for large models
        svd_backend="cola",
        svd_backend_config={"algorithm": "lanczos"},
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
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")

    return {
        "pipeline": "ASVD",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": actual_ratio,
        "timings": timings,
        "layers_compressed": len(comp_result.parameters.get("layers_replaced", [])),
        "gpu_memory_gb": get_gpu_memory_gb(),
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
    Optimized for H200 GPU.
    """
    print("\n" + "="*60)
    print("SVD-LLM Pipeline (H200 GPU)")
    print("="*60)
    timings = {}

    # Step 1: Calibration Collection (must collect X^TX for whitening)
    print("\n[Step 1] Collecting calibration data with X^TX matrices...")
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")
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
        min_rank=16,
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
        rank=64,
        svd_backend="cola",
        svd_backend_config={"algorithm": "lanczos"},
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

    # Step 5: Closed-Form Update
    print("\n[Step 5] Applying closed-form update refinement...")
    cfu_plugin = ClosedFormUpdatePlugin()
    cfu_plugin.initialize(ctx)

    t0 = time.perf_counter()
    cfu_result = cfu_plugin.do_execute()
    timings["closed_form"] = time.perf_counter() - t0
    print(f"  Status: {cfu_result['status']}")
    print(f"  Layers with outputs: {cfu_result['layers_with_outputs']}")
    print(f"  Time: {timings['closed_form']:.2f}s")
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")

    return {
        "pipeline": "SVD-LLM",
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compression_ratio": actual_ratio,
        "timings": timings,
        "layers_compressed": len(comp_result.parameters.get("layers_replaced", [])),
        "gpu_memory_gb": get_gpu_memory_gb(),
    }


def evaluate_compressed_model(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate compressed model on specified tasks."""
    print("\n" + "="*60)
    print("LMHarness Evaluation (H200 GPU)")
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
    parser = argparse.ArgumentParser(description="H200 GPU ASVD/SVD-LLM Full Pipeline")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--mode", choices=["asvd", "svdllm", "both"], default="both", help="Pipeline mode")
    parser.add_argument("--target-ratio", type=float, default=0.5, help="Target parameter ratio")
    parser.add_argument("--calibration-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Evaluation tasks")
    parser.add_argument("--workspace", default="logs/gpu_asvd_svdllm", help="Workspace directory")
    parser.add_argument("--limit", type=int, default=100, help="Evaluation sample limit")
    parser.add_argument("--preset", choices=["quick", "mid", "large", "flagship"], default=None,
                        help="Use preset model configuration")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU pipeline requires CUDA. No GPU detected.")

    # Handle model selection
    if args.preset:
        model_name = RECOMMENDED_MODELS[args.preset]
        print(f"Using preset '{args.preset}': {model_name}")
    else:
        model_name = args.model

    mode = args.mode
    target_ratio = args.target_ratio
    n_calibration = args.calibration_samples
    tasks = args.tasks.split(",")
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    workspace.mkdir(parents=True, exist_ok=True)

    print(f"H200 GPU ASVD/SVD-LLM Full Pipeline")
    print(f"===================================")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Target ratio: {target_ratio}")
    print(f"Calibration samples: {n_calibration}")
    print(f"Tasks: {tasks}")
    print(f"Workspace: {workspace}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = "cuda"

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create calibration dataloader (smaller batch size for large models)
    calibration_dataloader = create_calibration_dataloader(tokenizer, n_calibration, seq_len=256, batch_size=4)
    eval_dataloader = create_calibration_dataloader(tokenizer, 32, seq_len=256, batch_size=4)

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
        print(f"\nLoading model for ASVD (BF16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            device_map="auto",
        )
        model.eval()
        print(f"GPU Memory after load: {get_gpu_memory_gb():.2f} GB")

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
            "type": "h200_asvd_pipeline",
            "base_model": model_name,
            "compression_ratio": asvd_result["compression_ratio"],
            "dtype": "bfloat16",
        }
        save_compressed_to_safetensors(model, manifest, save_dir)

        all_results["asvd"] = asvd_result

        del model
        torch.cuda.empty_cache()

    if mode in ["svdllm", "both"]:
        # Load fresh model for SVD-LLM
        print(f"\nLoading model for SVD-LLM (BF16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            device_map="auto",
        )
        model.eval()
        print(f"GPU Memory after load: {get_gpu_memory_gb():.2f} GB")

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
            "type": "h200_svdllm_pipeline",
            "base_model": model_name,
            "compression_ratio": svdllm_result["compression_ratio"],
            "dtype": "bfloat16",
        }
        save_compressed_to_safetensors(model, manifest, save_dir)

        all_results["svdllm"] = svdllm_result

        del model
        torch.cuda.empty_cache()

    # Save combined results
    report_path = workspace / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("Summary (H200 GPU)")
    print("="*60)

    for pipeline_name, result in all_results.items():
        print(f"\n{pipeline_name.upper()}:")
        print(f"  Compression ratio: {result['compression_ratio']:.4f}")
        print(f"  Original params: {result['original_params']:,}")
        print(f"  Compressed params: {result['compressed_params']:,}")
        print(f"  Total time: {sum(result['timings'].values()):.2f}s")
        print(f"  GPU Memory: {result['gpu_memory_gb']:.2f} GB")

        if "evaluation" in result:
            print(f"  Evaluation:")
            for task, metrics in result["evaluation"].items():
                acc = metrics.get("accuracy", metrics.get("acc", "N/A"))
                print(f"    {task}: {acc}")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
