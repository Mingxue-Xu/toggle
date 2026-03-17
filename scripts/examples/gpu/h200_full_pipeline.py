#!/usr/bin/env python3
"""
H200 GPU Full Pipeline Orchestration.

End-to-end orchestration example demonstrating the complete compression workflow
on H200 GPU with 140GB VRAM for large-scale models.

Pipeline stages:
1. Model loading (BF16)
2. Activation analysis
3. Compression strategy selection
4. SVD/Tucker/CP/TT compression
5. Model consolidation
6. Evaluation (LMHarness)
7. Memory profiling
8. Results export

Recommended Execution Strategy:
- Phase 1 (Mid-Size): Llama-3.1-8B, Qwen2.5-14B, Gemma-2-9B
- Phase 2 (Large): Gemma-2-27B, Mixtral-8x7B
- Phase 3 (Flagship): Llama-3.1-70B, Qwen2.5-72B

Usage:
  HF_TOKEN=... python scripts/examples/gpu/h200_full_pipeline.py [options]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
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
from src.framework.memory_profiler import MemoryProfiler
from src.plugins.analysis.activation_metrics import ActivationMetricsPlugin
from src.plugins.compression.calibration_collector import CalibrationCollectorPlugin
from src.plugins.compression.svd_activation_scaling import ActivationScalingPlugin
from src.plugins.compression.svd_binary_search_rank import BinarySearchRankPlugin
from src.plugins.compression.consolidator import ModelConsolidator
from src.plugins.evaluation.lm_eval import LMHarness


# H200 GPU Model Tiers
MODEL_TIERS = {
    "quick": ["meta-llama/Llama-3.1-8B"],
    "mid": ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-14B", "google/gemma-2-9b"],
    "large": ["google/gemma-2-27b", "mistralai/Mixtral-8x7B-v0.1"],
    "flagship": ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"],
}

# Compression methods available
COMPRESSION_METHODS = ["svd", "tucker", "cp", "tensor_train"]

# Default evaluation tasks
DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]


class CalibrationDataset(Dataset):
    """Calibration dataset for activation analysis."""

    def __init__(self, tokenizer, n_samples: int = 128, seq_len: int = 256):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can be compressed effectively.",
            "Large language models have revolutionized NLP.",
            "Tensor decomposition reduces computational requirements.",
            "Activation-aware compression preserves model quality.",
        ]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]
        encoding = self.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.seq_len
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def run_full_pipeline(
    model_name: str,
    methods: List[str],
    ratios: List[float],
    tasks: List[str],
    workspace: Path,
    calibration_samples: int = 128,
    eval_limit: Optional[int] = 100,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full compression and evaluation pipeline on H200 GPU."""

    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU pipeline requires CUDA. No GPU detected.")

    device = "cuda"
    timestamp = datetime.now().isoformat()
    results = {
        "model": model_name,
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "methods": {},
        "baseline": {},
    }

    print(f"\n{'='*70}")
    print(f"H200 GPU Full Pipeline Orchestration")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Tasks: {tasks}")
    print(f"GPU: {results['gpu']}")
    print(f"GPU Memory: {results['gpu_memory_total_gb']:.1f} GB")
    print(f"{'='*70}")

    # Load tokenizer
    print("\n[Pipeline] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create calibration dataloader
    calib_dataset = CalibrationDataset(tokenizer, calibration_samples, seq_len=256)
    calib_dataloader = DataLoader(calib_dataset, batch_size=4, shuffle=False)

    # ========== BASELINE EVALUATION ==========
    print("\n[Pipeline] Stage 1: Baseline Evaluation")
    print("-" * 50)

    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        device_map="auto",
    )
    baseline_model.eval()
    baseline_params = params_count(baseline_model)

    print(f"  Parameters: {baseline_params:,}")
    print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")

    # Evaluate baseline
    evaluator = LMHarness(tasks=tasks)
    baseline_metrics = {}
    t0 = time.perf_counter()
    for task in tasks:
        result = evaluator.evaluate_task(baseline_model, tokenizer, task, limit=eval_limit)
        baseline_metrics[task] = result.metrics
        print(f"  {task}: {result.metrics.get('accuracy', 'N/A')}")
    baseline_eval_time = time.perf_counter() - t0

    results["baseline"] = {
        "params": baseline_params,
        "gpu_memory_gb": get_gpu_memory_gb(),
        "evaluation": baseline_metrics,
        "eval_time_sec": baseline_eval_time,
    }

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    # ========== COMPRESSION RUNS ==========
    for method in methods:
        results["methods"][method] = {}

        for ratio in ratios:
            print(f"\n[Pipeline] Stage 2: {method.upper()} Compression @ {ratio}")
            print("-" * 50)

            run_result = {
                "method": method,
                "target_ratio": ratio,
                "stages": {},
            }

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=hf_token,
                device_map="auto",
            )
            model.eval()
            original_params = params_count(model)
            print(f"  Loaded model: {original_params:,} params")

            method_workspace = workspace / f"{method}_{ratio}"
            method_workspace.mkdir(parents=True, exist_ok=True)
            ctx = PipelineContext(config={}, workspace_dir=method_workspace)
            ctx.state.model = model
            ctx.state.tokenizer = tokenizer

            try:
                # Stage 2a: Activation Analysis
                print("  [2a] Activation analysis...")
                act_plugin = ActivationMetricsPlugin()
                act_plugin.initialize(ctx)
                t0 = time.perf_counter()
                act_report = act_plugin.do_execute(model=model, tokenizer=tokenizer)
                run_result["stages"]["activation_analysis"] = {
                    "time_sec": time.perf_counter() - t0,
                    "layers_analyzed": len(act_report.get("per_layer", [])),
                }
                print(f"    Time: {run_result['stages']['activation_analysis']['time_sec']:.2f}s")

                # Stage 2b: Calibration Collection (for ASVD methods)
                print("  [2b] Calibration collection...")
                calib_plugin = CalibrationCollectorPlugin(
                    n_samples=min(calibration_samples, 32),
                    collect_activations=False,
                    collect_outputs=False,
                    collect_xtx=True,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                calib_plugin.initialize(ctx)
                t0 = time.perf_counter()
                calib_result = calib_plugin.do_execute(model=model, dataloader=calib_dataloader)
                run_result["stages"]["calibration"] = {
                    "time_sec": time.perf_counter() - t0,
                    "layers_collected": calib_result["layers_collected"],
                    "samples": calib_result["samples"],
                }
                print(f"    Layers: {calib_result['layers_collected']}, Time: {run_result['stages']['calibration']['time_sec']:.2f}s")
                gc.collect()
                torch.cuda.empty_cache()

                # Stage 2c: Rank Allocation
                print("  [2c] Rank allocation...")
                rank_plugin = BinarySearchRankPlugin(
                    target_mode="param_ratio",
                    param_ratio_target=ratio,
                    sensitivity_source="stable_rank",
                    min_rank=32,
                )
                rank_plugin.initialize(ctx)

                target_modules = [
                    "model.layers[*].self_attn.q_proj",
                    "model.layers[*].self_attn.k_proj",
                    "model.layers[*].self_attn.v_proj",
                    "model.layers[*].self_attn.o_proj",
                    "model.layers[*].mlp.gate_proj",
                    "model.layers[*].mlp.up_proj",
                    "model.layers[*].mlp.down_proj",
                ]

                t0 = time.perf_counter()
                rank_result = rank_plugin.do_execute(model=model, target_layers=target_modules)
                run_result["stages"]["rank_allocation"] = {
                    "time_sec": time.perf_counter() - t0,
                    "layers_allocated": len(rank_result["ranks"]),
                }
                print(f"    Time: {run_result['stages']['rank_allocation']['time_sec']:.2f}s")

                # Free calibration data before heavy compression
                deleted = CalibrationCollectorPlugin.clear_calibration_data(ctx.state)
                print(f"    Freed {deleted} calibration entries")
                torch.cuda.empty_cache()

                # Stage 2d: Compression
                print("  [2d] Model compression...")
                consolidator_kwargs = {
                    "compression_method": method,
                    "target_modules": target_modules,
                    "device": device,
                }

                sample_m, sample_n = 4096, 4096
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        sample_m, sample_n = module.weight.shape
                        break

                if method == "svd":
                    rank = max(1, int(ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
                    consolidator_kwargs.update({
                        "rank": min(rank, min(sample_m, sample_n)),
                        "svd_backend": "torch",
                    })
                elif method == "tucker":
                    rank = max(1, int(ratio * min(sample_m, sample_n) * 0.7))
                    consolidator_kwargs["tucker_ranks"] = [min(rank, sample_m), min(rank, sample_n)]
                elif method == "cp":
                    rank = max(1, int(ratio * (sample_m * sample_n) / (sample_m + sample_n)))
                    consolidator_kwargs["cp_rank"] = min(rank, min(sample_m, sample_n))
                else:  # tensor_train
                    rank = max(1, int(ratio * min(sample_m, sample_n) * 0.5))
                    consolidator_kwargs["tensor_ranks"] = [1, min(rank, sample_m), min(rank, sample_n), 1]

                consolidator = ModelConsolidator(**consolidator_kwargs)
                consolidator.initialize(ctx)

                t0 = time.perf_counter()
                comp_result = consolidator.compress_model_with_surgery(model)
                compression_time = time.perf_counter() - t0

                compressed_params = params_count(model)
                actual_ratio = compressed_params / original_params

                if actual_ratio >= 0.99:
                    layers_replaced = comp_result.parameters.get("layers_replaced", [])
                    print(f"    WARNING: Compression ratio {actual_ratio:.4f} indicates no effective compression!")
                    print(f"    Layers replaced: {len(layers_replaced)}")

                run_result["stages"]["compression"] = {
                    "time_sec": compression_time,
                    "original_params": original_params,
                    "compressed_params": compressed_params,
                    "actual_ratio": actual_ratio,
                    "layers_replaced": len(comp_result.parameters.get("layers_replaced", [])),
                }
                print(f"    Compression: {compression_time:.2f}s, ratio: {actual_ratio:.4f}")
                print(f"    GPU Memory: {get_gpu_memory_gb():.2f} GB")

                # Stage 2e: Evaluation
                print("  [2e] Model evaluation...")
                eval_metrics = {}
                t0 = time.perf_counter()
                for task in tasks:
                    result = evaluator.evaluate_task(model, tokenizer, task, limit=eval_limit)
                    eval_metrics[task] = result.metrics
                    print(f"    {task}: {result.metrics.get('accuracy', 'N/A')}")
                eval_time = time.perf_counter() - t0

                run_result["stages"]["evaluation"] = {
                    "time_sec": eval_time,
                    "metrics": eval_metrics,
                }

                # Stage 2f: Memory Profiling
                print("  [2f] Memory profiling...")
                run_result["gpu_memory_gb"] = get_gpu_memory_gb()

                # Save compressed model
                save_dir = method_workspace / "compressed"
                save_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "type": f"h200_full_pipeline_{method}",
                    "base_model": model_name,
                    "method": method,
                    "target_ratio": ratio,
                    "actual_ratio": actual_ratio,
                    "dtype": "bfloat16",
                }
                save_compressed_to_safetensors(model, manifest, save_dir)
                print(f"    Saved to: {save_dir}")

            except Exception as e:
                print(f"  ERROR: {e}")
                run_result["error"] = str(e)

            results["methods"][method][str(ratio)] = run_result

            del model
            gc.collect()
            torch.cuda.empty_cache()

    return results


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save pipeline results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "full_pipeline_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown summary
    summary_path = output_dir / "full_pipeline_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# H200 GPU Full Pipeline Results\n\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"GPU: {results['gpu']}\n")
        f.write(f"Generated: {results['timestamp']}\n\n")

        f.write("## Baseline\n\n")
        f.write(f"- Parameters: {results['baseline']['params']:,}\n")
        f.write(f"- GPU Memory: {results['baseline']['gpu_memory_gb']:.2f} GB\n\n")

        f.write("## Compression Results\n\n")
        f.write("| Method | Ratio | Actual | Compressed Params | GPU Mem (GB) | Time (s) |\n")
        f.write("|--------|-------|--------|-------------------|--------------|----------|\n")

        for method, ratios in results["methods"].items():
            for ratio, run in ratios.items():
                if "error" in run:
                    f.write(f"| {method} | {ratio} | ERROR | - | - | - |\n")
                else:
                    comp = run["stages"].get("compression", {})
                    f.write(f"| {method} | {ratio} | {comp.get('actual_ratio', 0):.4f} | ")
                    f.write(f"{comp.get('compressed_params', 0):,} | ")
                    f.write(f"{run.get('gpu_memory_gb', 0):.1f} | ")
                    f.write(f"{comp.get('time_sec', 0):.1f} |\n")

        f.write("\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Full Pipeline Orchestration")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--methods", default="svd,tucker", help="Compression methods")
    parser.add_argument("--ratios", default="0.5", help="Compression ratios")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS[:3]), help="Evaluation tasks")
    parser.add_argument("--calibration-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--eval-limit", type=int, default=100, help="Evaluation sample limit")
    parser.add_argument("--workspace", default="logs/gpu_full_pipeline", help="Workspace directory")
    parser.add_argument("--tier", choices=["quick", "mid", "large", "flagship"], default=None,
                        help="Model tier preset")
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle model selection
    if args.tier:
        model_name = MODEL_TIERS[args.tier][0]
        print(f"Using tier '{args.tier}': {model_name}")
    else:
        model_name = args.model

    methods = args.methods.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    tasks = args.tasks.split(",")
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    results = run_full_pipeline(
        model_name=model_name,
        methods=methods,
        ratios=ratios,
        tasks=tasks,
        workspace=workspace,
        calibration_samples=args.calibration_samples,
        eval_limit=args.eval_limit,
        hf_token=hf_token,
    )

    save_results(results, workspace)


if __name__ == "__main__":
    main()
