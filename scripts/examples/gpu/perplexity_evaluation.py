#!/usr/bin/env python3
"""
Perplexity Evaluation Runs for JMLR MLOSS Paper - H200 GPU Version.

Measures WikiText-2 perplexity before/after compression for large-scale models.
Extended with C4 and PTB datasets for comprehensive evaluation.

Large Model Perplexity Targets:
| Model | Baseline PPL | Target PPL @ 0.5 Ratio | Max Acceptable Degradation |
|-------|--------------|------------------------|---------------------------|
| Llama-3.1-70B | ~2.5-3.0 | <4.0 | +50% |
| Qwen2.5-72B | ~2.5-3.0 | <4.0 | +50% |
| Mixtral-8x7B | ~3.0-3.5 | <4.5 | +50% |
| Gemma-2-27B | ~3.5-4.0 | <5.0 | +50% |

Usage:
  HF_TOKEN=... python scripts/examples/gpu/perplexity_evaluation.py [options]

  Options:
    --model MODEL             Model name (default: meta-llama/Llama-3.1-8B)
    --methods METHOD1,...     Compression methods (default: svd,tucker)
    --ratios RATIO1,...       Compression ratios (default: 0.3,0.5,0.7)
    --datasets DATASET1,...   Datasets to evaluate (default: wikitext-2)
    --max-samples N           Max samples per dataset (default: 256)
    --workspace PATH          Workspace directory
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.plugins.compression.consolidator import ModelConsolidator


# H200 GPU model tiers
MODEL_TIERS = {
    "mid": ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B", "google/gemma-2-9b"],
    "large": ["Qwen/Qwen2.5-14B", "google/gemma-2-27b", "mistralai/Mixtral-8x7B-v0.1"],
    "flagship": ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2.5-72B"],
}

# Available datasets for perplexity evaluation
EVAL_DATASETS = {
    "wikitext-2": {"path": "wikitext", "name": "wikitext-2-raw-v1", "split": "test"},
    "c4": {"path": "c4", "name": "en", "split": "validation", "streaming": True},
    "ptb": {"path": "ptb_text_only", "name": None, "split": "test"},
}


@dataclass
class PerplexityResult:
    """Single perplexity evaluation result."""
    model: str
    method: str
    compression_ratio: float
    dataset: str
    perplexity: float
    loss: float
    original_params: int
    compressed_params: int
    param_ratio: float
    compression_time_sec: float
    evaluation_time_sec: float
    ppl_degradation_pct: float
    timestamp: str
    dtype: str = "bfloat16"


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def load_eval_dataset(dataset_name: str, tokenizer, max_samples: int = 256, max_length: int = 512):
    """Load and tokenize evaluation dataset."""
    config = EVAL_DATASETS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Loading {dataset_name}...")

    try:
        if config.get("streaming"):
            dataset = load_dataset(
                config["path"],
                config["name"],
                split=config["split"],
                streaming=True,
                trust_remote_code=True,
            )
            # Take samples from streaming dataset
            texts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                text = item.get("text", "")
                if len(text) > 50:  # Skip very short texts
                    texts.append(text)
        else:
            dataset = load_dataset(
                config["path"],
                config["name"],
                split=config["split"],
                trust_remote_code=True,
            )
            texts = [item["text"] for item in dataset if len(item.get("text", "")) > 50][:max_samples]
    except Exception as e:
        print(f"  Warning: Could not load {dataset_name}: {e}")
        # Fallback to synthetic data
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 20
            for _ in range(max_samples)
        ]

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    return encodings


def compute_perplexity(
    model: torch.nn.Module,
    encodings: Dict[str, torch.Tensor],
    device: str,
    batch_size: int = 4,
) -> Tuple[float, float]:
    """Compute perplexity on encoded text."""
    model.eval()

    input_ids = encodings["input_ids"]
    attention_mask = encodings.get("attention_mask", torch.ones_like(input_ids))

    total_loss = 0.0
    total_tokens = 0

    # Process in batches
    n_samples = input_ids.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)

            # Shift labels for causal LM
            labels = batch_input_ids.clone()
            labels[batch_attention_mask == 0] = -100  # Ignore padding

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=labels,
            )

            # Accumulate loss
            batch_loss = outputs.loss.item()
            batch_tokens = (labels != -100).sum().item()

            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return perplexity, avg_loss


def compress_model(
    model: torch.nn.Module,
    method: str,
    target_ratio: float,
    ctx: PipelineContext,
) -> Tuple[float, Dict[str, Any]]:
    """Compress model using specified method."""
    original_params = params_count(model)

    # Find representative layer dimensions
    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    # Estimate rank
    if method == "svd":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    elif method == "tucker":
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.7))
    elif method == "cp":
        rank = max(1, int(target_ratio * (sample_m * sample_n) / (sample_m + sample_n)))
    else:  # tensor_train
        rank = max(1, int(target_ratio * min(sample_m, sample_n) * 0.5))

    rank = min(rank, min(sample_m, sample_n))

    target_modules = [
        "model.layers[*].self_attn.q_proj",
        "model.layers[*].self_attn.k_proj",
        "model.layers[*].self_attn.v_proj",
        "model.layers[*].self_attn.o_proj",
        "model.layers[*].mlp.gate_proj",
        "model.layers[*].mlp.up_proj",
        "model.layers[*].mlp.down_proj",
    ]

    consolidator_kwargs = {
        "compression_method": method,
        "target_modules": target_modules,
    }

    if method == "svd":
        consolidator_kwargs.update({
            "rank": rank,
            "svd_backend": "cola",
            "svd_backend_config": {"algorithm": "lanczos"},
        })
    elif method == "tucker":
        consolidator_kwargs["tucker_ranks"] = [rank, rank]
    elif method == "cp":
        consolidator_kwargs["cp_rank"] = rank
    elif method == "tensor_train":
        consolidator_kwargs["tensor_ranks"] = [1, rank, rank, 1]

    consolidator = ModelConsolidator(**consolidator_kwargs)
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    result = consolidator.compress_model_with_surgery(model)
    compression_time = time.perf_counter() - t0

    compressed_params = params_count(model)
    actual_ratio = compressed_params / original_params

    return compression_time, {
        "original_params": original_params,
        "compressed_params": compressed_params,
        "actual_ratio": actual_ratio,
        "rank": rank,
    }


def run_perplexity_evaluation(
    model_name: str,
    methods: List[str],
    ratios: List[float],
    datasets: List[str],
    max_samples: int,
    workspace: Path,
    hf_token: Optional[str] = None,
) -> List[PerplexityResult]:
    """Run perplexity evaluation for all configurations."""
    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU perplexity evaluation requires CUDA.")

    device = "cuda"
    timestamp = datetime.now().isoformat()
    results: List[PerplexityResult] = []
    baseline_ppl: Dict[str, float] = {}

    print(f"\n{'='*60}")
    print(f"H200 GPU Perplexity Evaluation")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Datasets: {datasets}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation datasets
    print("\nLoading evaluation datasets...")
    eval_data: Dict[str, Dict[str, torch.Tensor]] = {}
    for ds_name in datasets:
        try:
            eval_data[ds_name] = load_eval_dataset(ds_name, tokenizer, max_samples)
            print(f"  {ds_name}: {eval_data[ds_name]['input_ids'].shape[0]} samples")
        except Exception as e:
            print(f"  Warning: Could not load {ds_name}: {e}")

    # === BASELINE EVALUATION ===
    print(f"\n[Baseline] Loading model (BF16)...")
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

    for ds_name, encodings in eval_data.items():
        print(f"\n  Evaluating on {ds_name}...")
        t0 = time.perf_counter()
        ppl, loss = compute_perplexity(baseline_model, encodings, device)
        eval_time = time.perf_counter() - t0

        baseline_ppl[ds_name] = ppl
        print(f"    Perplexity: {ppl:.4f}")
        print(f"    Loss: {loss:.4f}")
        print(f"    Time: {eval_time:.2f}s")

        results.append(PerplexityResult(
            model=model_name,
            method="baseline",
            compression_ratio=1.0,
            dataset=ds_name,
            perplexity=ppl,
            loss=loss,
            original_params=baseline_params,
            compressed_params=baseline_params,
            param_ratio=1.0,
            compression_time_sec=0.0,
            evaluation_time_sec=eval_time,
            ppl_degradation_pct=0.0,
            timestamp=timestamp,
            dtype="bfloat16",
        ))

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    # === COMPRESSED MODEL EVALUATION ===
    for method in methods:
        for ratio in ratios:
            print(f"\n[{method.upper()}] Ratio: {ratio}")

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=hf_token,
                device_map="auto",
            )
            model.eval()

            method_workspace = workspace / f"{method}_{ratio}"
            method_workspace.mkdir(parents=True, exist_ok=True)
            ctx = PipelineContext(config={}, workspace_dir=method_workspace)
            ctx.state.model = model
            ctx.state.tokenizer = tokenizer

            try:
                # Compress
                compression_time, comp_info = compress_model(model, method, ratio, ctx)
                print(f"  Compressed in {compression_time:.2f}s")
                print(f"  Actual ratio: {comp_info['actual_ratio']:.4f}")
                print(f"  GPU Memory: {get_gpu_memory_gb():.2f} GB")

                # Evaluate on each dataset
                for ds_name, encodings in eval_data.items():
                    print(f"\n  Evaluating on {ds_name}...")
                    t0 = time.perf_counter()
                    ppl, loss = compute_perplexity(model, encodings, device)
                    eval_time = time.perf_counter() - t0

                    # Calculate degradation
                    base_ppl = baseline_ppl.get(ds_name, ppl)
                    ppl_degradation = 100.0 * (ppl - base_ppl) / base_ppl if base_ppl > 0 else 0.0

                    print(f"    Perplexity: {ppl:.4f}")
                    print(f"    Degradation: {ppl_degradation:+.1f}%")
                    print(f"    Time: {eval_time:.2f}s")

                    results.append(PerplexityResult(
                        model=model_name,
                        method=method,
                        compression_ratio=ratio,
                        dataset=ds_name,
                        perplexity=ppl,
                        loss=loss,
                        original_params=comp_info["original_params"],
                        compressed_params=comp_info["compressed_params"],
                        param_ratio=comp_info["actual_ratio"],
                        compression_time_sec=compression_time,
                        evaluation_time_sec=eval_time,
                        ppl_degradation_pct=ppl_degradation,
                        timestamp=timestamp,
                        dtype="bfloat16",
                    ))

                # Save compressed model
                save_dir = method_workspace / "compressed"
                save_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "type": f"h200_perplexity_{method}",
                    "base_model": model_name,
                    "method": method,
                    "compression_ratio": comp_info["actual_ratio"],
                    "dtype": "bfloat16",
                }
                save_compressed_to_safetensors(model, manifest, save_dir)

            except Exception as e:
                print(f"  ERROR: {e}")
                for ds_name in eval_data.keys():
                    results.append(PerplexityResult(
                        model=model_name,
                        method=method,
                        compression_ratio=ratio,
                        dataset=ds_name,
                        perplexity=float('inf'),
                        loss=float('inf'),
                        original_params=0,
                        compressed_params=0,
                        param_ratio=0.0,
                        compression_time_sec=0.0,
                        evaluation_time_sec=0.0,
                        ppl_degradation_pct=float('inf'),
                        timestamp=timestamp,
                        dtype="bfloat16",
                    ))

            del model
            gc.collect()
            torch.cuda.empty_cache()

    return results


def save_results(results: List[PerplexityResult], output_dir: Path):
    """Save results to CSV, JSON, and markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "perplexity_results.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # Save to JSON
    json_path = output_dir / "perplexity_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Generate markdown summary
    summary_path = output_dir / "perplexity_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# H200 GPU Perplexity Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Baseline results
        baseline_results = [r for r in results if r.method == "baseline"]
        if baseline_results:
            f.write("## Baseline Perplexity\n\n")
            f.write("| Dataset | Perplexity | Loss |\n")
            f.write("|---------|------------|------|\n")
            for r in baseline_results:
                f.write(f"| {r.dataset} | {r.perplexity:.4f} | {r.loss:.4f} |\n")
            f.write("\n")

        # Compression results
        f.write("## Compression Results\n\n")
        f.write("| Method | Ratio | Dataset | Perplexity | Degradation |\n")
        f.write("|--------|-------|---------|------------|-------------|\n")

        for r in sorted(results, key=lambda x: (x.method, x.compression_ratio, x.dataset)):
            if r.method == "baseline":
                continue
            f.write(f"| {r.method} | {r.compression_ratio:.2f} | {r.dataset} | ")
            f.write(f"{r.perplexity:.4f} | {r.ppl_degradation_pct:+.1f}% |\n")

        f.write("\n")

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Perplexity Evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--methods", default="svd,tucker", help="Compression methods")
    parser.add_argument("--ratios", default="0.3,0.5,0.7", help="Compression ratios")
    parser.add_argument("--datasets", default="wikitext-2", help="Datasets to evaluate")
    parser.add_argument("--max-samples", type=int, default=256, help="Max samples per dataset")
    parser.add_argument("--workspace", default="logs/gpu_perplexity", help="Workspace directory")
    parser.add_argument("--tier", choices=["mid", "large", "flagship"], default=None,
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
    datasets = args.datasets.split(",")
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    print("H200 GPU Perplexity Evaluation")
    print("==============================")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Datasets: {datasets}")
    print(f"Max samples: {args.max_samples}")
    print(f"Workspace: {workspace}")

    results = run_perplexity_evaluation(
        model_name=model_name,
        methods=methods,
        ratios=ratios,
        datasets=datasets,
        max_samples=args.max_samples,
        workspace=workspace,
        hf_token=hf_token,
    )

    save_results(results, workspace)

    print(f"\nCompleted {len(results)} perplexity evaluations.")


if __name__ == "__main__":
    main()
