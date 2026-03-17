#!/usr/bin/env python3
"""
Perplexity Evaluation Runs for JMLR MLOSS Paper.

Runs WikiText-2 perplexity evaluation before and after compression.
Following PromptBench/T5X patterns for perplexity evaluation.

Demonstrates:
- WikiText-2 perplexity computation
- Per-method perplexity impact analysis
- Compression ratio vs perplexity tradeoff curves

Usage:
  HF_TOKEN=... python scripts/examples/perplexity_evaluation.py [options]

  Options:
    --model MODEL             Model name (default: Qwen/Qwen2.5-0.5B)
    --methods METHOD1,...     Compression methods (default: svd,tucker,cp,tensor_train)
    --ratios RATIO1,...       Compression ratios (default: 0.3,0.5,0.7,0.9)
    --workspace PATH          Workspace directory (default: logs/perplexity)
    --max-samples N           Maximum evaluation samples (default: 256)
    --seq-len N               Sequence length for evaluation (default: 512)
"""
from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.compressed_io import save_compressed_to_safetensors
from src.plugins.compression.consolidator import ModelConsolidator


@dataclass
class PerplexityResult:
    """Single perplexity evaluation result."""
    model: str
    method: str
    compression_ratio: float
    perplexity: float
    loss: float
    total_tokens: int
    original_params: int
    compressed_params: int
    param_ratio: float
    evaluation_time_sec: float
    timestamp: str


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for perplexity evaluation."""

    def __init__(self, tokenizer, max_samples: int = 256, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.seq_len = seq_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        """Load WikiText-2 data."""
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

            # Filter and prepare samples
            texts = []
            for row in ds:
                text = row.get("text", "")
                if isinstance(text, str) and len(text.strip()) > 50:
                    texts.append(text.strip())

            # Combine texts and split into chunks
            combined = " ".join(texts[:self.max_samples * 2])
            tokens = self.tokenizer.encode(combined, add_special_tokens=False)

            # Create fixed-length samples
            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                if len(self.samples) >= self.max_samples:
                    break
                chunk = tokens[i:i + self.seq_len]
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

        except Exception as e:
            print(f"Warning: Could not load WikiText-2: {e}")
            print("Using fallback synthetic data...")
            self._generate_fallback_data()

    def _generate_fallback_data(self):
        """Generate fallback data if WikiText loading fails."""
        fallback_texts = [
            "The quick brown fox jumps over the lazy dog. This is a sample text for perplexity evaluation.",
            "Machine learning models can be compressed using various tensor decomposition techniques.",
            "Natural language processing has advanced significantly with the advent of large language models.",
            "Compression techniques like SVD and Tucker decomposition reduce model parameters while preserving accuracy.",
            "Perplexity is a common metric for evaluating language model quality after compression.",
        ]

        for _ in range(self.max_samples):
            text = " ".join(fallback_texts)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)[:self.seq_len]
            if len(tokens) < self.seq_len:
                tokens = tokens + [self.tokenizer.pad_token_id or 0] * (self.seq_len - len(tokens))
            self.samples.append(torch.tensor(tokens[:self.seq_len], dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        attention_mask = torch.ones_like(input_ids)
        # For causal LM, labels are the same as input_ids (shifted internally)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, float, int]:
    """
    Compute perplexity on the given dataset.

    Returns:
        Tuple of (perplexity, average_loss, total_tokens)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float('inf'), float('inf'), 0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss, total_tokens


def compress_model(
    model: torch.nn.Module,
    method: str,
    target_ratio: float,
    ctx: PipelineContext,
) -> Tuple[float, Dict[str, Any]]:
    """Compress model using specified method."""
    original_params = params_count(model)

    # Find representative layer dimensions
    sample_m, sample_n = 512, 512
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

    # Configure consolidator
    consolidator_kwargs = {
        "compression_method": method,
        "target_modules": target_modules,
    }

    if method == "svd":
        consolidator_kwargs.update({"rank": rank, "svd_backend": "torch"})
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
    max_samples: int,
    seq_len: int,
    workspace: Path,
    hf_token: Optional[str] = None,
) -> List[PerplexityResult]:
    """Run perplexity evaluation for all configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().isoformat()
    results: List[PerplexityResult] = []

    print(f"\n{'='*60}")
    print(f"Perplexity Evaluation")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create WikiText dataset
    print(f"\nLoading WikiText-2 dataset (max {max_samples} samples, seq_len {seq_len})...")
    dataset = WikiTextDataset(tokenizer, max_samples, seq_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    print(f"Loaded {len(dataset)} samples")

    # === BASELINE EVALUATION ===
    print(f"\n[Baseline] Loading and evaluating...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
    )
    baseline_model.to(device).eval()
    baseline_params = params_count(baseline_model)

    t0 = time.perf_counter()
    baseline_ppl, baseline_loss, baseline_tokens = compute_perplexity(
        baseline_model, dataloader, device
    )
    baseline_time = time.perf_counter() - t0

    print(f"  Perplexity: {baseline_ppl:.4f}")
    print(f"  Loss: {baseline_loss:.4f}")
    print(f"  Tokens: {baseline_tokens:,}")
    print(f"  Time: {baseline_time:.2f}s")

    results.append(PerplexityResult(
        model=model_name,
        method="baseline",
        compression_ratio=1.0,
        perplexity=baseline_ppl,
        loss=baseline_loss,
        total_tokens=baseline_tokens,
        original_params=baseline_params,
        compressed_params=baseline_params,
        param_ratio=1.0,
        evaluation_time_sec=baseline_time,
        timestamp=timestamp,
    ))

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === COMPRESSED MODEL EVALUATIONS ===
    for method in methods:
        for ratio in ratios:
            print(f"\n[{method.upper()}] Ratio: {ratio}")

            # Load fresh model
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32, token=hf_token
            )
            model.to(device).eval()

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

                # Evaluate perplexity
                t0 = time.perf_counter()
                ppl, loss, tokens = compute_perplexity(model, dataloader, device)
                eval_time = time.perf_counter() - t0

                print(f"  Perplexity: {ppl:.4f}")
                print(f"  Loss: {loss:.4f}")
                print(f"  PPL delta: {ppl - baseline_ppl:+.4f}")

                results.append(PerplexityResult(
                    model=model_name,
                    method=method,
                    compression_ratio=ratio,
                    perplexity=ppl,
                    loss=loss,
                    total_tokens=tokens,
                    original_params=comp_info["original_params"],
                    compressed_params=comp_info["compressed_params"],
                    param_ratio=comp_info["actual_ratio"],
                    evaluation_time_sec=eval_time,
                    timestamp=timestamp,
                ))

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(PerplexityResult(
                    model=model_name,
                    method=method,
                    compression_ratio=ratio,
                    perplexity=float('inf'),
                    loss=float('inf'),
                    total_tokens=0,
                    original_params=0,
                    compressed_params=0,
                    param_ratio=0.0,
                    evaluation_time_sec=0.0,
                    timestamp=timestamp,
                ))

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def save_results(results: List[PerplexityResult], output_dir: Path):
    """Save results to CSV and JSON files."""
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
        f.write("# WikiText-2 Perplexity Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Find baseline
        baseline = next((r for r in results if r.method == "baseline"), None)
        if baseline:
            f.write("## Baseline\n\n")
            f.write(f"- Model: {baseline.model}\n")
            f.write(f"- Parameters: {baseline.original_params:,}\n")
            f.write(f"- Perplexity: {baseline.perplexity:.4f}\n")
            f.write(f"- Loss: {baseline.loss:.4f}\n\n")

        # Method comparison table
        f.write("## Compression Results\n\n")
        f.write("| Method | Target Ratio | Actual Ratio | Perplexity | PPL Delta |\n")
        f.write("|--------|--------------|--------------|------------|----------|\n")

        for r in sorted(results, key=lambda x: (x.method, x.compression_ratio)):
            if r.method == "baseline":
                continue
            ppl_delta = r.perplexity - baseline.perplexity if baseline else 0.0
            f.write(f"| {r.method} | {r.compression_ratio:.2f} | {r.param_ratio:.4f} | ")
            f.write(f"{r.perplexity:.4f} | {ppl_delta:+.4f} |\n")

        f.write("\n")

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiText-2 Perplexity Evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--methods", default="svd,tucker,cp,tensor_train", help="Methods to evaluate")
    parser.add_argument("--ratios", default="0.3,0.5,0.7,0.9", help="Compression ratios")
    parser.add_argument("--workspace", default="logs/perplexity", help="Workspace directory")
    parser.add_argument("--max-samples", type=int, default=256, help="Max evaluation samples")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model
    methods = args.methods.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    workspace = Path(args.workspace).resolve()

    hf_token = os.environ.get("HF_TOKEN")

    print("WikiText-2 Perplexity Evaluation")
    print("=================================")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Max samples: {args.max_samples}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Workspace: {workspace}")

    results = run_perplexity_evaluation(
        model_name=model_name,
        methods=methods,
        ratios=ratios,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        workspace=workspace,
        hf_token=hf_token,
    )

    save_results(results, workspace)

    print(f"\nCompleted {len(results)} perplexity evaluations.")


if __name__ == "__main__":
    main()
