#!/usr/bin/env python3
"""
Profile Compressed Models - H200 GPU Version.

MemoryProfiler for compressed model I/O and inference profiling.
Optimized for H200 GPU with 140GB VRAM.

Usage:
  HF_TOKEN=... python scripts/examples/gpu/profile_compressed.py [options]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.framework.context import PipelineContext
from src.framework.memory_profiler import MemoryProfiler
from src.framework.compressed_io import save_compressed_to_safetensors, load_compressed_from_safetensors
from src.plugins.compression.consolidator import ModelConsolidator


@dataclass
class ProfileResult:
    """Single profiling result."""
    model: str
    profile_type: str
    load_time_sec: float
    save_time_sec: float
    inference_time_ms: float
    model_memory_gb: float
    peak_gpu_memory_gb: float
    tokens_per_second: float
    file_size_gb: float
    timestamp: str


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def profile_inference(model, batch_size: int = 1, seq_len: int = 256, warmup: int = 5, repeats: int = 20):
    """Profile model inference."""
    model.eval()
    device = next(model.parameters()).device
    vocab_size = getattr(model.config, "vocab_size", 32000)

    inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Profile
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg_time_ms = sum(times) / len(times)
    tokens_per_second = (batch_size * seq_len) / (avg_time_ms / 1000)
    peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    return {
        "avg_inference_time_ms": avg_time_ms,
        "tokens_per_second": tokens_per_second,
        "peak_gpu_memory_gb": peak_gpu,
    }


def run_profiling(
    model_name: str,
    workspace: Path,
    compression_ratio: float = 0.5,
    hf_token: Optional[str] = None,
) -> List[ProfileResult]:
    """Run comprehensive profiling."""
    if not torch.cuda.is_available():
        raise RuntimeError("H200 GPU profiling requires CUDA.")

    timestamp = datetime.now().isoformat()
    results: List[ProfileResult] = []

    print(f"\n{'='*60}")
    print(f"H200 GPU Compressed Model Profiling")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)

    # === BASELINE PROFILING ===
    print("\n[Baseline] Loading and profiling...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()
    load_time = time.perf_counter() - t0

    model_memory_gb = get_gpu_memory_gb()
    inf_profile = profile_inference(model)

    print(f"  Load time: {load_time:.2f}s")
    print(f"  Model memory: {model_memory_gb:.2f} GB")
    print(f"  Inference: {inf_profile['avg_inference_time_ms']:.2f} ms")
    print(f"  Tokens/sec: {inf_profile['tokens_per_second']:.1f}")

    # Save baseline
    save_dir = workspace / "baseline"
    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest = {"type": "baseline", "base_model": model_name, "dtype": "bfloat16"}
    save_compressed_to_safetensors(model, manifest, save_dir)
    save_time = time.perf_counter() - t0

    file_size = sum(f.stat().st_size for f in save_dir.glob("*.safetensors")) / 1024**3

    results.append(ProfileResult(
        model=model_name,
        profile_type="baseline",
        load_time_sec=load_time,
        save_time_sec=save_time,
        inference_time_ms=inf_profile["avg_inference_time_ms"],
        model_memory_gb=model_memory_gb,
        peak_gpu_memory_gb=inf_profile["peak_gpu_memory_gb"],
        tokens_per_second=inf_profile["tokens_per_second"],
        file_size_gb=file_size,
        timestamp=timestamp,
    ))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # === COMPRESSED PROFILING ===
    print("\n[Compressed] Loading, compressing, and profiling...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        token=hf_token, device_map="auto",
    )
    model.eval()
    load_time = time.perf_counter() - t0

    # Compress
    ctx = PipelineContext(config={}, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer

    sample_m, sample_n = 4096, 4096
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sample_m, sample_n = module.weight.shape
            break

    rank = max(1, int(compression_ratio * (sample_m * sample_n) / (sample_m + sample_n + 1)))
    rank = min(rank, min(sample_m, sample_n))

    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=[
            "model.layers[*].self_attn.q_proj",
            "model.layers[*].self_attn.k_proj",
            "model.layers[*].self_attn.v_proj",
            "model.layers[*].self_attn.o_proj",
            "model.layers[*].mlp.gate_proj",
            "model.layers[*].mlp.up_proj",
            "model.layers[*].mlp.down_proj",
        ],
        rank=rank,
        svd_backend="cola",
    )
    consolidator.initialize(ctx)

    t0 = time.perf_counter()
    consolidator.compress_model_with_surgery(model)
    compress_time = time.perf_counter() - t0
    print(f"  Compression time: {compress_time:.2f}s")

    model_memory_gb = get_gpu_memory_gb()
    inf_profile = profile_inference(model)

    print(f"  Model memory: {model_memory_gb:.2f} GB")
    print(f"  Inference: {inf_profile['avg_inference_time_ms']:.2f} ms")
    print(f"  Tokens/sec: {inf_profile['tokens_per_second']:.1f}")

    # Save compressed
    save_dir = workspace / "compressed"
    save_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest = {
        "type": "compressed_svd",
        "base_model": model_name,
        "compression_ratio": compression_ratio,
        "dtype": "bfloat16",
    }
    save_compressed_to_safetensors(model, manifest, save_dir)
    save_time = time.perf_counter() - t0

    file_size = sum(f.stat().st_size for f in save_dir.glob("*.safetensors")) / 1024**3

    results.append(ProfileResult(
        model=model_name,
        profile_type="compressed_svd",
        load_time_sec=load_time + compress_time,
        save_time_sec=save_time,
        inference_time_ms=inf_profile["avg_inference_time_ms"],
        model_memory_gb=model_memory_gb,
        peak_gpu_memory_gb=inf_profile["peak_gpu_memory_gb"],
        tokens_per_second=inf_profile["tokens_per_second"],
        file_size_gb=file_size,
        timestamp=timestamp,
    ))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def save_results(results: List[ProfileResult], output_dir: Path):
    """Save profiling results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "profile_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    summary_path = output_dir / "profile_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# H200 GPU Compressed Model Profiling\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Results\n\n")
        f.write("| Type | Load (s) | Save (s) | Inference (ms) | Memory (GB) | Tok/s | Size (GB) |\n")
        f.write("|------|----------|----------|----------------|-------------|-------|----------|\n")

        for r in results:
            f.write(f"| {r.profile_type} | {r.load_time_sec:.2f} | {r.save_time_sec:.2f} | ")
            f.write(f"{r.inference_time_ms:.2f} | {r.model_memory_gb:.2f} | ")
            f.write(f"{r.tokens_per_second:.0f} | {r.file_size_gb:.2f} |\n")

    print(f"\nResults saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 GPU Compressed Model Profiling")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--ratio", type=float, default=0.5, help="Compression ratio")
    parser.add_argument("--workspace", default="logs/gpu_profile_compressed", help="Workspace")
    return parser.parse_args()


def main():
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    hf_token = os.environ.get("HF_TOKEN")

    results = run_profiling(
        model_name=args.model,
        workspace=workspace,
        compression_ratio=args.ratio,
        hf_token=hf_token,
    )

    save_results(results, workspace)


if __name__ == "__main__":
    main()
