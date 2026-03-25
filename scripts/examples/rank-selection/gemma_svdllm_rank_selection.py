#!python
"""
Standalone Gemma SVD-LLM rank-selection example.

This mirrors the dataset and model setup from
scripts/examples/rank-selection/gemma_asvd_ppl_rank_selection.py, but runs the
current production SVD-LLM pipeline instead of the ASVD + PPL sensitivity
pipeline.

Flow:
1. Load Gemma and tokenizer
2. Build a PipelineContext
3. Collect short wikitext calibration/eval subsets
4. Evaluate the baseline model with LMHarness
5. Run the sequential SVD-LLM pipeline on the requested target layers
6. Evaluate the compressed model with LMHarness
7. Save the compressed model to safetensors

Usage:
  HF_TOKEN=... python \
      scripts/examples/rank-selection/gemma_svdllm_rank_selection.py
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.framework.compressed_io import save_compressed_to_safetensors
from goldcrest.framework.context import PipelineContext
from goldcrest.framework.layers import FactorLinear
from goldcrest.plugins.compression.svdllm_pipeline import SVDLLMPipelinePlugin
from goldcrest.plugins.evaluation.lm_eval import LMHarness

MODEL_ID = "google/gemma-3-270m-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--workspace",
        default="logs/examples/rank-selection/gemma_svdllm_rank_selection",
        help="Workspace directory for PipelineContext artifacts",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save the compressed model; defaults to <workspace>/compressed",
    )
    parser.add_argument(
        "--targets",
        default="model.layers.0.mlp.gate_proj,model.layers.0.self_attn.q_proj",
        help="Comma-separated target module paths",
    )
    parser.add_argument("--calibration-samples", type=int, default=4)
    parser.add_argument("--calibration-seq-len", type=int, default=32)
    parser.add_argument(
        "--param-ratio-target",
        "--rank-ratio-target",
        dest="param_ratio_target",
        type=float,
        default=0.1,
        help="Target ratio mapped to rank with k = floor(m*n*r/(m+n))",
    )
    parser.add_argument("--min-rank", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument(
        "--disable-closed-form-update",
        action="store_true",
        help="Disable the SVD-LLM closed-form refinement step",
    )
    parser.add_argument("--lm-limit", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-text-chars", type=int, default=96)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Preferred runtime device",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_short_wikitext_texts(
    *,
    split: str,
    count: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("This example requires the 'datasets' package.") from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts: list[str] = []

    for row in dataset:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if len(text) < min_chars or len(text) > max_chars:
            continue
        texts.append(text)
        if len(texts) >= count:
            break

    if len(texts) < count:
        raise RuntimeError(
            f"Unable to collect {count} short wikitext examples from split={split!r} "
            f"with {min_chars} <= len(text) <= {max_chars}."
        )

    return texts


def make_text_loader(
    tokenizer,
    texts: list[str],
    *,
    seq_len: int = 32,
    include_labels: bool = False,
) -> DataLoader:
    samples = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if include_labels:
            batch["labels"] = encoded["input_ids"].clone()
        samples.append(batch)
    return DataLoader(samples, batch_size=None)


@contextlib.contextmanager
def patched_wikitext_subset(texts: list[str]) -> Iterator[None]:
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError("This example requires the 'datasets' package for LMHarness.") from exc

    if not texts:
        raise ValueError("Expected at least one short wikitext text.")
    if not all(len(text) < 100 for text in texts):
        raise ValueError("All local wikitext texts must be shorter than 100 characters.")

    split_dataset = datasets.Dataset.from_dict({"page": list(texts)})
    dataset = datasets.DatasetDict(
        {
            "train": split_dataset,
            "validation": split_dataset,
            "test": split_dataset,
        }
    )
    original_load_dataset = datasets.load_dataset
    original_nested = getattr(getattr(datasets, "load", None), "load_dataset", None)

    def patched_load_dataset(path, *args, **kwargs):
        dataset_name = args[0] if args else kwargs.get("name")
        if path == "EleutherAI/wikitext_document_level" and dataset_name == "wikitext-2-raw-v1":
            return dataset
        return original_load_dataset(path, *args, **kwargs)

    datasets.load_dataset = patched_load_dataset
    if original_nested is not None:
        datasets.load.load_dataset = patched_load_dataset

    try:
        yield
    finally:
        datasets.load_dataset = original_load_dataset
        if original_nested is not None:
            datasets.load.load_dataset = original_nested


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"Module path not found: {path}")
    return modules[path]


def factor_shapes(layer: FactorLinear) -> list[tuple[int, ...]]:
    return [tuple(factor.weight.shape) for factor in layer.weight.factors]


def summarize_task_results(task_name: str, results: dict) -> dict:
    task_result = results[task_name]
    return {
        "num_samples": task_result.num_samples,
        "metrics": task_result.metrics,
    }


def build_save_manifest(
    *,
    model: torch.nn.Module,
    model_name: str,
    compression_result: dict,
) -> dict:
    layers_replaced = compression_result.get("layers_replaced", [])
    layer_results = compression_result.get("layer_results", {}) or {}

    surgery_stats: dict[str, dict] = {}
    module_types: dict[str, str] = {}
    func_names: dict[str, str] = {}
    factor_sizes: dict[str, list[list[int]]] = {}
    selected_ranks: dict[str, int] = {}

    for state_name, layer_result in layer_results.items():
        surgery_name = str(layer_result.get("surgery_name") or state_name)
        rank = layer_result.get("rank")
        if rank is not None:
            selected_ranks[surgery_name] = int(rank)

        compression_stats = layer_result.get("compression", {}) or {}
        surgery_stats.update(compression_stats.get("surgery_stats", {}) or {})

    for path in layers_replaced:
        stats = surgery_stats.get(path, {}) or {}
        if isinstance(stats.get("layer_type"), str):
            module_types[path] = stats["layer_type"]

        module = get_module_by_path(model, path)
        func_name = getattr(module, "func_name", None) or getattr(
            getattr(module, "weight", None),
            "func_name",
            None,
        )
        if isinstance(func_name, str):
            func_names[path] = func_name

        if isinstance(module, FactorLinear):
            sizes = []
            for factor in getattr(getattr(module, "weight", None), "factors", []) or []:
                if hasattr(factor, "weight") and hasattr(factor.weight, "shape"):
                    sizes.append(list(factor.weight.shape))
            if sizes:
                factor_sizes[path] = sizes

    manifest = {
        "type": "gemma_svdllm_rank_selection",
        "base_model": model_name,
        "modules_replaced": layers_replaced,
        "save_format": "safetensors_state_dict",
        "use_closed_form_update": compression_result.get("use_closed_form_update"),
    }
    if selected_ranks:
        manifest["selected_ranks"] = selected_ranks
    if module_types:
        manifest["module_types"] = module_types
    if func_names:
        manifest["func_names"] = func_names
    if factor_sizes:
        manifest["factor_sizes"] = factor_sizes
    return manifest


def main() -> int:
    args = parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]

    if not targets:
        raise ValueError("At least one target layer is required.")

    workspace = Path(args.workspace)
    save_dir = Path(args.save_dir) if args.save_dir else workspace / "compressed"
    workspace.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"trust_remote_code": True}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token

    runtime_device = resolve_device(args.device)
    print(f"Loading model: {args.model}")
    print(f"Using device: {runtime_device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        **kwargs,
    ).to(runtime_device).eval()

    context = PipelineContext(config={}, workspace_dir=workspace)
    context.state.model = model
    context.state.tokenizer = tokenizer

    calibration_texts = load_short_wikitext_texts(
        split="train",
        count=args.calibration_samples,
        min_chars=args.min_text_chars,
        max_chars=args.max_text_chars,
    )
    eval_texts = load_short_wikitext_texts(
        split="test",
        count=args.lm_limit,
        min_chars=args.min_text_chars,
        max_chars=args.max_text_chars,
    )

    with patched_wikitext_subset(eval_texts):
        evaluator = LMHarness(
            tasks=["wikitext"],
            batch_size=1,
            max_length=args.max_length,
            limit=args.lm_limit,
            backend="auto",
            device=str(model_device(model)),
            name="lm_harness",
        )
        evaluator.initialize(context)

        baseline_results = evaluator.execute(model=model, tokenizer=tokenizer)
        baseline_summary = summarize_task_results("wikitext", baseline_results)
        print("\nBaseline LMHarness summary:")
        print(json.dumps(baseline_summary, indent=2, default=str))

        calibration_loader = make_text_loader(
            tokenizer,
            calibration_texts,
            seq_len=args.calibration_seq_len,
        )
        pipeline = SVDLLMPipelinePlugin(
            target_modules=targets,
            rank_ratio=args.param_ratio_target,
            min_rank=args.min_rank,
            regularization=args.regularization,
            svd_backend="torch",
            use_closed_form_update=not args.disable_closed_form_update,
            name="svdllm_pipeline",
        )
        pipeline.initialize(context)
        compression_result = pipeline.execute(
            model=model,
            dataloader=calibration_loader,
            n_samples=args.calibration_samples,
        )
        print("\nSVD-LLM pipeline result:")
        print(
            json.dumps(
                {
                    "targets": compression_result.get("targets", []),
                    "layers_processed": compression_result.get("layers_processed"),
                    "layers_replaced": compression_result.get("layers_replaced", []),
                    "failures": compression_result.get("failures", {}),
                    "use_closed_form_update": compression_result.get("use_closed_form_update"),
                },
                indent=2,
                default=str,
            )
        )

        if compression_result.get("failures"):
            raise RuntimeError(
                "SVD-LLM pipeline reported failures: "
                f"{json.dumps(compression_result['failures'], indent=2, default=str)}"
            )

        print("\nSelected factor shapes:")
        for state_name, layer_result in (compression_result.get("layer_results", {}) or {}).items():
            surgery_name = str(layer_result.get("surgery_name") or state_name)
            layer = get_module_by_path(model, surgery_name)
            chosen_rank = layer_result.get("rank", context.state.get(f"svd.ranks.{state_name}"))
            if not isinstance(layer, FactorLinear):
                raise RuntimeError(f"Expected FactorLinear at {surgery_name}, got {type(layer)}")
            print(
                json.dumps(
                    {
                        "target": surgery_name,
                        "state_name": state_name,
                        "chosen_rank": chosen_rank,
                        "factor_shapes": factor_shapes(layer),
                    },
                    indent=2,
                )
            )

        compressed_results = evaluator.execute(model=model, tokenizer=tokenizer)
        compressed_summary = summarize_task_results("wikitext", compressed_results)
        print("\nCompressed LMHarness summary:")
        print(json.dumps(compressed_summary, indent=2, default=str))

        finite_metrics = [
            float(value)
            for value in compressed_summary["metrics"].values()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if not finite_metrics:
            raise RuntimeError("Compressed LMHarness run did not produce any finite numeric metrics.")

        manifest = build_save_manifest(
            model=model,
            model_name=args.model,
            compression_result=compression_result,
        )
        manifest_path, weights_path = save_compressed_to_safetensors(model, manifest, save_dir)
        print("\nSaved compressed model:")
        print(
            json.dumps(
                {
                    "save_dir": str(save_dir),
                    "manifest_path": str(manifest_path),
                    "weights_path": str(weights_path),
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
