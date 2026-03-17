#!/usr/bin/env python3
"""
Loop Gemma-3 SVD compression over a list of ranks from CSV.

Reads rank specs from a CSV file and runs the Gemma-3 SVD pipeline for each row,
mirroring the structure used by scripts/examples/svd_gemma3.py.
Evaluation tasks are read from the YAML config (or overridden via --tasks) and
propagated into the per-run config/results.

CSV formats supported:
  - Headered: rank,mlp_rank,label
  - Positional: rank[,mlp_rank[,label]]

Usage:
  python scripts/examples/loop_svd_gemma3.py --input-csv path/to/ranks.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM as AutoModel, AutoTokenizer

from src.framework.compressed_io import save_compressed_to_safetensors
from src.framework.context import PipelineContext
from src.framework.layers import FactorLinear
from src.plugins.compression.consolidator import ModelConsolidator
from src.plugins.evaluation.lm_eval import LMHarness

DEFAULT_CONFIG = "config/svd_gemma3.yaml"
DEFAULT_INPUT_CSV = "config/gemma3_ranks.csv"
RESULT_KEYS = [
    "rank_attention",
    "rank_mlp",
    "tasks",
    "backend",
    "workspace",
    "save_dir",
    "orig_model_params",
    "comp_model_params",
    "compression_ratio",
    "compression_sec",
    "evaluation_sec",
    "evaluation_metrics",
    "total_sec",
    "status",
    "error_message",
]


@dataclass
class RankSpec:
    attention_rank: int
    mlp_rank: Optional[int] = None
    label: Optional[str] = None

    def label_suffix(self) -> str:
        mlp = self.mlp_rank if self.mlp_rank is not None else self.attention_rank
        if self.label:
            return self.label
        if mlp == self.attention_rank:
            return f"rank{self.attention_rank}"
        return f"rank_att{self.attention_rank}_mlp{mlp}"


def _configure_linalg_backend():
    preferred = os.environ.get("TORCH_LINALG_PREFERRED")
    if preferred:
        torch.backends.cuda.preferred_linalg_library(preferred)
        print(f"[linalg] TORCH_LINALG_PREFERRED={preferred} -> set preferred backend")
    else:
        current = torch.backends.cuda.preferred_linalg_library()
        print(f"[linalg] using default backend: {current}")
    return torch.backends.cuda.preferred_linalg_library()


def _run_with_linalg_retry(fn, *, fallback_env: str = "TORCH_LINALG_FALLBACK"):
    try:
        return fn()
    except RuntimeError as exc:
        msg = str(exc)
        lowered = msg.lower()
        if "cusolver" in lowered or "cusolverdncreate" in lowered:
            fallback = os.environ.get(fallback_env, "magma")
            print(f"[linalg] detected cuSolver failure: {msg}")
            if fallback:
                torch.backends.cuda.preferred_linalg_library(fallback)
                print(f"[linalg] retrying with fallback backend={fallback} (set {fallback_env} to override)")
                return fn()
        raise


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _safe_empty_cache() -> None:
    try:
        if _cuda_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _parse_yes_no(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"Unrecognized yes/no value: {value}")


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _select_backend(use_cola: bool, svd_cfg: dict) -> tuple[str, dict]:
    backend = svd_cfg.get("backend_when_cola" if use_cola else "backend_when_no_cola")
    if not backend:
        backend = svd_cfg.get("backend", "cola" if use_cola else "torch")

    base_cfg = dict(svd_cfg.get("backend_config") or {})
    specific = svd_cfg.get("cola" if use_cola else "torch") or svd_cfg.get("backend_config_no_cola") or {}
    merged_cfg = {**base_cfg, **specific}
    if use_cola and not merged_cfg:
        merged_cfg = {"algorithm": "dense"}
    return backend, merged_cfg


def _select_workspace(runtime_cfg: dict, use_cola: bool, override: str | None) -> Path:
    if override:
        return _resolve_path(override)
    log_root = runtime_cfg.get("log_root", "logs")
    log_root_path = _resolve_path(log_root)
    names = runtime_cfg.get("workspace_names", {}) or {}
    default_name = "examples/gemma3_svd_cola" if use_cola else "examples/gemma3_svd"
    workspace_name = names.get("cola" if use_cola else "non_cola", default_name)
    return (log_root_path / workspace_name).resolve()


def _select_save_dir(cfg: dict, workspace: Path, compressed_subdir: str) -> Path:
    save_cfg = cfg.get("save", {}) or {}
    compressed_dir = save_cfg.get("compressed_dir")
    if compressed_dir:
        return _resolve_path(compressed_dir)
    return (workspace / compressed_subdir).resolve()


def random_ids_inputs(model, batch_size: int, seq_len: int, device: str):
    vocab = getattr(getattr(model, "config", None), "vocab_size", 32000)
    input_ids = torch.randint(low=0, high=int(vocab), size=(batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def params_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _parse_int_field(value: str | None) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_rank_row(row: dict | list[str]) -> Optional[RankSpec]:
    if isinstance(row, dict):
        lowered = {str(k).lower(): v for k, v in row.items()}
        attn = _parse_int_field(
            lowered.get("rank")
            or lowered.get("attention_rank")
            or lowered.get("attention")
        )
        if attn is None:
            return None
        mlp = _parse_int_field(lowered.get("mlp_rank") or lowered.get("mlp"))
        label = (lowered.get("label") or lowered.get("name") or "").strip() or None
        return RankSpec(attention_rank=attn, mlp_rank=mlp, label=label)

    vals = [c.strip() for c in row if c and c.strip()]
    if not vals:
        return None
    attn = _parse_int_field(vals[0])
    if attn is None:
        return None
    mlp = _parse_int_field(vals[1]) if len(vals) > 1 else None
    label = vals[2] if len(vals) > 2 else None
    return RankSpec(attention_rank=attn, mlp_rank=mlp, label=label)


def load_rank_specs(csv_path: Path, start_line_idx: Optional[int]) -> list[RankSpec]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as rf:
        specs: list[RankSpec] = []
        reader = csv.reader(rf)
        for idx, row in enumerate(reader, start=1):
            if start_line_idx is not None and idx != start_line_idx:
                continue
            spec = _parse_rank_row(row)
            if spec:
                specs.append(spec)
            if start_line_idx is not None and idx >= start_line_idx:
                break
    return specs


def _mlp_patterns(objects: list[str]) -> list[str]:
    bases = []
    if any(obj.startswith("model.language_model.layers") for obj in objects):
        bases.append("model.language_model.layers")
    if any(obj.startswith("model.layers") for obj in objects):
        bases.append("model.layers")
    if not bases:
        bases.append("model.layers")
    patterns = []
    for base in bases:
        patterns.extend(
            [
                f"{base}[*].mlp.gate_proj",
                f"{base}[*].mlp.up_proj",
                f"{base}[*].mlp.down_proj",
            ]
        )
    return patterns


def _upsert_override(overrides: list[dict], pattern: str, func_name: str, rank: int) -> None:
    for ov in overrides:
        if ov.get("pattern") == pattern:
            ov["rank"] = rank
            ov.setdefault("func_name", func_name)
            return
    overrides.append({"pattern": pattern, "func_name": func_name, "rank": rank})


def apply_rank_to_config(cfg: dict, spec: RankSpec) -> dict:
    cfg_copy = deepcopy(cfg)
    fact_cfg = cfg_copy.setdefault("factorization", {})
    func_name = fact_cfg.get("func_name", "svd")
    fact_cfg["rank"] = int(spec.attention_rank)
    mlp_rank = int(spec.mlp_rank) if spec.mlp_rank is not None else fact_cfg["rank"]

    patterns = _mlp_patterns(fact_cfg.get("objects", []) or [])
    overrides = list(fact_cfg.get("overrides") or [])
    for pattern in patterns:
        _upsert_override(overrides, pattern, func_name, mlp_rank)
    if overrides:
        fact_cfg["overrides"] = overrides
    return cfg_copy


def _normalize_tasks(tasks_val) -> list[str]:
    if tasks_val is None:
        return []
    if isinstance(tasks_val, str):
        return [t.strip() for t in tasks_val.split(",") if t and t.strip()]
    if isinstance(tasks_val, (list, tuple)):
        tasks: list[str] = []
        for t in tasks_val:
            text = str(t).strip()
            if text:
                tasks.append(text)
        return tasks
    return []


def _resolve_tasks(cfg: dict, override: Optional[str]) -> list[str]:
    if override is not None:
        return _normalize_tasks(override)
    eval_cfg = (cfg.get("evaluation", {}) or {})
    return _normalize_tasks(eval_cfg.get("tasks"))


def _tasks_to_str(tasks: list[str]) -> str:
    return ",".join(tasks)


def _resolve_eval_device(eval_cfg: dict) -> str:
    requested = str(eval_cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested.startswith("cuda") and not _cuda_available():
        print(f"[device] Requested CUDA device '{requested}' but CUDA is unavailable; falling back to cpu.")
        return "cpu"
    return requested


def _evaluate_model(
    model,
    tokenizer,
    tasks: list[str],
    eval_cfg: dict,
    ctx: PipelineContext,
):
    if not tasks:
        return {}, 0.0

    plugin = LMHarness(
        tasks=tasks,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        max_length=int(eval_cfg.get("max_length", 2048)),
        device=_resolve_eval_device(eval_cfg),
        limit=eval_cfg.get("limit"),
        model_type=eval_cfg.get("model_type", "auto"),
        compat_mode=None,
        compat_plugin_name="LMEval",
        backend=eval_cfg.get("backend", "hf"),
        hf_model_name=eval_cfg.get("hf_model_name"),
        hf_kwargs=eval_cfg.get("hf_kwargs"),
    )
    plugin.initialize(ctx)
    t0 = time.perf_counter()
    results = plugin.do_execute(model=model, tokenizer=tokenizer, tasks=tasks)
    elapsed = time.perf_counter() - t0
    metrics = {task: res.metrics for task, res in (results or {}).items()}
    return metrics, elapsed


def run_single(cfg: dict, *, use_cola: bool, workspace: Path) -> dict:
    _safe_empty_cache()
    runtime_cfg = cfg.get("runtime", {}) or {}
    compressed_subdir = runtime_cfg.get("compressed_subdir", "compressed")
    save_dir = _select_save_dir(cfg, workspace, compressed_subdir)
    workspace.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    token_env = cfg.get("auth", {}).get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)

    import transformers.modeling_utils as _mu
    if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
        _mu.ALL_PARALLEL_STYLES = {"colwise", "rowwise", "sequence", "context"}

    _configure_linalg_backend()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, token=token)
    device = "cuda" if _cuda_available() else "cpu"
    model.to(device).eval()

    ctx = PipelineContext(config=cfg, workspace_dir=workspace)
    ctx.state.model = model
    ctx.state.tokenizer = tok

    ids_cfg = (cfg.get("memory_inference", {}) or {}).get("input", {})
    bs = int(ids_cfg.get("batch_size", 1))
    seqlen = int(ids_cfg.get("seq_len", 128))


    original_params = params_count(model)
    t0 =  time.perf_counter()
    fact_cfg = cfg.get("factorization", {}) or {}
    objects = fact_cfg.get("objects", [])
    svd_cfg = (cfg.get("compression") or {}).get("svd", {}) or {}
    svd_backend, svd_backend_config = _select_backend(use_cola, svd_cfg)
    eval_cfg = (cfg.get("evaluation") or {})
    consolidator = ModelConsolidator(
        compression_method=fact_cfg.get("func_name", "svd"),
        target_modules=objects,
        method_overrides=fact_cfg.get("overrides"),
        rank=fact_cfg.get("rank", fact_cfg.get("svd_rank", 8)),
        svd_backend=svd_backend,
        svd_backend_config=svd_backend_config,
    )
    consolidator.initialize(ctx)
    t_comp0 = time.perf_counter()
    comp_result = _run_with_linalg_retry(
        lambda: consolidator.compress_model_with_surgery(model, method_overrides=fact_cfg.get("overrides"))
    )
    t_comp1 = time.perf_counter()

    compressed_params = params_count(model)
    ratio = float(compressed_params) / float(original_params) if original_params > 0 else 1.0
    tasks_list = _normalize_tasks((cfg.get("evaluation") or {}).get("tasks"))
    eval_metrics, eval_time = _evaluate_model(model, tok, tasks_list, eval_cfg, ctx)

    layers_replaced = comp_result.parameters.get("layers_replaced", [])
    module_types = {}
    func_names = {}
    factor_sizes = {}
    for path in layers_replaced:
        stats = comp_result.parameters.get("surgery_stats", {}).get(path, {})
        if stats.get("layer_type"):
            module_types[path] = stats["layer_type"]
        mod = consolidator._get_module_by_name(model, path)
        func_name = getattr(mod, "func_name", None) or getattr(getattr(mod, "weight", None), "func_name", None)
        if isinstance(func_name, str):
            func_names[path] = func_name
        if isinstance(mod, FactorLinear):
            sizes = []
            for factor in getattr(getattr(mod, "weight", None), "factors", []) or []:
                if hasattr(factor, "weight") and hasattr(factor.weight, "shape"):
                    sizes.append(list(factor.weight.shape))
            if sizes:
                factor_sizes[path] = sizes

    manifest = {
        "type": "gemma3_svd_cola" if use_cola else "gemma3_svd",
        "base_model": model_name,
        "modules_replaced": layers_replaced,
        "save_format": "safetensors_state_dict",
    }
    if module_types:
        manifest["module_types"] = module_types
    if func_names:
        manifest["func_names"] = func_names
    if factor_sizes:
        manifest["factor_sizes"] = factor_sizes

    save_compressed_to_safetensors(model, manifest, save_dir)

    total_time = time.perf_counter() - t0
    timings = {
        "total": total_time,
        "compression": t_comp1 - t_comp0,
        "evaluation": eval_time,
    }

    result = {
        "workspace": str(workspace),
        "save_dir": str(save_dir),
        "backend": svd_backend,
        "tasks": _tasks_to_str(tasks_list),
        "orig_model_params": original_params,
        "comp_model_params": compressed_params,
        "compression_ratio": ratio,
        "compression_sec": timings["compression"],
        "evaluation_sec": timings["evaluation"],
        "evaluation_metrics": json.dumps(eval_metrics) if eval_metrics else "",
        "total_sec": total_time,
        "status": "ok",
        "error_message": "",
    }

    del model
    _safe_empty_cache()
    return result


def _append_csv_row(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()
    with open(out_path, "a", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=RESULT_KEYS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_KEYS})


def _append_text_log(out_path: Path, row: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [
        "rank_attention",
        "rank_mlp",
        "tasks",
        "backend",
        "workspace",
        "save_dir",
        "orig_model_params",
        "comp_model_params",
        "compression_ratio",
        "compression_sec",
        "evaluation_sec",
        "evaluation_metrics",
        "total_sec",
        "status",
        "error_message",
    ]
    with open(out_path, "a") as wf:
        for key in ordered:
            val = row.get(key, "")
            wf.write(f"{key}\t{val}\n")
        wf.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop Gemma-3 SVD compression over ranks from CSV.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base YAML config (per-run ranks are injected).")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="CSV containing rank specs.")
    parser.add_argument(
        "--cola",
        choices=["yes", "no"],
        default="no",
        help="Use CoLA backend (yes/no). CLI overrides config default.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace root; per-run subfolders are appended automatically.",
    )
    parser.add_argument(
        "--start-line-idx",
        type=int,
        default=None,
        help="1-based line index in input CSV to process a single spec; if unset, process all lines.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV path to append summarized results (created if missing).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional text log file path (key\\tvalue lines per run). Defaults to ../logs/act-remote-eval/gemma3_svd_eval_results.txt",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated tasks override; defaults to evaluation.tasks in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    base_cfg = yaml.safe_load(cfg_path.read_text())
    runtime_cfg = base_cfg.get("runtime", {}) or {}
    tasks = _resolve_tasks(base_cfg, args.tasks)
    tasks_str = _tasks_to_str(tasks)
    if tasks:
        base_cfg.setdefault("evaluation", {})["tasks"] = tasks
    default_log = _resolve_path(args.log_file) if args.log_file else (ROOT.parent / "logs" / "act-remote-eval" / "gemma3_svd_eval_results.txt")

    specs = load_rank_specs(_resolve_path(args.input_csv), args.start_line_idx)
    if not specs:
        print(f"No rank specs parsed from {args.input_csv}")
        return

    cli_cola = _parse_yes_no(args.cola)
    for idx, spec in enumerate(specs, start=1):
        cfg = apply_rank_to_config(base_cfg, spec)
        use_cola = cli_cola
        if use_cola is None:
            use_cola = _parse_yes_no(runtime_cfg.get("use_cola_default"))
        if use_cola is None:
            use_cola = True

        base_workspace = _select_workspace(runtime_cfg, use_cola, args.workspace)
        workspace = (base_workspace / spec.label_suffix()).resolve()

        print(f"\n== Run {idx}: attn_rank={spec.attention_rank} mlp_rank={spec.mlp_rank or spec.attention_rank} ==")
        print(f"workspace: {workspace}")
        print(f"config: {cfg_path}")
        if tasks:
            print(f"tasks: {tasks_str}")
        try:
            res = run_single(cfg, use_cola=use_cola, workspace=workspace)
            res["rank_attention"] = spec.attention_rank
            res["rank_mlp"] = spec.mlp_rank or spec.attention_rank
            res["tasks"] = tasks_str
            for k in RESULT_KEYS:
                if k in res:
                    print(f"{k}\t{res[k]}")
            if args.out_csv:
                _append_csv_row(_resolve_path(args.out_csv), res)
            if default_log:
                _append_text_log(default_log, res)
        except Exception as exc:
            err_row = {
                "rank_attention": spec.attention_rank,
                "rank_mlp": spec.mlp_rank or spec.attention_rank,
                "tasks": tasks_str,
                "status": "error",
                "error_message": str(exc),
                "workspace": str(workspace),
                "save_dir": "",
                "backend": "",
                "orig_model_params": "",
                "comp_model_params": "",
                "compression_ratio": "",
                "compression_sec": "",
                "evaluation_sec": "",
                "evaluation_metrics": "",
                "total_sec": "",
            }
            print(json.dumps(err_row, indent=2))
            if args.out_csv:
                _append_csv_row(_resolve_path(args.out_csv), err_row)
            if default_log:
                _append_text_log(default_log, err_row)
            continue


if __name__ == "__main__":
    main()
