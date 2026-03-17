"""
MemoryInferenceProfiler plugin: profiles memory during inference-only runs
using random inputs (ids or raw tensor), without requiring a tokenizer.

Config keys (under 'memory_inference'):
- enabled: bool (optional)
- device: 'auto'|'cpu'|'cuda' (optional)
- repeats: int, number of forward repeats per phase (default 1)
- output_dir: str, directory to write reports (default test_logs/memory)
- isolate_subprocess: bool, run profiling in a fresh subprocess when possible (default true)
- subprocess: dict, optional overrides for isolation
    - script_path: path to subprocess runner script
    - model_path: explicit model id/path for subprocess loading
    - timeout: optional subprocess timeout (seconds)
- input:
    kind: 'ids' | 'tensor'
    batch_size: int (ids)
    seq_len: int (ids)
    tensor_shape: list[int] (tensor)
    arg_name: str, forward kwarg name for tensor input (default 'inputs')

Additional kwargs to execute():
- tag: Optional[str] to distinguish phases (e.g., 'original', 'pruned')
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import torch
import sys

from ...framework.plugins import Plugin, PluginMetadata
from ...framework.memory_profiler import MemoryProfiler, DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS
from ...framework.reproducibility import get_generator, get_seed


class MemoryInferenceProfiler(Plugin):
    """Profiles inference memory with random inputs and saves JSON reports."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description="Profile inference memory on random inputs (ids/tensor)",
            category="analysis",
        )

    def _resolve_device(self, preferred: Optional[str]) -> torch.device:
        if preferred and preferred != "auto":
            return torch.device(preferred)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_ids_inputs(self, model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        bs = int(cfg.get("batch_size", 1))
        seqlen = int(cfg.get("seq_len", 16))

        vocab_size = 1000
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
            emb = model.model.get_input_embeddings()
        else:
            emb = None
        if emb is not None and hasattr(emb, "num_embeddings"):
            vocab_size = int(emb.num_embeddings)

        # Use seeded generator for reproducibility
        generator = get_generator()
        input_ids = torch.randint(low=0, high=max(2, vocab_size), size=(bs, seqlen), dtype=torch.long, device=device, generator=generator)
        attention_mask = torch.ones((bs, seqlen), dtype=torch.long, device=device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _build_tensor_inputs(self, cfg: Dict[str, Any], device: torch.device) -> Tuple[str, torch.Tensor]:
        shape = cfg.get("tensor_shape") or [1, 8]
        arg_name = cfg.get("arg_name", "inputs")
        # Use seeded generator for reproducibility
        generator = get_generator()
        t = torch.randn(*[int(x) for x in shape], dtype=torch.float32, device=device, generator=generator)
        return arg_name, t

    def _forward(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> Any:
        model.eval()
        with torch.no_grad():
            return model(**inputs) if inputs and isinstance(inputs, dict) else model(inputs)

    def do_execute(self, **kwargs) -> Dict[str, Any]:
        config = dict(self._configuration or {})
        mem_cfg = dict(config.get("memory_inference") or {})
        if not mem_cfg.get("enabled", True):
            return {"enabled": False}

        tag = kwargs.get("tag") or mem_cfg.get("tag") or "inference"

        # Resolve model and device
        model: Optional[torch.nn.Module] = kwargs.get("model")
        if model is None:
            model = self.get_model("current")
        device = self._resolve_device(mem_cfg.get("device"))
        model = model.to(device)

        repeats = int(mem_cfg.get("repeats", 1))
        warmup = int(mem_cfg.get("warmup", 1))
        sync_gpu = bool(mem_cfg.get("synchronize_gpu", True))
        output_dir = mem_cfg.get("output_dir", "test_logs/memory")
        isolate_flag = mem_cfg.get("isolate_subprocess")
        if isolate_flag is None:
            isolate_flag = DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS
        isolate_subprocess = bool(isolate_flag or (mem_cfg.get("subprocess") or {}).get("enabled"))
        sub_cfg = dict(mem_cfg.get("subprocess") or {})

        # Build inputs
        input_cfg = dict(mem_cfg.get("input") or {})
        kind = input_cfg.get("kind", "tensor")

        if kind == "ids":
            inputs = self._build_ids_inputs(model, input_cfg, device)
        else:
            arg, tensor = self._build_tensor_inputs(input_cfg, device)
            inputs = {arg: tensor}

        # Optional warmup to exclude lazy initialization/caching from profiling
        if warmup > 0:
            with torch.inference_mode():
                for _ in range(int(warmup)):
                    _ = self._forward(model, inputs)

        phase_name = f"{tag}_forward"
        profiler = MemoryProfiler()
        out_path: Optional[Path] = None
        used_subprocess = False

        if isolate_subprocess:
            script_path = sub_cfg.get("script_path")
            if script_path is None:
                script_path = Path(__file__).resolve().parents[3] / "scripts" / "utils" / "profile_inference_subprocess.py"
            script_path = Path(script_path)

            model_path = sub_cfg.get("model_path") or sub_cfg.get("model")
            if model_path is None:
                # Try to save the model for subprocess loading
                model_dir = Path(output_dir) / f"{tag}_model"
                try:
                    model_dir.mkdir(parents=True, exist_ok=True)
                    if any(m.__class__.__name__.startswith("Factor") for m in model.modules()):
                        from ...framework.compressed_io import save_compressed_to_safetensors
                        base_name = (config.get("model") or {}).get("name") or getattr(model, "name_or_path", None)
                        if not base_name:
                            raise RuntimeError("Compressed model requires base_model to serialize for subprocess.")
                        module_paths = []
                        module_types = {}
                        func_names = {}
                        factor_sizes = {}
                        from ...framework.layers import FactorLinear, FactorEmbedding
                        for name, mod in model.named_modules():
                            if isinstance(mod, (FactorLinear, FactorEmbedding)):
                                module_paths.append(name)
                                module_types[name] = mod.__class__.__name__
                                fn = getattr(mod, "func_name", None) or getattr(getattr(mod, "weight", None), "func_name", None)
                                if isinstance(fn, str):
                                    func_names[name] = fn
                                if isinstance(mod, FactorLinear):
                                    sizes = []
                                    for f in getattr(getattr(mod, "weight", None), "factors", []) or []:
                                        if hasattr(f, "weight") and hasattr(f.weight, "shape"):
                                            sizes.append(list(f.weight.shape))
                                    if sizes:
                                        factor_sizes[name] = sizes
                        manifest = {
                            "type": "memory_inference",
                            "base_model": base_name,
                            "modules_replaced": module_paths,
                            "save_format": "safetensors_state_dict",
                        }
                        if module_types:
                            manifest["module_types"] = module_types
                        if func_names:
                            manifest["func_names"] = func_names
                        if factor_sizes:
                            manifest["factor_sizes"] = factor_sizes
                        save_compressed_to_safetensors(model, manifest, model_dir)
                    elif hasattr(model, "save_pretrained"):
                        model.save_pretrained(model_dir)
                    else:
                        raise RuntimeError("Model does not support save_pretrained; cannot isolate in subprocess.")
                    model_path = str(model_dir)
                except Exception as exc:
                    self.logger.warning("Subprocess isolation unavailable; falling back to in-process (%s)", exc)
                    model_path = None

            if model_path and script_path.exists():
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{tag}_inference.json"
                # Get seed for subprocess reproducibility
                seed = get_seed() or 42
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--model", str(model_path),
                    "--output", str(out_path),
                    "--device", str(device),
                    "--phase_name", str(phase_name),
                    "--input_kind", str(kind),
                    "--repeats", str(repeats),
                    "--warmup", str(warmup),
                    "--seed", str(seed),
                ]
                if sync_gpu:
                    cmd.append("--sync_gpu")
                if kind == "ids":
                    cmd += ["--batch_size", str(input_cfg.get("batch_size", 1)), "--seq_len", str(input_cfg.get("seq_len", 16))]
                else:
                    shape = input_cfg.get("tensor_shape") or [1, 8]
                    cmd += ["--tensor_shape", ",".join(str(int(x)) for x in shape), "--arg_name", str(input_cfg.get("arg_name", "inputs"))]
                trust_remote = bool((config.get("model") or {}).get("trust_remote_code", False))
                if trust_remote:
                    cmd.append("--trust_remote_code")
                profiler.profile_execution_subprocess(
                    command=cmd,
                    report_path=out_path,
                    phase_name=phase_name,
                    timeout=sub_cfg.get("timeout"),
                    replace=True,
                )
                used_subprocess = True

        if not used_subprocess:
            with profiler.profile_execution(phase_name, sync_gpu=sync_gpu) as _p:
                with torch.inference_mode():
                    for _ in range(max(1, repeats)):
                        _ = self._forward(model, inputs)
                        if sync_gpu and torch.cuda.is_available():
                            torch.cuda.synchronize()

            # Save report JSON
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tag}_inference.json"
            profiler.export_report(str(out_path))
        elif out_path is None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tag}_inference.json"

        # Augment report with isolated inference peaks for easier comparison
        prof = profiler.get_profile(phase_name)
        if prof is not None:
            peak_gpu = 0.0
            peak_rss = 0.0
            if prof.peak_memory is not None:
                peak_gpu = max(0.0, float(prof.peak_memory.gpu_allocated_mb - prof.pre_execution.gpu_allocated_mb))
                peak_rss = max(0.0, float(prof.peak_memory.rss_mb - prof.pre_execution.rss_mb))
            # Update JSON
            import json as _json
            data = _json.loads(out_path.read_text())
            data.setdefault("summary", {})
            data["summary"]["inference_peak_gpu_delta_mb"] = peak_gpu
            data["summary"]["inference_peak_rss_delta_mb"] = peak_rss
            out_path.write_text(_json.dumps(data, indent=2))

        # Return loaded report content for convenience
        report = __import__("json").loads(out_path.read_text())
        result = {"tag": tag, "report_path": str(out_path), "report": report}
        if self.state_manager:
            self.state_manager.state.set(f"memory_inference.{tag}", result)
        return result
