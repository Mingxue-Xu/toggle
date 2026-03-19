"""
Inference subprocess utilities for isolated memory profiling.

Provides both a class-based API (InferenceSubprocess) and a CLI-compatible entrypoint
used by scripts/utils/profile_inference_subprocess.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS = True
DEFAULT_SYNC_GPU = True
DEFAULT_SUBPROCESS_TIMEOUT = 300  # 5 minutes default


def _subprocess_worker(
    func: Callable,
    args: tuple,
    kwargs: dict,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker function that runs in the subprocess."""
    try:
        result = func(*args, **kwargs)
        result_queue.put({"success": True, "result": result})
    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        })


class InferenceSubprocess:
    """
    Subprocess-based inference execution for memory isolation.

    This class provides a way to run arbitrary callables in a separate process,
    ensuring memory isolation between the main process and the inference process.
    This is particularly useful for:
    - Memory profiling without contaminating the main process
    - Running multiple models sequentially without memory leaks
    - Isolating GPU memory usage

    Example:
        >>> subprocess = InferenceSubprocess(timeout=60)
        >>> def compute():
        ...     return {"result": 42}
        >>> result = subprocess.run(compute)
        >>> print(result)
        {'result': 42}

    Example with arguments:
        >>> def add(a, b):
        ...     return a + b
        >>> result = subprocess.run(add, args=(5, 3))
        >>> print(result)
        8
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize the InferenceSubprocess.

        Args:
            timeout: Maximum time in seconds to wait for subprocess completion.
                     If None, uses DEFAULT_SUBPROCESS_TIMEOUT (300 seconds).
        """
        self.timeout = timeout if timeout is not None else DEFAULT_SUBPROCESS_TIMEOUT
        self._process: Optional[multiprocessing.Process] = None
        self._result_queue: Optional[multiprocessing.Queue] = None

    def run(
        self,
        func: Callable[..., Any],
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run a callable in a subprocess and return its result.

        Args:
            func: The callable to execute in the subprocess.
            args: Positional arguments to pass to the callable.
            kwargs: Keyword arguments to pass to the callable.

        Returns:
            The return value of the callable.

        Raises:
            TimeoutError: If the subprocess exceeds the timeout.
            RuntimeError: If the subprocess encounters an error.
        """
        args = args or ()
        kwargs = kwargs or {}

        # Create a queue for receiving results
        self._result_queue = multiprocessing.Queue()

        # Create and start the subprocess
        self._process = multiprocessing.Process(
            target=_subprocess_worker,
            args=(func, args, kwargs, self._result_queue),
        )
        self._process.start()

        try:
            # Wait for result with timeout
            self._process.join(timeout=self.timeout)

            if self._process.is_alive():
                # Process timed out
                self._process.terminate()
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join()
                raise TimeoutError(
                    f"Subprocess timed out after {self.timeout} seconds"
                )

            # Get result from queue
            if not self._result_queue.empty():
                result_data = self._result_queue.get_nowait()

                if result_data.get("success"):
                    return result_data["result"]
                else:
                    # Re-raise the exception from subprocess
                    error_msg = result_data.get("error", "Unknown error")
                    error_type = result_data.get("error_type", "RuntimeError")
                    tb = result_data.get("traceback", "")

                    # Try to raise the original exception type
                    if error_type == "ValueError":
                        raise ValueError(error_msg)
                    elif error_type == "TypeError":
                        raise TypeError(error_msg)
                    elif error_type == "KeyError":
                        raise KeyError(error_msg)
                    else:
                        raise RuntimeError(f"{error_type}: {error_msg}\n{tb}")
            else:
                # Process finished but no result - likely crashed
                exit_code = self._process.exitcode
                if exit_code != 0:
                    raise RuntimeError(
                        f"Subprocess exited with code {exit_code}"
                    )
                return None

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """
        Clean up subprocess resources.

        This method ensures the subprocess is properly terminated and
        resources are released. It is called automatically after each run,
        but can also be called explicitly if needed.
        """
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join()
            self._process = None

        if self._result_queue is not None:
            try:
                self._result_queue.close()
                self._result_queue.join_thread()
            except Exception:
                pass
            self._result_queue = None

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.cleanup()


def _parse_shape(value: str) -> List[int]:
    if not value:
        return []
    parts = value.replace("x", ",").replace(" ", ",").split(",")
    return [int(p) for p in parts if p.strip()]


def _resolve_device(preferred: str) -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def _load_model_and_tokenizer(
    model_id: str,
    device: str,
    trust_remote_code: bool,
    need_tokenizer: bool,
) -> Tuple[torch.nn.Module, Optional[object]]:
    from transformers import AutoModel, AutoTokenizer

    # Load from local dir if exists; support compressed manifest+safetensors format
    if Path(model_id).exists():
        manifest = Path(model_id) / "manifest.json"
        safetensors_path = Path(model_id) / "model.safetensors"
        if manifest.exists() and safetensors_path.exists():
            from .compressed_io import load_compressed_from_safetensors
            data = json.loads(manifest.read_text())
            base_model = data.get("base_model")
            tokenizer = None
            if need_tokenizer:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                except Exception:
                    if not base_model:
                        raise RuntimeError("Compressed manifest missing base_model; cannot load tokenizer")
                    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
            model = load_compressed_from_safetensors(base_model_name=base_model, save_dir=model_id, device=device, debug=False)
            return model, tokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code) if need_tokenizer else None
        model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.float32)
        return model, tokenizer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    kw = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kw) if need_tokenizer else None
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, torch_dtype=torch.float32, **kw)
    return model, tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile inference in a fresh process")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", required=True, help="Path to write MemoryProfiler JSON report")
    parser.add_argument("--device", default="cpu", help="cpu|cuda|auto")
    parser.add_argument("--phase_name", default="inference")
    parser.add_argument("--input_kind", default="text", choices=["text", "ids", "tensor"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--tensor_shape", default="")
    parser.add_argument("--arg_name", default="inputs")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--text", default="Hello world from Toggle")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sync_gpu", dest="sync_gpu", action="store_true", default=DEFAULT_SYNC_GPU)
    parser.add_argument("--no_sync_gpu", dest="sync_gpu", action="store_false")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    # Set reproducibility seed at subprocess start
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(args.device)
    input_kind = args.input_kind
    need_tokenizer = input_kind == "text"

    # Compatibility shims for some torch/transformers combos
    try:
        import transformers.modeling_utils as _mu
        if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
            _mu.ALL_PARALLEL_STYLES = {"colwise", "rowwise", "sequence", "context"}
    except Exception:
        pass

    model, tokenizer = _load_model_and_tokenizer(
        model_id=args.model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        need_tokenizer=need_tokenizer,
    )

    model = model.to(device)
    model.eval()

    if input_kind == "text":
        if tokenizer is None:
            raise RuntimeError("Tokenizer required for text inputs")
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            args.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif input_kind == "ids":
        vocab_size = 1000
        try:
            emb = model.get_input_embeddings()
            if hasattr(emb, "num_embeddings"):
                vocab_size = int(emb.num_embeddings)
        except Exception:
            pass
        # Use seeded generator for reproducibility
        generator = torch.Generator(device='cpu').manual_seed(seed)
        input_ids = torch.randint(
            low=0,
            high=max(2, vocab_size),
            size=(int(args.batch_size), int(args.seq_len)),
            dtype=torch.long,
            generator=generator,
        ).to(device)
        attention_mask = torch.ones((int(args.batch_size), int(args.seq_len)), dtype=torch.long, device=device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        shape = _parse_shape(args.tensor_shape) or [1, 8]
        # Use seeded generator for reproducibility
        generator = torch.Generator(device='cpu').manual_seed(seed)
        tensor = torch.randn(*shape, dtype=torch.float32, generator=generator).to(device)
        inputs = {str(args.arg_name): tensor}

    def _forward():
        return model(**inputs) if isinstance(inputs, dict) else model(inputs)

    # Warmup
    if args.warmup > 0:
        with torch.inference_mode():
            for _ in range(int(args.warmup)):
                _ = _forward()
                if args.sync_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()

    from .memory_profiler import MemoryProfiler
    profiler = MemoryProfiler()

    with profiler.profile_execution(args.phase_name, sync_gpu=args.sync_gpu) as _p:
        with torch.inference_mode():
            for _ in range(max(1, int(args.repeats))):
                _ = _forward()
                if args.sync_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_report(str(out_path))
    report = json.loads(out_path.read_text())
    report.setdefault("summary", {})
    report["summary"]["isolate_subprocess"] = DEFAULT_MEMORY_INFERENCE_ISOLATE_SUBPROCESS
    report["summary"]["sync_gpu"] = bool(args.sync_gpu)
    report["summary"]["input_kind"] = input_kind
    out_path.write_text(json.dumps(report, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)
