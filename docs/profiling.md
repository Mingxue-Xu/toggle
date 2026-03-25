# Profiling in the Goldcrest Compression Pipeline

This document describes how to profile memory usage, latency, and process
isolation during model compression using Goldcrest's profiling infrastructure.
Six capabilities are covered:

1. **RSS memory** — resident set size snapshots and deltas
2. **VMS memory** — virtual memory size snapshots and deltas
3. **GPU memory** — CUDA allocated/reserved snapshots, deltas, and aliases
4. **Peak memory** — background-sampled peak RSS/VMS plus CUDA peak counters
5. **Latency** — per-phase wall-clock timing and phase comparison
6. **Subprocess isolation** — running inference in a separate process for
   memory-isolated profiling

All profiling is built around `MemoryProfiler` (a context-manager-based
profiler) and `InferenceSubprocess` (a multiprocessing-based executor). The
serialized GPU reserved-memory field is still named `gpu_cached_mb` for
backward compatibility; `MemorySnapshot.gpu_reserved` exposes the same value
through reserved terminology.

---

## Background: VMS vs RSS

VMS (Virtual Memory Size) includes all memory the process has mapped -- code, heap, stack, memory-mapped files, shared libraries, and pages that have been allocated but never touched or swapped out. RSS (Resident Set Size) counts only the pages physically in RAM right now.

VMS > RSS because:

1. **Allocated but untouched pages** -- `malloc`/`mmap` reserves virtual address space, but the kernel doesn't back it with physical pages until first access (demand paging).
2. **Swapped-out pages** -- pages moved to swap count in VMS but not RSS.
3. **Memory-mapped files** -- the full file size counts toward VMS, but only the accessed portions are in RSS.
4. **Shared libraries** -- the entire library is mapped (VMS), but only used pages are resident (RSS).

VMS >= RSS is essentially always true. The gap is especially large with PyTorch/CUDA since the GPU runtime reserves large virtual address ranges upfront.

For CUDA runs, `MemoryProfiler` also records:

- `torch.cuda.memory_allocated()` as `gpu_allocated_mb`
- `torch.cuda.memory_reserved()` as `gpu_cached_mb`

The object-level alias `gpu_reserved` points to `gpu_cached_mb`.

---

## Prerequisites

```python
import time
import torch
from goldcrest.framework.context import PipelineContext
from goldcrest.framework.memory_profiler import MemoryProfiler
from goldcrest.plugins.compression.consolidator import ModelConsolidator
```

Load a model and create a pipeline context (shared across all examples):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it",
    torch_dtype=torch.float32,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

ctx = PipelineContext(config={}, workspace_dir="workspace/profiling")
ctx.state.model = model
ctx.state.tokenizer = tokenizer
```

---

## 1. RSS Memory Profiling

### Snapshot before and after compression

```python
mp = MemoryProfiler()

with mp.profile_execution("svd_compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].self_attn.q_proj"],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["svd_compression"]
print(f"Pre RSS:  {profile.pre_execution.rss_mb:.1f} MB")
print(f"Post RSS: {profile.post_execution.rss_mb:.1f} MB")
assert profile.pre_execution.rss_mb > 0
assert profile.post_execution.rss_mb > 0
```

### RSS delta

The `memory_delta` property computes the difference between post and pre
snapshots:

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].mlp.gate_proj"],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

delta = mp.profiles["compression"].memory_delta
print(f"RSS delta: {delta['rss_delta_mb']:.2f} MB")

import math
assert math.isfinite(delta["rss_delta_mb"])
```

`memory_delta` returns:

| Key | Description |
|---|---|
| `rss_delta_mb` | Post RSS - Pre RSS |
| `vms_delta_mb` | Post VMS - Pre VMS |
| `gpu_allocated_delta_mb` | Post GPU allocated - Pre GPU allocated |
| `gpu_cached_delta_mb` | Post GPU reserved/cached - Pre GPU reserved/cached |

GPU delta keys are always present. On CPU-only runs they remain `0.0`.

**Source:** `tests/e2e/test_e2e_profiling.py`, class `TestRSSMemory`

---

## 2. VMS Memory Profiling

VMS snapshots work identically to RSS. The same `MemoryProfile` object exposes
both:

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].self_attn.k_proj"],
        rank=16,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["compression"]
print(f"Pre VMS:  {profile.pre_execution.vms_mb:.1f} MB")
print(f"Post VMS: {profile.post_execution.vms_mb:.1f} MB")
assert profile.pre_execution.vms_mb > 0
assert profile.post_execution.vms_mb > 0
```

### VMS delta

```python
delta = mp.profiles["compression"].memory_delta
print(f"VMS delta: {delta['vms_delta_mb']:.2f} MB")

import math
assert math.isfinite(delta["vms_delta_mb"])
```

**Source:** `tests/e2e/test_e2e_profiling.py`, class `TestVMSMemory`

---

## 3. GPU Memory Profiling

On CUDA builds, each `MemorySnapshot` records both allocator usage counters.
The implementation maps `torch.cuda.memory_allocated()` to
`gpu_allocated_mb` and `torch.cuda.memory_reserved()` to `gpu_cached_mb`. The
property `MemorySnapshot.gpu_reserved` is a read-only alias for
`gpu_cached_mb`.

That means every snapshot object in a profile (`pre_execution`,
`post_execution`, and `peak_memory` when present) carries the same GPU pair:
allocated memory as `gpu_allocated_mb` and reserved memory as
`gpu_cached_mb`/`gpu_reserved`.

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].self_attn.k_proj"],
        rank=16,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["compression"]
print(f"Pre GPU allocated:  {profile.pre_execution.gpu_allocated_mb:.1f} MB")
print(f"Pre GPU reserved:   {profile.pre_execution.gpu_reserved:.1f} MB")
print(f"Post GPU allocated: {profile.post_execution.gpu_allocated_mb:.1f} MB")
print(f"Post GPU reserved:  {profile.post_execution.gpu_reserved:.1f} MB")

delta = profile.memory_delta
print(f"GPU allocated delta: {delta['gpu_allocated_delta_mb']:.2f} MB")
print(f"GPU reserved delta:  {delta['gpu_cached_delta_mb']:.2f} MB")
```

Notes:

- JSON reports and `memory_delta` dictionaries use the stored key names `gpu_allocated_mb` / `gpu_cached_mb` and `gpu_allocated_delta_mb` / `gpu_cached_delta_mb`.
- `gpu_reserved` is an object-level alias only; there is no separate `gpu_reserved_mb` or `gpu_reserved_delta_mb` key in serialized reports.
- `MemorySnapshot.create(gpu_allocated=..., gpu_reserved=...)` maps those friendly names back to the serialized fields `gpu_allocated_mb` and `gpu_cached_mb`.
- With `cleanup_before=True` (the default), `profile_execution` runs `torch.cuda.empty_cache()` before the pre-execution snapshot.
- On CPU-only runs, all GPU fields remain `0.0`.

**Source:** `goldcrest/framework/memory_profiler.py`, `tests/e2e-codex/test_evaluation_and_profiling.py`

---

## 4. Peak Memory Tracking

`MemoryProfiler` runs a background sampling thread during `profile_execution`.
The thread periodically snapshots RSS/VMS and records the highest sampled CPU
snapshot. On CUDA, `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` are folded into
`peak_memory.gpu_allocated_mb` and `peak_memory.gpu_cached_mb`.

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=[
            "model.layers[0].self_attn.q_proj",
            "model.layers[0].mlp.gate_proj",
        ],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["compression"]
assert profile.peak_memory is not None
assert profile.peak_memory.rss_mb >= profile.pre_execution.rss_mb
print(f"Peak RSS: {profile.peak_memory.rss_mb:.1f} MB")
print(f"Pre RSS:  {profile.pre_execution.rss_mb:.1f} MB")
print(f"Peak GPU allocated: {profile.peak_memory.gpu_allocated_mb:.1f} MB")
print(f"Peak GPU reserved:  {profile.peak_memory.gpu_reserved:.1f} MB")
```

CPU peak sampling and CUDA peak counters do not have to come from the same
instant. CPU peaks come from the background sampler, while GPU peaks use CUDA's
max-memory counters.

### Multi-layer compression peak

Compressing more layers produces a higher peak because intermediate SVD
temporaries accumulate:

```python
mp = MemoryProfiler()

targets = [f"model.layers[{i}].mlp.gate_proj" for i in range(4)]
with mp.profile_execution("multi_layer_compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=targets,
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["multi_layer_compression"]
assert profile.peak_memory.rss_mb >= profile.pre_execution.rss_mb
```

**Source:** `tests/e2e/test_e2e_profiling.py`, class `TestPeakMemory`

---

## 5. Latency Profiling

### Single-phase latency

Latency is derived from the timestamp difference between pre and post snapshots:

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].self_attn.q_proj"],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

profile = mp.profiles["compression"]
latency = profile.post_execution.timestamp - profile.pre_execution.timestamp
print(f"Compression latency: {latency:.3f} s")
assert latency > 0
```

### Multi-phase latency (baseline → compress → compressed inference)

Profile multiple phases independently in the same `MemoryProfiler` instance:

```python
mp = MemoryProfiler()

# Phase 1: Baseline inference
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
with mp.profile_execution("baseline_inference"):
    with torch.no_grad():
        _ = model(**inputs)

# Phase 2: Compression
with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].mlp.gate_proj"],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

# Phase 3: Compressed inference
inputs2 = tokenizer("Hello", return_tensors="pt").to(model.device)
with mp.profile_execution("compressed_inference"):
    with torch.no_grad():
        _ = model(**inputs2)

for phase in ["baseline_inference", "compression", "compressed_inference"]:
    profile = mp.profiles[phase]
    latency = profile.post_execution.timestamp - profile.pre_execution.timestamp
    print(f"{phase}: {latency:.3f} s")
    assert latency > 0
```

### Phase comparison

`compare_phases` computes the difference in memory deltas between two profiled
phases:

```python
mp = MemoryProfiler()

inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
with mp.profile_execution("phase_a"):
    with torch.no_grad():
        _ = model(**inputs)

# Compress between phases
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[0].mlp.gate_proj"],
    rank=16,
    svd_backend="torch",
)
consolidator.initialize(ctx)
consolidator.compress_model_with_surgery(model)

inputs2 = tokenizer("Hello", return_tensors="pt").to(model.device)
with mp.profile_execution("phase_b"):
    with torch.no_grad():
        _ = model(**inputs2)

comparison = mp.compare_phases("phase_a", "phase_b")
print(f"RSS difference: {comparison['rss_difference_mb']:.2f} MB")
print(f"VMS difference: {comparison['vms_difference_mb']:.2f} MB")
```

`compare_phases` returns:

| Key | Description |
|---|---|
| `rss_difference_mb` | phase2 RSS delta - phase1 RSS delta |
| `vms_difference_mb` | phase2 VMS delta - phase1 VMS delta |
| `gpu_allocated_difference_mb` | GPU allocated delta difference |
| `gpu_cached_difference_mb` | GPU reserved/cached delta difference |
| `phase1_total_delta_mb` | phase1 RSS delta |
| `phase2_total_delta_mb` | phase2 RSS delta |

**Source:** `tests/e2e/test_e2e_profiling.py`, class `TestLatency`

---

## 6. Subprocess Isolation

`InferenceSubprocess` runs an arbitrary callable in a separate
`multiprocessing.Process` so that memory usage is fully isolated from the parent.
This is useful for accurate memory profiling without contamination.

### Basic usage

```python
from goldcrest.framework.inference_subprocess import InferenceSubprocess

def simple_task():
    return {"status": "ok", "value": 42}

sp = InferenceSubprocess(timeout=60)
result = sp.run(simple_task)
assert result["status"] == "ok"
assert result["value"] == 42
```

### Timeout handling

If the callable exceeds the timeout, a `TimeoutError` is raised:

```python
def slow_task():
    import time
    time.sleep(120)
    return {}

sp = InferenceSubprocess(timeout=2)
try:
    sp.run(slow_task)
except TimeoutError:
    print("Subprocess timed out as expected")
```

### Error propagation

Exceptions raised inside the subprocess are re-raised in the parent:

```python
def failing_task():
    raise ValueError("deliberate test error")

sp = InferenceSubprocess(timeout=30)
try:
    sp.run(failing_task)
except (ValueError, RuntimeError) as e:
    print(f"Error propagated: {e}")
```

The original exception type is preserved for `ValueError`, `TypeError`, and
`KeyError`. Other exception types are wrapped in `RuntimeError` with the
original traceback.

### Running model inference in a subprocess

Save the model to disk, then load and run inference inside the subprocess:

```python
save_dir = "workspace/model_for_subprocess"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

def inference_in_subprocess():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(
        save_dir, trust_remote_code=True, torch_dtype=torch.float32,
    )
    t = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    inputs = t("Hello", return_tensors="pt")
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=4, do_sample=False)
    text = t.decode(out[0], skip_special_tokens=True)
    return {"text": text, "length": len(text)}

sp = InferenceSubprocess(timeout=120)
result = sp.run(inference_in_subprocess)
assert "text" in result
assert result["length"] > 0
```

### Combining MemoryProfiler with InferenceSubprocess

Wrap subprocess execution inside `profile_execution` to capture the parent
process memory footprint while the subprocess runs:

```python
mp = MemoryProfiler()

with mp.profile_execution("subprocess_test"):
    def compute():
        return sum(range(1000000))
    sp = InferenceSubprocess(timeout=30)
    result = sp.run(compute)

profile = mp.profiles["subprocess_test"]
assert profile.pre_execution.rss_mb > 0
assert profile.post_execution.rss_mb > 0
print(f"Result: {result}")
```

**Source:** `tests/e2e/test_e2e_profiling.py`, class `TestSubprocessIsolation`

---

## MemoryProfiler API Reference

```python
class MemoryProfiler:
    def __init__(self, isolate: bool = False): ...

    profiles: Dict[str, MemoryProfile]     # phase_name -> profile
    baseline_memory: MemorySnapshot         # snapshot at init time

    def profile_execution(
        self,
        phase_name: str,
        cleanup_before: bool = True,
        sync_gpu: bool = True,
        sample_interval: float = 0.02,
        track_cuda_peak: bool = True,
    ) -> ContextManager: ...

    def profile(
        self,
        phase_name: str,
        cleanup_before: bool = True,
        sync_gpu: bool = True,
    ) -> ContextManager: ...                # alias for profile_execution

    def get_profile(self, phase_name: str) -> Optional[MemoryProfile]: ...
    def compare_phases(self, phase1: str, phase2: str) -> Dict[str, float]: ...
    def get_memory_efficiency_score(self, phase_name: str) -> float: ...
    def export_report(self, output_path: str) -> None: ...
    def import_report(self, report: Dict[str, Any], replace: bool = True) -> None: ...
    def load_report(self, report_path: Union[str, Path], replace: bool = True) -> Dict[str, Any]: ...
    def profile_execution_subprocess(
        self,
        command: List[str],
        report_path: Union[str, Path],
        phase_name: Optional[str] = None,
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        replace: bool = True,
    ) -> MemoryProfile: ...
    def print_summary(self) -> None: ...
    def clear_profiles(self) -> None: ...
```

### MemorySnapshot fields

| Field | Type | Description |
|---|---|---|
| `rss_mb` | `float` | Resident Set Size in MB |
| `vms_mb` | `float` | Virtual Memory Size in MB |
| `gpu_allocated_mb` | `float` | GPU memory allocated in MB |
| `gpu_cached_mb` | `float` | Backward-compatible field name for GPU reserved/cached memory in MB |
| `timestamp` | `float` | `time.time()` at capture |
| `phase_name` | `Optional[str]` | Label for this snapshot |

Aliases: `cpu_used` → `rss_mb`, `gpu_allocated` → `gpu_allocated_mb`,
`gpu_reserved` → `gpu_cached_mb`.

JSON reports serialize the field names above; aliases are convenience
properties only.

### MemoryProfile fields

| Field | Type | Description |
|---|---|---|
| `phase_name` | `str` | Name of the profiled phase |
| `pre_execution` | `MemorySnapshot` | Snapshot taken before the phase |
| `post_execution` | `MemorySnapshot` | Snapshot taken after the phase |
| `peak_memory` | `Optional[MemorySnapshot]` | Highest sampled RSS/VMS snapshot, augmented with CUDA peak allocated/reserved counters when available |
| `memory_delta` | `Dict[str, float]` | Post - Pre for all metrics (property) |

---

## InferenceSubprocess API Reference

```python
class InferenceSubprocess:
    def __init__(self, timeout: Optional[float] = None): ...
        # Default timeout: 300 seconds

    def run(
        self,
        func: Callable[..., Any],
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any: ...
        # Raises TimeoutError or RuntimeError on failure

    def cleanup(self) -> None: ...
```

---

## Exporting and Importing Reports

`MemoryProfiler` can serialise all profiles to a JSON report and reload them
later:

```python
mp = MemoryProfiler()

with mp.profile_execution("compression"):
    consolidator = ModelConsolidator(
        compression_method="svd",
        target_modules=["model.layers[0].self_attn.q_proj"],
        rank=32,
        svd_backend="torch",
    )
    consolidator.initialize(ctx)
    consolidator.compress_model_with_surgery(model)

# Export
mp.export_report("workspace/profiling/report.json")

# Import into a fresh profiler
mp2 = MemoryProfiler()
mp2.load_report("workspace/profiling/report.json")
assert "compression" in mp2.profiles
```

The JSON report contains:

```json
{
  "baseline_memory": {
    "rss_mb": ...,
    "vms_mb": ...,
    "gpu_allocated_mb": ...,
    "gpu_cached_mb": ...,
    ...
  },
  "execution_profiles": {
    "compression": {
      "phase_name": "compression",
      "pre_execution": { ... },
      "post_execution": { ... },
      "peak_memory": { ... },
      "memory_delta": { ... }
    }
  },
  "summary": {
    "total_phases": 1,
    "total_memory_delta_mb": ...,
    "peak_memory_usage_mb": ...,
    "baseline_memory_mb": ...,
    "average_efficiency_score": ...,
    "most_efficient_phase": "compression",
    "least_efficient_phase": "compression"
  }
}
```

On disk, reserved GPU memory is serialized as `gpu_cached_mb`; the
`gpu_reserved` alias is not emitted into JSON.

### Subprocess-emitted reports

For full isolation, run profiling in a subprocess via CLI and load the report:

```python
import sys

mp = MemoryProfiler()
profile = mp.profile_execution_subprocess(
    command=[
        sys.executable, "scripts/utils/profile_inference_subprocess.py",
        "--model", "google/gemma-3-270m-it",
        "--output", "workspace/profiling/subprocess_report.json",
        "--device", "cpu",
        "--phase_name", "inference",
    ],
    report_path="workspace/profiling/subprocess_report.json",
)
print(f"Subprocess RSS: {profile.post_execution.rss_mb:.1f} MB")
```

The subprocess runner appends `summary.isolate_subprocess`,
`summary.sync_gpu`, and `summary.input_kind` to the emitted report.
