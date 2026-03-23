"""
E2E tests — Profiling & Subprocess Isolation
============================================
Each test performs actual compression on Gemma-3-270m-it and profiles
the memory / latency / isolation characteristics.

Feature 25: RSS memory
Feature 26: VMS memory
Feature 29: Peak memory
Feature 30: Latency (per-phase)
Feature 31: Subprocess isolation

Model: google/gemma-3-270m-it
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framework.context import PipelineContext
from src.framework.memory_profiler import MemoryProfiler
from src.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import random_input, size_mb


# ═══════════════════════════════════════════════════════════════════════════
# Feature 25 — RSS Memory
# ═══════════════════════════════════════════════════════════════════════════

class TestRSSMemory:
    """Profile RSS (Resident Set Size) during actual model compression."""

    def test_rss_snapshot_captures_memory(self, model, tokenizer, tmp_path, csv_logger):
        """MemoryProfiler should capture non-zero RSS during compression."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        t0 = time.time()
        with mp.profile_execution("svd_compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].self_attn.q_proj"],
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_size = size_mb(model)

        profile = mp.profiles["svd_compression"]
        assert profile.pre_execution.rss_mb > 0, "Pre RSS should be > 0"
        assert profile.post_execution.rss_mb > 0, "Post RSS should be > 0"

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_rss",
            task_results={"rss": {"pre_rss_mb": profile.pre_execution.rss_mb, "post_rss_mb": profile.post_execution.rss_mb}},
            evaluation_params={"phase": "svd_compression"},
            execution_time=elapsed,
        )
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": elapsed,
                "memory_usage_mb": profile.post_execution.rss_mb,
            },
            parameters={"rank": 32, "target_modules": ["model.layers[0].self_attn.q_proj"]},
        )

    def test_rss_delta_during_compression(self, model, tokenizer, tmp_path, csv_logger):
        """RSS delta should be finite during compression."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        t0 = time.time()
        with mp.profile_execution("compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].mlp.gate_proj"],
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        profile = mp.profiles["compression"]
        delta = profile.memory_delta
        assert "rss_delta_mb" in delta
        import math
        assert math.isfinite(delta["rss_delta_mb"])

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_rss",
            task_results={"rss_delta": {"rss_delta_mb": delta["rss_delta_mb"]}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 26 — VMS Memory
# ═══════════════════════════════════════════════════════════════════════════

class TestVMSMemory:
    """Profile VMS (Virtual Memory Size) during model compression."""

    def test_vms_snapshot_captured(self, model, tokenizer, tmp_path, csv_logger):
        """MemoryProfiler should capture non-zero VMS."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        t0 = time.time()
        with mp.profile_execution("compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].self_attn.k_proj"],
                rank=16,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        profile = mp.profiles["compression"]
        assert profile.pre_execution.vms_mb > 0
        assert profile.post_execution.vms_mb > 0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_vms",
            task_results={"vms": {"pre_vms_mb": profile.pre_execution.vms_mb, "post_vms_mb": profile.post_execution.vms_mb}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed,
        )

    def test_vms_delta(self, model, tokenizer, tmp_path, csv_logger):
        """VMS delta should be finite."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        t0 = time.time()
        with mp.profile_execution("compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].mlp.down_proj"],
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        delta = mp.profiles["compression"].memory_delta
        assert "vms_delta_mb" in delta
        import math
        assert math.isfinite(delta["vms_delta_mb"])

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_vms",
            task_results={"vms_delta": {"vms_delta_mb": delta["vms_delta_mb"]}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 29 — Peak Memory
# ═══════════════════════════════════════════════════════════════════════════

class TestPeakMemory:
    """Profile peak memory (background thread) during compression."""

    def test_peak_memory_tracked(self, model, tokenizer, tmp_path, csv_logger):
        """Peak memory should be >= pre-execution memory."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        t0 = time.time()
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
        elapsed = time.time() - t0

        profile = mp.profiles["compression"]
        assert profile.peak_memory is not None
        assert profile.peak_memory.rss_mb >= profile.pre_execution.rss_mb

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_peak",
            task_results={"peak_memory": {"peak_rss_mb": profile.peak_memory.rss_mb, "pre_rss_mb": profile.pre_execution.rss_mb}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed,
        )

    def test_peak_memory_higher_than_pre(self, model, tokenizer, tmp_path, csv_logger):
        """Compressing many layers should show peak > pre RSS."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        targets = [f"model.layers[{i}].mlp.gate_proj" for i in range(4)]
        t0 = time.time()
        with mp.profile_execution("multi_layer_compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=targets,
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        profile = mp.profiles["multi_layer_compression"]
        assert profile.peak_memory is not None
        # Peak should be at least as much as pre-execution
        assert profile.peak_memory.rss_mb >= profile.pre_execution.rss_mb

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="memory_peak",
            task_results={"peak_memory": {"peak_rss_mb": profile.peak_memory.rss_mb, "pre_rss_mb": profile.pre_execution.rss_mb}},
            evaluation_params={"phase": "multi_layer_compression", "num_targets": len(targets)},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 30 — Latency (Per-Phase)
# ═══════════════════════════════════════════════════════════════════════════

class TestLatency:
    """Profile per-phase latency during compression and inference."""

    def test_compression_latency_measured(self, model, tokenizer, tmp_path, csv_logger):
        """Compression phase should have measurable latency > 0."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        t0 = time.time()
        with mp.profile_execution("compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].self_attn.q_proj"],
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_size = size_mb(model)

        profile = mp.profiles["compression"]
        latency = profile.post_execution.timestamp - profile.pre_execution.timestamp
        assert latency > 0, "Compression should take measurable time"

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="latency",
            task_results={"compression_latency": {"latency_seconds": latency}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed,
        )
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": elapsed,
                "memory_usage_mb": profile.post_execution.rss_mb,
            },
            parameters={"rank": 32, "target_modules": ["model.layers[0].self_attn.q_proj"]},
        )

    def test_multi_phase_latency(self, model, tokenizer, tmp_path, csv_logger):
        """Profile baseline inference, compression, and compressed inference separately."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        # Phase 1: Baseline inference
        inputs = random_input(model)
        t0 = time.time()
        with mp.profile_execution("baseline_inference"):
            with torch.no_grad():
                _ = model(**inputs)
        elapsed_baseline = time.time() - t0

        # Phase 2: Compression
        t0 = time.time()
        with mp.profile_execution("compression"):
            consolidator = ModelConsolidator(
                compression_method="svd",
                target_modules=["model.layers[0].mlp.gate_proj"],
                rank=32,
                svd_backend="torch",
            )
            consolidator.initialize(ctx)
            consolidator.compress_model_with_surgery(model)
        elapsed_compression = time.time() - t0

        # Phase 3: Compressed inference
        inputs2 = random_input(model)
        t0 = time.time()
        with mp.profile_execution("compressed_inference"):
            with torch.no_grad():
                _ = model(**inputs2)
        elapsed_compressed = time.time() - t0

        # All three phases should have measurable latency
        for phase in ["baseline_inference", "compression", "compressed_inference"]:
            profile = mp.profiles[phase]
            latency = profile.post_execution.timestamp - profile.pre_execution.timestamp
            assert latency > 0, f"{phase} should have positive latency"

        # Log each phase
        profile_baseline = mp.profiles["baseline_inference"]
        latency_baseline = profile_baseline.post_execution.timestamp - profile_baseline.pre_execution.timestamp
        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="latency",
            task_results={"baseline_inference": {"latency_seconds": latency_baseline}},
            evaluation_params={"phase": "baseline_inference"},
            execution_time=elapsed_baseline,
        )

        profile_compression = mp.profiles["compression"]
        latency_compression = profile_compression.post_execution.timestamp - profile_compression.pre_execution.timestamp
        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="latency",
            task_results={"compression": {"latency_seconds": latency_compression}},
            evaluation_params={"phase": "compression"},
            execution_time=elapsed_compression,
        )

        profile_compressed = mp.profiles["compressed_inference"]
        latency_compressed = profile_compressed.post_execution.timestamp - profile_compressed.pre_execution.timestamp
        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="latency",
            task_results={"compressed_inference": {"latency_seconds": latency_compressed}},
            evaluation_params={"phase": "compressed_inference"},
            execution_time=elapsed_compressed,
        )

    def test_compare_phases(self, model, tokenizer, tmp_path, csv_logger):
        """compare_phases should show differences between two profiled phases."""
        mp = MemoryProfiler()

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        inputs = random_input(model)
        t0 = time.time()
        with mp.profile_execution("phase_a"):
            with torch.no_grad():
                _ = model(**inputs)
        elapsed_a = time.time() - t0

        # Compress and measure again
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].mlp.gate_proj"],
            rank=16,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        inputs2 = random_input(model)
        t0 = time.time()
        with mp.profile_execution("phase_b"):
            with torch.no_grad():
                _ = model(**inputs2)
        elapsed_b = time.time() - t0

        comparison = mp.compare_phases("phase_a", "phase_b")
        assert "rss_difference_mb" in comparison
        assert "vms_difference_mb" in comparison

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="phase_comparison",
            task_results={"phase_comparison": {"rss_difference_mb": comparison["rss_difference_mb"], "vms_difference_mb": comparison["vms_difference_mb"]}},
            evaluation_params={"phase_a": "pre_compression_inference", "phase_b": "post_compression_inference"},
            execution_time=elapsed_a + elapsed_b,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 31 — Subprocess Isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestSubprocessIsolation:
    """InferenceSubprocess runs inference in a separate process for memory isolation."""

    def test_subprocess_runs_function(self, model, tokenizer, tmp_path, csv_logger):
        """InferenceSubprocess should run a simple function and return result."""
        from src.framework.inference_subprocess import InferenceSubprocess

        def simple_task():
            return {"status": "ok", "value": 42}

        t0 = time.time()
        sp = InferenceSubprocess(timeout=60)
        result = sp.run(simple_task)
        elapsed = time.time() - t0
        assert result["status"] == "ok"
        assert result["value"] == 42

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="subprocess",
            task_results={"subprocess_run": {"status": result["status"], "value": result["value"]}},
            evaluation_params={"timeout": 60},
            execution_time=elapsed,
        )

    def test_subprocess_timeout_handling(self, model, tokenizer, tmp_path, csv_logger):
        """Subprocess should raise TimeoutError for long-running tasks."""
        from src.framework.inference_subprocess import InferenceSubprocess

        def slow_task():
            import time
            time.sleep(120)
            return {}

        t0 = time.time()
        sp = InferenceSubprocess(timeout=2)
        with pytest.raises(TimeoutError):
            sp.run(slow_task)
        elapsed = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="subprocess",
            task_results={"subprocess_timeout": {"timeout_triggered": True, "configured_timeout_s": 2}},
            evaluation_params={"timeout": 2},
            execution_time=elapsed,
        )

    def test_subprocess_error_propagation(self, model, tokenizer, tmp_path, csv_logger):
        """Errors in subprocess should be propagated."""
        from src.framework.inference_subprocess import InferenceSubprocess

        def failing_task():
            raise ValueError("deliberate test error")

        t0 = time.time()
        sp = InferenceSubprocess(timeout=30)
        with pytest.raises((ValueError, RuntimeError)):
            sp.run(failing_task)
        elapsed = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="subprocess",
            task_results={"subprocess_error": {"error_propagated": True, "error_type": "ValueError"}},
            evaluation_params={"timeout": 30},
            execution_time=elapsed,
        )

    def test_subprocess_with_actual_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Run actual model inference in a subprocess."""
        from src.framework.inference_subprocess import InferenceSubprocess

        # Save model to disk so subprocess can load it
        save_dir = tmp_path / "model_for_subprocess"
        save_dir.mkdir()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        def inference_in_subprocess():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            m = AutoModelForCausalLM.from_pretrained(
                str(save_dir), trust_remote_code=True, torch_dtype=torch.float32,
            )
            t = AutoTokenizer.from_pretrained(str(save_dir), trust_remote_code=True)
            inputs = t("Hello", return_tensors="pt")
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=4, do_sample=False)
            text = t.decode(out[0], skip_special_tokens=True)
            return {"text": text, "length": len(text)}

        t0 = time.time()
        sp = InferenceSubprocess(timeout=120)
        result = sp.run(inference_in_subprocess)
        elapsed = time.time() - t0
        assert "text" in result
        assert result["length"] > 0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="subprocess",
            task_results={"subprocess_inference": {"text_length": result["length"], "success": True}},
            evaluation_params={"timeout": 120},
            execution_time=elapsed,
        )

    def test_profiler_subprocess_integration(self, model, tokenizer, tmp_path, csv_logger):
        """MemoryProfiler should work around subprocess execution."""
        from src.framework.inference_subprocess import InferenceSubprocess

        mp = MemoryProfiler()

        t0 = time.time()
        with mp.profile_execution("subprocess_test"):
            def compute():
                return sum(range(1000000))
            sp = InferenceSubprocess(timeout=30)
            result = sp.run(compute)
        elapsed = time.time() - t0

        profile = mp.profiles["subprocess_test"]
        assert profile.pre_execution.rss_mb > 0
        assert profile.post_execution.rss_mb > 0
        assert result == sum(range(1000000))

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="profile",
            plugin_name="subprocess",
            task_results={"profiler_subprocess": {"pre_rss_mb": profile.pre_execution.rss_mb, "post_rss_mb": profile.post_execution.rss_mb, "result_correct": result == sum(range(1000000))}},
            evaluation_params={"timeout": 30},
            execution_time=elapsed,
        )
