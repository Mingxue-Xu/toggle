"""
E2E tests — Evaluation (Features 23–24)
=========================================
Each test compresses the model then evaluates it using actual evaluation
plugins on real Gemma-3-270m-it.

Feature 23: lm-eval-harness integration (LMHarness plugin)
Feature 24: Self-defined language task evaluation (LMEvaluator plugin)

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
from src.framework.eval_interface import ModelEvalInterface
from src.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import param_count, run_forward, size_mb, DEVICE, MODEL_ID


# ═══════════════════════════════════════════════════════════════════════════
# Feature 23 — lm-eval-harness Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestLMHarness:
    """Evaluate compressed model with lm-eval-harness."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_lm_eval(self):
        try:
            import lm_eval
        except ImportError:
            pytest.skip("lm-eval-harness not installed")

    def test_lm_harness_on_compressed_model(self, model, tokenizer, tmp_path, csv_logger):
        """Compress → evaluate with lm-eval-harness on arc_easy (limit=4)."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        # Compress a few layers first
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[0].self_attn.q_proj"], "rank": 64, "svd_backend": "torch"},
        )

        # Evaluate with lm-eval-harness
        from src.plugins.evaluation.lm_eval import LMHarness

        evaluator = LMHarness(
            tasks=["arc_easy"],
            batch_size=1,
            max_length=256,
            device=str(model.device),
            limit=4,
            backend="auto",
            name="lm_eval",
        )
        evaluator.initialize(ctx)
        t0 = time.time()
        result = evaluator.execute()
        eval_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="lm_eval_harness",
            task_results={"arc_easy": {"result": str(result)}},
            evaluation_params={"tasks": ["arc_easy"], "limit": 4},
            execution_time=eval_time,
        )

        assert result is not None
        # Should have arc_easy results
        assert "arc_easy" in str(result)

    def test_lm_harness_on_uncompressed_model(self, model, tokenizer, tmp_path, csv_logger):
        """Evaluate the uncompressed model as a baseline with lm-eval-harness."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        from src.plugins.evaluation.lm_eval import LMHarness

        evaluator = LMHarness(
            tasks=["arc_easy"],
            batch_size=1,
            max_length=256,
            device=str(model.device),
            limit=4,
            backend="auto",
            name="lm_eval",
        )
        evaluator.initialize(ctx)
        t0 = time.time()
        result = evaluator.execute()
        eval_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="baseline",
            evaluation_type="lm_eval",
            plugin_name="lm_eval_harness",
            task_results={"arc_easy": {"result": str(result)}},
            evaluation_params={"tasks": ["arc_easy"], "limit": 4},
            execution_time=eval_time,
        )

        assert result is not None

    def test_lm_harness_multiple_tasks(self, model, tokenizer, tmp_path, csv_logger):
        """Evaluate with multiple tasks (arc_easy, hellaswag) limited to 2 samples."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        # Compress first
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].mlp.gate_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[0].mlp.gate_proj"], "rank": 32, "svd_backend": "torch"},
        )

        from src.plugins.evaluation.lm_eval import LMHarness

        evaluator = LMHarness(
            tasks=["arc_easy", "hellaswag"],
            batch_size=1,
            max_length=256,
            device=str(model.device),
            limit=2,
            backend="auto",
            name="lm_eval",
        )
        evaluator.initialize(ctx)
        t0 = time.time()
        result = evaluator.execute()
        eval_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="lm_eval_harness",
            task_results={
                "arc_easy": {"result": str(result)},
                "hellaswag": {"result": str(result)},
            },
            evaluation_params={"tasks": ["arc_easy", "hellaswag"], "limit": 2},
            execution_time=eval_time,
        )

        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# Feature 24 — Self-Defined Language Task Evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TestSelfDefinedEval:
    """LMEvaluator with custom perplexity and classification tasks."""

    def test_perplexity_on_compressed_model(self, model, tokenizer, tmp_path, csv_logger):
        """Compute perplexity on compressed model using ModelEvalInterface."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        # Compress
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[0].self_attn.q_proj"], "rank": 64, "svd_backend": "torch"},
        )

        # Compute perplexity directly via ModelEvalInterface
        eval_interface = ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=str(model.device),
        )

        test_texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        t0 = time.time()
        results = eval_interface.loglikelihood_rolling(test_texts)
        eval_time = time.time() - t0

        task_results = {}
        for i, r in enumerate(results):
            ll = r[0] if isinstance(r, tuple) else r
            task_results[f"text_{i}"] = {"rolling_ll": ll}

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="perplexity",
            task_results=task_results,
            evaluation_params={"test_texts": test_texts},
            execution_time=eval_time,
        )

        assert len(results) == len(test_texts)
        for r in results:
            ll = r[0] if isinstance(r, tuple) else r
            assert isinstance(ll, float)
            assert ll < 0, "Log-likelihood should be negative"

    def test_loglikelihood_on_compressed_model(self, model, tokenizer, tmp_path, csv_logger):
        """Compute log-likelihood of continuations on compressed model."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        # Compress
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].mlp.down_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[0].mlp.down_proj"], "rank": 32, "svd_backend": "torch"},
        )

        eval_interface = ModelEvalInterface(
            model=model,
            tokenizer=tokenizer,
            device=str(model.device),
        )

        requests = [
            ("The sky is", " blue"),
            ("Water is", " wet"),
        ]
        t0 = time.time()
        results = eval_interface.loglikelihood(requests)
        eval_time = time.time() - t0

        task_results = {}
        for i, r in enumerate(results):
            ll = r[0] if isinstance(r, tuple) else r
            task_results[f"request_{i}"] = {"ll": ll}

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="loglikelihood",
            task_results=task_results,
            evaluation_params={"requests": [list(req) for req in requests]},
            execution_time=eval_time,
        )

        assert len(results) == 2
        for r in results:
            ll = r[0] if isinstance(r, tuple) else r
            assert isinstance(ll, float)

    def test_perplexity_comparison_baseline_vs_compressed(self, model, tokenizer, tmp_path, csv_logger):
        """Compare perplexity before and after compression."""
        import os
        from transformers import AutoModelForCausalLM
        from tests.e2e.conftest import DTYPE, DEVICE, MODEL_ID

        test_texts = ["The cat sat on the mat and looked around."]

        # Baseline perplexity
        eval_baseline = ModelEvalInterface(
            model=model, tokenizer=tokenizer, device=str(model.device),
        )
        t0 = time.time()
        baseline_results = eval_baseline.loglikelihood_rolling(test_texts)
        baseline_eval_time = time.time() - t0
        baseline_ll = baseline_results[0][0] if isinstance(baseline_results[0], tuple) else baseline_results[0]

        csv_logger.log_evaluation_results(
            model_type="baseline",
            evaluation_type="lm_eval",
            plugin_name="perplexity",
            task_results={"text_0": {"rolling_ll": baseline_ll}},
            evaluation_params={"test_texts": test_texts},
            execution_time=baseline_eval_time,
        )

        original_size = size_mb(model)

        # Compress
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[*].mlp.gate_proj"],
            rank=16,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[*].mlp.gate_proj"], "rank": 16, "svd_backend": "torch"},
        )

        # Compressed perplexity
        eval_compressed = ModelEvalInterface(
            model=model, tokenizer=tokenizer, device=str(model.device),
        )
        t0 = time.time()
        compressed_results = eval_compressed.loglikelihood_rolling(test_texts)
        compressed_eval_time = time.time() - t0
        compressed_ll = compressed_results[0][0] if isinstance(compressed_results[0], tuple) else compressed_results[0]

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="perplexity",
            task_results={"text_0": {"rolling_ll": compressed_ll}},
            evaluation_params={"test_texts": test_texts},
            execution_time=compressed_eval_time,
        )

        # Both should be finite negative numbers
        assert baseline_ll < 0, "Baseline LL should be negative"
        assert compressed_ll < 0, "Compressed LL should be negative"
        # Compressed model may have slightly worse (lower) log-likelihood
        # But both should be finite
        import math
        assert math.isfinite(baseline_ll)
        assert math.isfinite(compressed_ll)

    def test_lm_evaluator_classification(self, model, tokenizer, tmp_path, csv_logger):
        """LMEvaluator with a synthetic classification task."""
        from src.plugins.evaluation.lm_eval import LMEvaluator

        ctx = PipelineContext(
            config={
                "evaluation": {
                    "device": str(model.device),
                    "batch_size": 1,
                    "max_length": 128,
                },
            },
            workspace_dir=tmp_path,
        )
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_size = size_mb(model)

        # Compress
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.v_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"target_modules": ["model.layers[0].self_attn.v_proj"], "rank": 32, "svd_backend": "torch"},
        )

        # Run evaluator with a small task
        evaluator = LMEvaluator(tasks=["wikitext"], name="self_eval")
        evaluator.initialize(ctx)
        t0 = time.time()
        result = evaluator.execute()
        eval_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed",
            evaluation_type="lm_eval",
            plugin_name="lm_evaluator",
            task_results={"wikitext": {"result": str(result)}},
            evaluation_params={"tasks": ["wikitext"]},
            execution_time=eval_time,
        )

        assert result is not None
