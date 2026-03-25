"""
E2E tests — Compression Algorithms (Features 1–5)
==================================================
Each test loads google/gemma-3-270m-it, applies a compression algorithm to
real weight matrices, replaces the layers, and runs inference.

Feature 1: SVD
Feature 2: Tucker
Feature 3: CP (CANDECOMP/PARAFAC)
Feature 4: Tensor Train
Feature 5: Weight Pruning (block pruning)

Model architecture (gemma-3-270m-it):
  hidden=640, intermediate=2048, heads=4, kv_heads=1, head_dim=256,
  layers=18, vocab=262144
  q_proj: (1024, 640)  k_proj: (256, 640)  v_proj: (256, 640)  o_proj: (640, 1024)
  gate/up_proj: (2048, 640)  down_proj: (640, 2048)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.layers import FactorLinear, FactorEmbedding
from goldcrest.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import param_count, run_forward, random_input, size_mb, DEVICE, MODEL_ID


# ═══════════════════════════════════════════════════════════════════════════
# Feature 1 — SVD Compression
# ═══════════════════════════════════════════════════════════════════════════

class TestSVDCompression:
    """SVD on real Gemma-3-270m-it layers with torch backend."""

    TARGET_LAYERS = [
        "model.layers[0].mlp.gate_proj",       # (2048, 640) — large
        "model.layers[0].mlp.down_proj",        # (640, 2048)
        "model.layers[0].self_attn.q_proj",     # (1024, 640)
        "model.layers[0].self_attn.v_proj",     # (256, 640)
    ]
    RANK = 32  # SVD truncation rank

    def test_svd_compresses_and_reduces_params(self, model, tokenizer, tmp_path, csv_logger):
        """SVD with rank=32 should reduce parameter count on targeted layers."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=self.TARGET_LAYERS,
            rank=self.RANK,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)
        assert compressed_params < original_params, (
            f"SVD should reduce params: {compressed_params} >= {original_params}"
        )
        assert len(result.parameters["layers_replaced"]) == len(self.TARGET_LAYERS)
        assert result.parameters["model_surgery_performed"] is True

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("svd", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"rank": self.RANK, "svd_backend": "torch", "target_modules": self.TARGET_LAYERS})

    def test_svd_replaced_layers_are_factor_linear(self, model, tokenizer, tmp_path, csv_logger):
        """Replaced layers should be FactorLinear instances with 3 SVD factors."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[1].self_attn.q_proj"],
            rank=self.RANK,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        # Navigate to the replaced layer
        replaced = model.model.layers[1].self_attn.q_proj
        assert isinstance(replaced, FactorLinear), f"Expected FactorLinear, got {type(replaced)}"
        # SVD produces 3 factors: U, diag(S), Vt
        factors = list(replaced.weight.factors)
        assert len(factors) == 3, f"SVD should produce 3 factors, got {len(factors)}"

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("svd", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"rank": self.RANK, "svd_backend": "torch", "target_modules": ["model.layers[1].self_attn.q_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "structure_check",
            {"structure": {"is_factor_linear": True, "num_factors": len(factors)}},
            {"target": "model.layers[1].self_attn.q_proj"}, elapsed)

    def test_svd_compressed_model_runs_inference(self, model, tokenizer, tmp_path, csv_logger):
        """After SVD compression, model should still produce valid text."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].mlp.gate_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compress_elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="The capital of France is")
        inference_elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("svd", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": compress_elapsed, "memory_usage_mb": 0.0,
        }, {"rank": 64, "svd_backend": "torch", "target_modules": ["model.layers[0].mlp.gate_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
            {"inference": {"output_length": len(output), "valid": True}},
            {"prompt": "The capital of France is"}, inference_elapsed)

    def test_svd_compression_ratio_stored(self, model, tokenizer, tmp_path, csv_logger):
        """Surgery stats should include per-layer compression statistics."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        target = "model.layers[2].mlp.up_proj"
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=[target],
            rank=16,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        stats = result.parameters["surgery_stats"]
        assert target in stats
        assert stats[target]["compressed_params"] < stats[target]["original_params"]
        assert stats[target]["layer_type"] == "FactorLinear"

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("svd", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"rank": 16, "svd_backend": "torch", "target_modules": [target]})

    def test_svd_different_ranks(self, model, tokenizer, tmp_path, csv_logger):
        """Smaller rank → more compression (fewer params)."""
        target = ["model.layers[3].self_attn.o_proj"]
        results = {}
        for rank in [16, 64, 128]:
            # Reload original weights before each compression
            from tests.e2e.conftest import DTYPE, DEVICE as _DEVICE, MODEL_ID as _MODEL_ID
            from transformers import AutoModelForCausalLM
            import os
            m = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE,
                token=os.environ.get("HF_TOKEN"),
            ).to(_DEVICE).eval()

            orig_params = param_count(m)
            orig_mb_val = size_mb(m)

            ctx = PipelineContext(config={}, workspace_dir=tmp_path / f"rank_{rank}")
            ctx.state.model = m
            ctx.state.tokenizer = tokenizer

            c = ModelConsolidator(
                compression_method="svd", target_modules=target,
                rank=rank, svd_backend="torch",
            )
            c.initialize(ctx)
            t0 = time.time()
            c.compress_model_with_surgery(m)
            elapsed = time.time() - t0
            comp_params = param_count(m)
            comp_mb_val = size_mb(m)
            results[rank] = comp_params

            # Logging per rank
            csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": orig_params, "size_mb": orig_mb_val}, 0.0, DEVICE, "float32")
            csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": comp_params, "size_mb": comp_mb_val}, 0.0, DEVICE, "float32")
            csv_logger.log_compression_results("svd", {
                "original_size_mb": orig_mb_val, "compressed_size_mb": comp_mb_val,
                "compression_ratio": orig_params / max(comp_params, 1),
                "compression_time": elapsed, "memory_usage_mb": 0.0,
            }, {"rank": rank, "svd_backend": "torch", "target_modules": target})

            del m

        assert results[16] < results[64] < results[128], (
            f"Lower rank should produce fewer params: {results}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 2 — Tucker Decomposition
# ═══════════════════════════════════════════════════════════════════════════

class TestTuckerCompression:
    """Tucker decomposition on real Gemma-3-270m-it layers."""

    def test_tucker_compresses_mlp_layer(self, model, tokenizer, tmp_path, csv_logger):
        """Tucker with ranks=[32, 32] on a 2D MLP weight matrix."""
        original_params = param_count(model)
        orig_mb = size_mb(model)
        target = ["model.layers[0].mlp.gate_proj"]

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=target,
            tucker_ranks=[32, 32],
            device="cpu",
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)
        assert compressed_params < original_params
        assert len(result.parameters["layers_replaced"]) == 1

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tucker", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"tucker_ranks": [32, 32], "backend": "pytorch", "target_modules": target})

    def test_tucker_compressed_model_runs_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Tucker-compressed model should still produce output."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=["model.layers[0].self_attn.v_proj"],
            tucker_ranks=[32, 32],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compress_elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Hello world")
        inference_elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tucker", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": compress_elapsed, "memory_usage_mb": 0.0,
        }, {"tucker_ranks": [32, 32], "backend": "pytorch", "target_modules": ["model.layers[0].self_attn.v_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
            {"inference": {"output_length": len(output), "valid": True}},
            {"prompt": "Hello world"}, inference_elapsed)

    def test_tucker_surgery_stats(self, model, tokenizer, tmp_path, csv_logger):
        """Tucker surgery should record original vs compressed params."""
        original_params = param_count(model)
        orig_mb = size_mb(model)
        target = "model.layers[1].mlp.down_proj"

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=[target],
            tucker_ranks=[16, 16],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        stats = result.parameters["surgery_stats"][target]
        assert stats["compressed_params"] < stats["original_params"]

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tucker", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"tucker_ranks": [16, 16], "backend": "pytorch", "target_modules": [target]})


# ═══════════════════════════════════════════════════════════════════════════
# Feature 3 — CP (CANDECOMP/PARAFAC) Decomposition
# ═══════════════════════════════════════════════════════════════════════════

class TestCPCompression:
    """CP decomposition on real Gemma-3-270m-it layers."""

    def test_cp_compresses_attention_layer(self, model, tokenizer, tmp_path, csv_logger):
        """CP with rank=16 on an attention projection."""
        original_params = param_count(model)
        orig_mb = size_mb(model)
        target = ["model.layers[0].self_attn.q_proj"]

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=target,
            cp_rank=16,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)
        assert compressed_params < original_params
        assert len(result.parameters["layers_replaced"]) == 1

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("cp", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"cp_rank": 16, "backend": "pytorch", "target_modules": target})

    def test_cp_compressed_model_runs_inference(self, model, tokenizer, tmp_path, csv_logger):
        """CP-compressed model should produce output."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=["model.layers[0].mlp.up_proj"],
            cp_rank=8,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compress_elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Machine learning is")
        inference_elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("cp", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": compress_elapsed, "memory_usage_mb": 0.0,
        }, {"cp_rank": 8, "backend": "pytorch", "target_modules": ["model.layers[0].mlp.up_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
            {"inference": {"output_length": len(output), "valid": True}},
            {"prompt": "Machine learning is"}, inference_elapsed)

    def test_cp_rank_affects_compression(self, model, tokenizer, tmp_path, csv_logger):
        """Lower cp_rank should yield fewer parameters."""
        target = "model.layers[2].self_attn.v_proj"  # (256, 640)
        stats = {}
        for rank in [4, 16]:
            from transformers import AutoModelForCausalLM
            import os
            from tests.e2e.conftest import DTYPE, DEVICE as _DEVICE, MODEL_ID as _MODEL_ID
            m = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE,
                token=os.environ.get("HF_TOKEN"),
            ).to(_DEVICE).eval()

            orig_params = param_count(m)
            orig_mb_val = size_mb(m)

            ctx = PipelineContext(config={}, workspace_dir=tmp_path / f"cp_{rank}")
            ctx.state.model = m
            ctx.state.tokenizer = tokenizer
            c = ModelConsolidator(
                compression_method="cp", target_modules=[target],
                cp_rank=rank, backend="pytorch",
            )
            c.initialize(ctx)
            t0 = time.time()
            c.compress_model_with_surgery(m)
            elapsed = time.time() - t0
            comp_params = param_count(m)
            comp_mb_val = size_mb(m)
            stats[rank] = comp_params

            # Logging per rank
            csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": orig_params, "size_mb": orig_mb_val}, 0.0, DEVICE, "float32")
            csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": comp_params, "size_mb": comp_mb_val}, 0.0, DEVICE, "float32")
            csv_logger.log_compression_results("cp", {
                "original_size_mb": orig_mb_val, "compressed_size_mb": comp_mb_val,
                "compression_ratio": orig_params / max(comp_params, 1),
                "compression_time": elapsed, "memory_usage_mb": 0.0,
            }, {"cp_rank": rank, "backend": "pytorch", "target_modules": [target]})

            del m

        assert stats[4] < stats[16], f"Lower CP rank should give fewer params: {stats}"


# ═══════════════════════════════════════════════════════════════════════════
# Feature 4 — Tensor Train Decomposition
# ═══════════════════════════════════════════════════════════════════════════

class TestTensorTrainCompression:
    """Tensor Train decomposition on real Gemma-3-270m-it layers."""

    def test_tensor_train_compresses_mlp(self, model, tokenizer, tmp_path, csv_logger):
        """TT with ranks=[1,4,1] on an MLP projection."""
        original_params = param_count(model)
        orig_mb = size_mb(model)
        target = ["model.layers[0].mlp.gate_proj"]

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=target,
            tensor_ranks=[1, 4, 1],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)
        assert compressed_params < original_params
        assert len(result.parameters["layers_replaced"]) == 1

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tensor_train", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"tensor_ranks": [1, 4, 1], "backend": "pytorch", "target_modules": target})

    def test_tensor_train_inference(self, model, tokenizer, tmp_path, csv_logger):
        """TT-compressed model runs inference."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=["model.layers[0].self_attn.o_proj"],
            tensor_ranks=[1, 4, 1],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        compress_elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="AI is")
        inference_elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tensor_train", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": compress_elapsed, "memory_usage_mb": 0.0,
        }, {"tensor_ranks": [1, 4, 1], "backend": "pytorch", "target_modules": ["model.layers[0].self_attn.o_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
            {"inference": {"output_length": len(output), "valid": True}},
            {"prompt": "AI is"}, inference_elapsed)

    def test_tensor_train_auto_reshape_2d(self, model, tokenizer, tmp_path, csv_logger):
        """TT should auto-reshape 2D weight matrices to 3D for decomposition."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        # down_proj is (640, 2048) — 2D, TT will reshape to 3D internally
        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=["model.layers[1].mlp.down_proj"],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        assert "model.layers[1].mlp.down_proj" in result.parameters["layers_replaced"]

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("tensor_train", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"tensor_ranks": [1, 4, 4, 1], "backend": "pytorch", "target_modules": ["model.layers[1].mlp.down_proj"]})
        csv_logger.log_evaluation_results("compressed", "profile", "structure_check",
            {"structure": {"auto_reshape_2d": True, "layer_replaced": "model.layers[1].mlp.down_proj"}},
            {"target": "model.layers[1].mlp.down_proj"}, elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# Feature 5 — Weight Pruning (Block Pruning)
# ═══════════════════════════════════════════════════════════════════════════

class TestPruning:
    """Block pruning on real Gemma-3-270m-it layers.
    Pruning requires an activation metrics report to score blocks."""

    def _make_dummy_report(self, num_layers: int = 18) -> dict:
        """Create a plausible activation metrics report for pruning."""
        import random
        random.seed(42)
        per_layer = []
        for i in range(num_layers):
            for module in ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.up_proj"]:
                per_layer.append({
                    "name": f"model.layers.{i}.{module}",
                    "statistics": {
                        "compute_lda_matrix.frobenius_norm": {"median": random.uniform(1.0, 100.0)},
                        "compute_lda_matrix.spectral_norm": {"median": random.uniform(0.1, 10.0)},
                        "compute_lda_matrix.l2_norm": {"median": random.uniform(0.5, 50.0)},
                    },
                })
        return {"per_layer": per_layer}

    def test_pruning_removes_blocks(self, model, tokenizer, tmp_path, csv_logger):
        """Pruning should remove transformer blocks and reduce layer count."""
        original_num_layers = len(model.model.layers)
        original_params = param_count(model)
        orig_mb = size_mb(model)

        report = self._make_dummy_report(original_num_layers)
        report_path = tmp_path / "activation_report.json"
        report_path.write_text(json.dumps(report))

        from goldcrest.plugins.compression.pruning import PruningPlugin
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        pruning = PruningPlugin(
            name="pruning",
            **{"pruning": {
                "selection_metric_type": "compute_lda_matrix.l2_norm.median",
                "report_path": str(report_path),
                "num_deleted_block": 2,
                "min_keep": 1,
                "container_path": "model.layers",
                "group_prefix": "layers",
            }},
        )
        pruning.initialize(ctx)
        t0 = time.time()
        result = pruning.execute(model=model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        assert len(model.model.layers) == original_num_layers - 2

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("block_pruning", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"num_deleted_block": 2, "min_keep": 1, "container_path": "model.layers",
            "selection_metric_type": "compute_lda_matrix.l2_norm.median"})

    def test_pruned_model_runs_inference(self, model, tokenizer, tmp_path, csv_logger):
        """A model with 2 blocks pruned should still produce output."""
        original_params = param_count(model)
        orig_mb = size_mb(model)

        report = self._make_dummy_report(len(model.model.layers))
        report_path = tmp_path / "activation_report.json"
        report_path.write_text(json.dumps(report))

        from goldcrest.plugins.compression.pruning import PruningPlugin
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        pruning = PruningPlugin(
            name="pruning",
            **{"pruning": {
                "selection_metric_type": "compute_lda_matrix.l2_norm.median",
                "report_path": str(report_path),
                "num_deleted_block": 2,
                "min_keep": 1,
                "container_path": "model.layers",
            }},
        )
        pruning.initialize(ctx)
        t0 = time.time()
        pruning.execute(model=model)
        compress_elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Pruning is")
        inference_elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("block_pruning", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": compress_elapsed, "memory_usage_mb": 0.0,
        }, {"num_deleted_block": 2, "min_keep": 1, "container_path": "model.layers",
            "selection_metric_type": "compute_lda_matrix.l2_norm.median"})
        csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
            {"inference": {"output_length": len(output), "valid": True}},
            {"prompt": "Pruning is"}, inference_elapsed)

    def test_pruning_ratio_mode(self, model, tokenizer, tmp_path, csv_logger):
        """Pruning by ratio should remove approximately the right fraction."""
        num_layers = len(model.model.layers)
        original_params = param_count(model)
        orig_mb = size_mb(model)

        report = self._make_dummy_report(num_layers)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))

        from goldcrest.plugins.compression.pruning import PruningPlugin
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        pruning = PruningPlugin(
            name="pruning",
            **{"pruning": {
                "selection_metric_type": "compute_lda_matrix.l2_norm.median",
                "report_path": str(report_path),
                "ratio_deleted_block": 0.25,
                "min_keep": 1,
                "container_path": "model.layers",
            }},
        )
        pruning.initialize(ctx)
        t0 = time.time()
        pruning.execute(model=model)
        elapsed = time.time() - t0

        compressed_params = param_count(model)
        comp_mb = size_mb(model)

        expected_removed = max(1, int(num_layers * 0.25))
        remaining = len(model.model.layers)
        # Allow ±1 due to rounding
        assert abs((num_layers - remaining) - expected_removed) <= 1

        # Logging
        csv_logger.log_model_info(MODEL_ID, "baseline", {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_model_info(MODEL_ID, "compressed", {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")
        csv_logger.log_compression_results("block_pruning", {
            "original_size_mb": orig_mb, "compressed_size_mb": comp_mb,
            "compression_ratio": original_params / max(compressed_params, 1),
            "compression_time": elapsed, "memory_usage_mb": 0.0,
        }, {"ratio_deleted_block": 0.25, "min_keep": 1, "container_path": "model.layers",
            "selection_metric_type": "compute_lda_matrix.l2_norm.median"})
