"""
E2E tests — Target Layers (Features 17–22)
============================================
Each test compresses a specific type of layer in the actual Gemma-3-270m-it model,
verifies the replacement, and runs inference.

Feature 17: MLP layers (gate_proj, up_proj, down_proj)
Feature 18: Q projection (self_attn.q_proj)
Feature 19: K projection (self_attn.k_proj)
Feature 20: V projection (self_attn.v_proj)
Feature 21: O projection (self_attn.o_proj)
Feature 22: Embedding layers

Model: google/gemma-3-270m-it
  hidden=640, intermediate=2048, heads=4, kv_heads=1, head_dim=256, layers=18
  q_proj: (1024, 640)  k_proj: (256, 640)  v_proj: (256, 640)  o_proj: (640, 1024)
  gate_proj: (2048, 640)  up_proj: (2048, 640)  down_proj: (640, 2048)
  embed_tokens: (262144, 640)
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
from src.framework.layers import FactorLinear, FactorEmbedding
from src.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import param_count, run_forward, random_input, size_mb, DEVICE, MODEL_ID


# ═══════════════════════════════════════════════════════════════════════════
# Feature 17 — MLP Layers (gate_proj, up_proj, down_proj)
# ═══════════════════════════════════════════════════════════════════════════

class TestMLPLayers:
    """Compress all three MLP projections in a transformer block."""

    def test_compress_all_mlp_projections(self, model, tokenizer, tmp_path, csv_logger):
        """SVD on gate_proj + up_proj + down_proj of layer 0."""
        original_params = param_count(model)
        original_size = size_mb(model)
        targets = [
            "model.layers[0].mlp.gate_proj",   # (2048, 640)
            "model.layers[0].mlp.up_proj",      # (2048, 640)
            "model.layers[0].mlp.down_proj",    # (640, 2048)
        ]
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=targets,
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert len(result.parameters["layers_replaced"]) == 3
        for t in targets:
            assert t in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "targets": targets,
                "rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_mlp_layers_replaced_with_factor_linear(self, model, tokenizer, tmp_path, csv_logger):
        """All MLP projections should become FactorLinear instances."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=[
                "model.layers[0].mlp.gate_proj",
                "model.layers[0].mlp.up_proj",
                "model.layers[0].mlp.down_proj",
            ],
            rank=48,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        mlp = model.model.layers[0].mlp
        assert isinstance(mlp.gate_proj, FactorLinear)
        assert isinstance(mlp.up_proj, FactorLinear)
        assert isinstance(mlp.down_proj, FactorLinear)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="structure_verification",
            plugin_name="structure_check",
            task_results={
                "mlp_replacement": {
                    "gate_proj_is_factor_linear": 1.0,
                    "up_proj_is_factor_linear": 1.0,
                    "down_proj_is_factor_linear": 1.0,
                }
            },
            evaluation_params={"rank": 48, "method": "svd"},
            execution_time=elapsed,
        )

    def test_mlp_compression_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Model with compressed MLP should still generate text."""
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
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Neural networks")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "mlp_compression_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "gate_proj", "rank": 64, "method": "svd"},
            execution_time=elapsed,
        )

    def test_mlp_gate_proj_dimensions(self, model, tokenizer, tmp_path, csv_logger):
        """gate_proj (2048, 640) → 3 SVD factors with rank=32."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].mlp.gate_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        layer = model.model.layers[0].mlp.gate_proj
        factors = list(layer.weight.factors)
        # U: (2048, 32), diag(S): (32, 32), Vt: (32, 640)
        assert factors[0].weight.shape == (2048, 32)
        assert factors[1].weight.shape == (32, 32)
        assert factors[2].weight.shape == (32, 640)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="dimension_verification",
            plugin_name="dimension_check",
            task_results={
                "gate_proj_factors": {
                    "U_shape_0": float(factors[0].weight.shape[0]),
                    "U_shape_1": float(factors[0].weight.shape[1]),
                    "S_shape_0": float(factors[1].weight.shape[0]),
                    "S_shape_1": float(factors[1].weight.shape[1]),
                    "Vt_shape_0": float(factors[2].weight.shape[0]),
                    "Vt_shape_1": float(factors[2].weight.shape[1]),
                }
            },
            evaluation_params={"target": "gate_proj", "rank": 32, "method": "svd"},
            execution_time=elapsed,
        )

    def test_mlp_wildcard_all_layers(self, model, tokenizer, tmp_path, csv_logger):
        """Wildcard model.layers[*].mlp.gate_proj should expand to all 18 layers."""
        original_size = size_mb(model)
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
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert len(result.parameters["layers_replaced"]) == 18

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "targets": "model.layers[*].mlp.gate_proj",
                "rank": 16,
                "wildcard": True,
                "layers_replaced": 18,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 18 — Q Projection (self_attn.q_proj)
# ═══════════════════════════════════════════════════════════════════════════

class TestQProjection:
    """Compress query projection: (1024, 640) = (heads*head_dim, hidden)."""

    def test_q_proj_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """SVD rank=64 on q_proj."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert isinstance(model.model.layers[0].self_attn.q_proj, FactorLinear)

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "q_proj",
                "rank": 64,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_q_proj_factor_dimensions(self, model, tokenizer, tmp_path, csv_logger):
        """q_proj (1024, 640) with rank=32 → U(1024,32), S(32,32), Vt(32,640)."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        factors = list(model.model.layers[0].self_attn.q_proj.weight.factors)
        assert factors[0].weight.shape == (1024, 32)
        assert factors[1].weight.shape == (32, 32)
        assert factors[2].weight.shape == (32, 640)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="dimension_verification",
            plugin_name="dimension_check",
            task_results={
                "q_proj_factors": {
                    "U_shape_0": float(factors[0].weight.shape[0]),
                    "U_shape_1": float(factors[0].weight.shape[1]),
                    "S_shape_0": float(factors[1].weight.shape[0]),
                    "S_shape_1": float(factors[1].weight.shape[1]),
                    "Vt_shape_0": float(factors[2].weight.shape[0]),
                    "Vt_shape_1": float(factors[2].weight.shape[1]),
                }
            },
            evaluation_params={"target": "q_proj", "rank": 32, "method": "svd"},
            execution_time=elapsed,
        )

    def test_q_proj_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Inference after compressing q_proj."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Query projection")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "q_proj_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "q_proj", "rank": 64, "method": "svd"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 19 — K Projection (self_attn.k_proj)
# ═══════════════════════════════════════════════════════════════════════════

class TestKProjection:
    """Compress key projection: (256, 640) = (kv_heads*head_dim, hidden) — GQA."""

    def test_k_proj_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """SVD on k_proj with GQA-sized weight (256, 640)."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.k_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert isinstance(model.model.layers[0].self_attn.k_proj, FactorLinear)

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "k_proj",
                "rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_k_proj_smaller_than_q_proj(self, model, tokenizer, tmp_path, csv_logger):
        """k_proj (256,640) should have smaller SVD factors than q_proj (1024,640)."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=[
                "model.layers[0].self_attn.q_proj",
                "model.layers[0].self_attn.k_proj",
            ],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        q_factors = list(model.model.layers[0].self_attn.q_proj.weight.factors)
        k_factors = list(model.model.layers[0].self_attn.k_proj.weight.factors)

        # U matrix first dimension differs: 1024 vs 256
        assert q_factors[0].weight.shape[0] == 1024
        assert k_factors[0].weight.shape[0] == 256

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="dimension_verification",
            plugin_name="dimension_check",
            task_results={
                "gqa_comparison": {
                    "q_proj_U_dim0": float(q_factors[0].weight.shape[0]),
                    "k_proj_U_dim0": float(k_factors[0].weight.shape[0]),
                    "q_larger_than_k": 1.0,
                }
            },
            evaluation_params={"rank": 32, "method": "svd", "gqa": True},
            execution_time=elapsed,
        )

    def test_k_proj_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Inference after compressing k_proj."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.k_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Key projection")
        elapsed = time.time() - t0
        assert isinstance(output, str)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "k_proj_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "k_proj", "rank": 32, "method": "svd"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 20 — V Projection (self_attn.v_proj)
# ═══════════════════════════════════════════════════════════════════════════

class TestVProjection:
    """Compress value projection: (256, 640) — same GQA dimensions as k_proj."""

    def test_v_proj_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """SVD on v_proj."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.v_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert isinstance(model.model.layers[0].self_attn.v_proj, FactorLinear)

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "v_proj",
                "rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_v_proj_factor_dimensions(self, model, tokenizer, tmp_path, csv_logger):
        """v_proj (256, 640) with rank=16 → U(256,16), S(16,16), Vt(16,640)."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.v_proj"],
            rank=16,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        factors = list(model.model.layers[0].self_attn.v_proj.weight.factors)
        assert factors[0].weight.shape == (256, 16)
        assert factors[1].weight.shape == (16, 16)
        assert factors[2].weight.shape == (16, 640)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="dimension_verification",
            plugin_name="dimension_check",
            task_results={
                "v_proj_factors": {
                    "U_shape_0": float(factors[0].weight.shape[0]),
                    "U_shape_1": float(factors[0].weight.shape[1]),
                    "S_shape_0": float(factors[1].weight.shape[0]),
                    "S_shape_1": float(factors[1].weight.shape[1]),
                    "Vt_shape_0": float(factors[2].weight.shape[0]),
                    "Vt_shape_1": float(factors[2].weight.shape[1]),
                }
            },
            evaluation_params={"target": "v_proj", "rank": 16, "method": "svd"},
            execution_time=elapsed,
        )

    def test_v_proj_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Inference after compressing v_proj."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.v_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Value projection")
        elapsed = time.time() - t0
        assert isinstance(output, str)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "v_proj_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "v_proj", "rank": 32, "method": "svd"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 21 — O Projection (self_attn.o_proj)
# ═══════════════════════════════════════════════════════════════════════════

class TestOProjection:
    """Compress output projection: (640, 1024) = (hidden, heads*head_dim)."""

    def test_o_proj_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """SVD on o_proj."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.o_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert isinstance(model.model.layers[0].self_attn.o_proj, FactorLinear)

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "o_proj",
                "rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_o_proj_factor_dimensions(self, model, tokenizer, tmp_path, csv_logger):
        """o_proj (640, 1024) with rank=32 → U(640,32), S(32,32), Vt(32,1024)."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.o_proj"],
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(model)
        elapsed = time.time() - t0

        factors = list(model.model.layers[0].self_attn.o_proj.weight.factors)
        assert factors[0].weight.shape == (640, 32)
        assert factors[1].weight.shape == (32, 32)
        assert factors[2].weight.shape == (32, 1024)

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="dimension_verification",
            plugin_name="dimension_check",
            task_results={
                "o_proj_factors": {
                    "U_shape_0": float(factors[0].weight.shape[0]),
                    "U_shape_1": float(factors[0].weight.shape[1]),
                    "S_shape_0": float(factors[1].weight.shape[0]),
                    "S_shape_1": float(factors[1].weight.shape[1]),
                    "Vt_shape_0": float(factors[2].weight.shape[0]),
                    "Vt_shape_1": float(factors[2].weight.shape[1]),
                }
            },
            evaluation_params={"target": "o_proj", "rank": 32, "method": "svd"},
            execution_time=elapsed,
        )

    def test_o_proj_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Inference after compressing o_proj."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.o_proj"],
            rank=64,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Output projection")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "o_proj_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "o_proj", "rank": 64, "method": "svd"},
            execution_time=elapsed,
        )

    def test_compress_all_attention_projections(self, model, tokenizer, tmp_path, csv_logger):
        """Compress q/k/v/o projections together in one block."""
        original_params = param_count(model)
        original_size = size_mb(model)
        targets = [
            "model.layers[0].self_attn.q_proj",
            "model.layers[0].self_attn.k_proj",
            "model.layers[0].self_attn.v_proj",
            "model.layers[0].self_attn.o_proj",
        ]
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=targets,
            rank=32,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert param_count(model) < original_params
        assert len(result.parameters["layers_replaced"]) == 4

        t1 = time.time()
        output = run_forward(model, tokenizer, prompt="All attention")
        inference_time = time.time() - t1
        assert isinstance(output, str) and len(output) > 0

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "targets": targets,
                "rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )
        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "all_attention_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"targets": targets, "rank": 32, "method": "svd"},
            execution_time=inference_time,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 22 — Embedding Layers
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingLayers:
    """Matrix-level compression of embeddings is unsupported.

    The SVD path returns FactorLinear which cannot handle integer token-id
    inputs, and weight tying with lm_head causes param count to increase
    rather than decrease.  See docs/layer_types.md for details.

    Use TestTokenLevelEmbeddingCompression for the supported
    vector-granularity approach.
    """

    pass


# ═══════════════════════════════════════════════════════════════════════════
# Feature 22b — Token-Level (Vector Granularity) Embedding Compression
# ═══════════════════════════════════════════════════════════════════════════

class TestTokenLevelEmbeddingCompression:
    """Token-level compression decomposes each embedding row independently.

    Each row (a 1D vector of embedding_dim elements) is reshaped via
    tensor_size into a higher-order tensor, then compressed with the chosen
    method.  The result is a FactorEmbedding with one FactorLayer per token.

    Full-vocab (262 144 rows) compression is very slow, so these tests build
    a small nn.Embedding and inject it into the model to keep runtime short.
    """

    @pytest.fixture()
    def small_embed_model(self, model):
        """Replace embed_tokens with a tiny (32, 640) embedding for fast tests."""
        import copy
        small = torch.nn.Embedding(32, 640)
        # Copy first 32 rows from original so weights are realistic
        with torch.no_grad():
            small.weight.copy_(model.model.embed_tokens.weight[:32])
        model.model.embed_tokens = small
        return model

    def test_svd_token_level(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """SVD vector-granularity: each embedding row → reshape (20, 32) → SVD rank=4."""
        model = small_embed_model
        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.embed_tokens"],
            rank=4,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[20, 32],
        )
        compression_time = time.time() - t0

        replaced = model.model.embed_tokens
        assert isinstance(replaced, FactorEmbedding), (
            f"Expected FactorEmbedding, got {type(replaced)}"
        )
        assert len(replaced.weight) == 32, "Should have one FactorLayer per token"

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "embed_tokens",
                "granularity": "vector",
                "tensor_size": [20, 32],
                "rank": 4,
                "num_tokens": 32,
            },
        )

    def test_tensor_train_token_level(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """TT vector-granularity: each row → reshape (10, 8, 8) → TT ranks=[1,4,4,1]."""
        model = small_embed_model
        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=["model.embed_tokens"],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[10, 8, 8],
        )
        compression_time = time.time() - t0

        replaced = model.model.embed_tokens
        assert isinstance(replaced, FactorEmbedding), (
            f"Expected FactorEmbedding, got {type(replaced)}"
        )
        assert len(replaced.weight) == 32

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tensor_train",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "embed_tokens",
                "granularity": "vector",
                "tensor_size": [10, 8, 8],
                "tensor_ranks": [1, 4, 4, 1],
                "num_tokens": 32,
            },
        )

    def test_tucker_token_level(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """Tucker vector-granularity: each row → reshape (20, 32) → Tucker ranks=[4, 4]."""
        model = small_embed_model
        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=["model.embed_tokens"],
            tucker_ranks=[4, 4],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[20, 32],
        )
        compression_time = time.time() - t0

        replaced = model.model.embed_tokens
        assert isinstance(replaced, FactorEmbedding), (
            f"Expected FactorEmbedding, got {type(replaced)}"
        )
        assert len(replaced.weight) == 32

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tucker",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "embed_tokens",
                "granularity": "vector",
                "tensor_size": [20, 32],
                "tucker_ranks": [4, 4],
                "num_tokens": 32,
            },
        )

    def test_cp_token_level(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """CP vector-granularity: each row → reshape (20, 32) → CP rank=4."""
        model = small_embed_model
        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=["model.embed_tokens"],
            cp_rank=4,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[20, 32],
        )
        compression_time = time.time() - t0

        replaced = model.model.embed_tokens
        assert isinstance(replaced, FactorEmbedding), (
            f"Expected FactorEmbedding, got {type(replaced)}"
        )
        assert len(replaced.weight) == 32

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="cp",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "embed_tokens",
                "granularity": "vector",
                "tensor_size": [20, 32],
                "cp_rank": 4,
                "num_tokens": 32,
            },
        )

    def test_token_level_reduces_params(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """Token-level SVD should reduce total parameter count."""
        model = small_embed_model
        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.embed_tokens"],
            rank=4,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[20, 32],
        )
        compression_time = time.time() - t0

        compressed_params = param_count(model)
        assert compressed_params < original_params, (
            f"Token-level SVD should reduce params: {compressed_params} >= {original_params}"
        )

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": "embed_tokens",
                "granularity": "vector",
                "tensor_size": [20, 32],
                "rank": 4,
                "original_params": original_params,
                "compressed_params": compressed_params,
            },
        )
        csv_logger.log_model_info(
            model_name=MODEL_ID,
            model_type="baseline",
            model_info={"num_parameters": original_params, "size_mb": original_size},
            loading_time=0.0,
            device=DEVICE,
            precision="float32",
        )
        csv_logger.log_model_info(
            model_name=MODEL_ID,
            model_type="compressed_token_level_svd",
            model_info={"num_parameters": compressed_params, "size_mb": compressed_size},
            loading_time=0.0,
            device=DEVICE,
            precision="float32",
        )

    def test_token_level_factor_layer_per_token(self, small_embed_model, tokenizer, tmp_path, csv_logger):
        """Each token should have its own FactorLayer with the correct func_name."""
        model = small_embed_model

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.embed_tokens"],
            rank=4,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        consolidator.compress_model_with_surgery(
            model, granularity="vector", tensor_size=[20, 32],
        )
        elapsed = time.time() - t0

        replaced = model.model.embed_tokens
        for i, factor_layer in enumerate(replaced.weight):
            assert hasattr(factor_layer, 'func_name'), (
                f"Token {i} FactorLayer missing func_name"
            )
            assert factor_layer.func_name == 'svd', (
                f"Token {i} expected func_name='svd', got '{factor_layer.func_name}'"
            )

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="structure_verification",
            plugin_name="structure_check",
            task_results={
                "token_level_factor_layers": {
                    "num_tokens": float(len(replaced.weight)),
                    "all_have_func_name": 1.0,
                    "all_func_name_svd": 1.0,
                }
            },
            evaluation_params={"target": "embed_tokens", "rank": 4, "method": "svd"},
            execution_time=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Remained feature-matrix cases — Tucker / CP / TT on Linear layers
# ═══════════════════════════════════════════════════════════════════════════

class TestLinearNonSVDMethods:
    """Tucker, CP, and Tensor-Train compression on MLP and Attention linear layers.

    Covers the '—' cells in the verified feature matrix (docs/layer_types.md).

    Each test deep-copies the model so surgery in one test does not pollute the
    next (load_state_dict only restores weights, not module structure).
    """

    @pytest.fixture()
    def fresh(self, model, tokenizer):
        """Return an independent (model, tokenizer) pair for each test."""
        import copy
        return copy.deepcopy(model), tokenizer

    # ── MLP (gate_proj: 2048×640) ────────────────────────────────────────

    def test_tucker_mlp_gate_proj(self, fresh, tmp_path, csv_logger):
        """Tucker on gate_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].mlp.gate_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=[target],
            tucker_ranks=[32, 32],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].mlp.gate_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tucker",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "tucker_ranks": [32, 32],
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_cp_mlp_gate_proj(self, fresh, tmp_path, csv_logger):
        """CP on gate_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].mlp.gate_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=[target],
            cp_rank=32,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].mlp.gate_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="cp",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "cp_rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_tensor_train_mlp_gate_proj(self, fresh, tmp_path, csv_logger):
        """Tensor Train on gate_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].mlp.gate_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=[target],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
            device="cpu",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].mlp.gate_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tensor_train",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "tensor_ranks": [1, 4, 4, 1],
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    # ── MLP inference ────────────────────────────────────────────────────

    def test_tucker_mlp_inference(self, fresh, tmp_path, csv_logger):
        """Inference after Tucker on gate_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=["model.layers[0].mlp.gate_proj"],
            tucker_ranks=[32, 32],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Tucker MLP")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "tucker_mlp_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "gate_proj", "method": "tucker", "tucker_ranks": [32, 32]},
            execution_time=elapsed,
        )

    def test_cp_mlp_inference(self, fresh, tmp_path, csv_logger):
        """Inference after CP on gate_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=["model.layers[0].mlp.gate_proj"],
            cp_rank=32,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="CP MLP")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "cp_mlp_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "gate_proj", "method": "cp", "cp_rank": 32},
            execution_time=elapsed,
        )

    def test_tensor_train_mlp_inference(self, fresh, tmp_path, csv_logger):
        """Inference after TT on gate_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=["model.layers[0].mlp.gate_proj"],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
            device="cpu",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="TT MLP")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "tt_mlp_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "gate_proj", "method": "tensor_train", "tensor_ranks": [1, 4, 4, 1]},
            execution_time=elapsed,
        )

    # ── Attention (q_proj: 1024×640) ─────────────────────────────────────

    def test_tucker_attention_q_proj(self, fresh, tmp_path, csv_logger):
        """Tucker on q_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].self_attn.q_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=[target],
            tucker_ranks=[32, 32],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].self_attn.q_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tucker",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "tucker_ranks": [32, 32],
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_cp_attention_q_proj(self, fresh, tmp_path, csv_logger):
        """CP on q_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].self_attn.q_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=[target],
            cp_rank=32,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].self_attn.q_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="cp",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "cp_rank": 32,
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    def test_tensor_train_attention_q_proj(self, fresh, tmp_path, csv_logger):
        """Tensor Train on q_proj → FactorLinear, params reduced."""
        model, tokenizer = fresh
        original_params = param_count(model)
        original_size = size_mb(model)
        target = "model.layers[0].self_attn.q_proj"
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=[target],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
            device="cpu",
        )
        consolidator.initialize(ctx)
        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        assert isinstance(model.model.layers[0].self_attn.q_proj, FactorLinear)
        assert param_count(model) < original_params
        assert target in result.parameters["layers_replaced"]

        compressed_size = size_mb(model)
        csv_logger.log_compression_results(
            compression_method="tensor_train",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={
                "target": target,
                "tensor_ranks": [1, 4, 4, 1],
                "layers_replaced": list(result.parameters["layers_replaced"]),
            },
        )

    # ── Attention inference ──────────────────────────────────────────────

    def test_tucker_attention_inference(self, fresh, tmp_path, csv_logger):
        """Inference after Tucker on q_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tucker",
            target_modules=["model.layers[0].self_attn.q_proj"],
            tucker_ranks=[32, 32],
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Tucker attention")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "tucker_attention_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "q_proj", "method": "tucker", "tucker_ranks": [32, 32]},
            execution_time=elapsed,
        )

    def test_cp_attention_inference(self, fresh, tmp_path, csv_logger):
        """Inference after CP on q_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="cp",
            target_modules=["model.layers[0].self_attn.q_proj"],
            cp_rank=32,
            backend="pytorch",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="CP attention")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "cp_attention_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "q_proj", "method": "cp", "cp_rank": 32},
            execution_time=elapsed,
        )

    def test_tensor_train_attention_inference(self, fresh, tmp_path, csv_logger):
        """Inference after TT on q_proj."""
        model, tokenizer = fresh
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="tensor_train",
            target_modules=["model.layers[0].self_attn.q_proj"],
            tensor_ranks=[1, 4, 4, 1],
            backend="pytorch",
            device="cpu",
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="TT attention")
        elapsed = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            model_type="gemma-3-270m-it",
            evaluation_type="inference_verification",
            plugin_name="inference_check",
            task_results={
                "tt_attention_inference": {
                    "output_length": float(len(output)),
                    "generates_text": 1.0,
                }
            },
            evaluation_params={"target": "q_proj", "method": "tensor_train", "tensor_ranks": [1, 4, 4, 1]},
            execution_time=elapsed,
        )
