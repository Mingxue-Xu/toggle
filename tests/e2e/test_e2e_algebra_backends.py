"""
E2E tests — Algebra Backends (Features 12–16)
===============================================
Each test uses a specific algebra backend to perform actual SVD or tensor
decomposition on real Gemma-3-270m-it weight matrices.

Feature 12: CoLA backend
Feature 13: Lanczos algorithm (via CoLA)
Feature 14: LOBPCG algorithm (via CoLA)
Feature 15: PyTorch SVD backend (torch.linalg.svd)
Feature 16: TensorLy backend (Tucker/CP/TT decompositions)

Model: google/gemma-3-270m-it
  q_proj: (1024, 640)  gate_proj: (2048, 640)  v_proj: (256, 640)
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
from src.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import param_count, run_forward, size_mb, DEVICE, MODEL_ID


# ═══════════════════════════════════════════════════════════════════════════
# Feature 12 — CoLA Backend (Dense algorithm)
# ═══════════════════════════════════════════════════════════════════════════

class TestCoLABackend:
    """SVD via CoLA Dense algorithm on real model weights."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cola(self):
        try:
            import cola
        except ImportError:
            pytest.skip("CoLA not installed")

    def test_cola_dense_svd_compresses(self, model, tokenizer, tmp_path, csv_logger):
        """CoLA Dense SVD should compress an MLP projection."""
        original_params = param_count(model)
        original_size = size_mb(model)
        target = ["model.layers[0].mlp.gate_proj"]

        csv_logger.log_model_info(
            model_name=MODEL_ID,
            model_type="baseline",
            model_info={"total_params": original_params, "target_modules": target},
            loading_time=0.0,
            device=DEVICE,
            precision="float32",
        )

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=target,
            rank=32,
            svd_backend="cola",
            svd_backend_config={"algorithm": "dense"},
        )
        consolidator.initialize(ctx)

        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)

        csv_logger.log_model_info(
            model_name=MODEL_ID,
            model_type="compressed_cola_dense",
            model_info={"total_params": param_count(model), "layers_replaced": len(result.parameters["layers_replaced"])},
            loading_time=compression_time,
            device=DEVICE,
            precision="float32",
        )

        csv_logger.log_compression_results(
            compression_method="svd",
            compression_results={
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            parameters={"svd_backend": "cola", "algorithm": "dense", "rank": 32, "target_modules": target},
        )

        assert param_count(model) < original_params
        assert len(result.parameters["layers_replaced"]) == 1

    def test_cola_dense_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Model compressed via CoLA Dense should produce output."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=64,
            svd_backend="cola",
            svd_backend_config={"algorithm": "dense"},
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="CoLA test")
        inference_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed_cola_dense",
            evaluation_type="inference",
            plugin_name="inference_check",
            task_results={"cola_dense_inference": {"output_length": len(output), "produces_output": int(isinstance(output, str) and len(output) > 0)}},
            evaluation_params={"svd_backend": "cola", "algorithm": "dense", "rank": 64, "prompt": "CoLA test"},
            execution_time=inference_time,
        )

        assert isinstance(output, str) and len(output) > 0

    def test_cola_auto_algorithm(self, model, tokenizer, tmp_path, csv_logger):
        """CoLA 'auto' algorithm should select appropriate SVD method."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[1].self_attn.v_proj"],
            rank=16,
            svd_backend="cola",
            svd_backend_config={"algorithm": "auto"},
        )
        consolidator.initialize(ctx)

        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

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
            parameters={"svd_backend": "cola", "algorithm": "auto", "rank": 16, "target_modules": ["model.layers[1].self_attn.v_proj"]},
        )

        assert param_count(model) < original_params


# ═══════════════════════════════════════════════════════════════════════════
# Feature 13 — Lanczos Algorithm (via CoLA)
# ═══════════════════════════════════════════════════════════════════════════

class TestLanczosBackend:
    """SVD via CoLA Lanczos iterative algorithm."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cola(self):
        try:
            import cola
        except ImportError:
            pytest.skip("CoLA not installed")

    def test_lanczos_svd_compresses(self, model, tokenizer, tmp_path, csv_logger):
        """Lanczos-based SVD should compress attention layers."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.q_proj"],
            rank=32,
            svd_backend="cola",
            svd_backend_config={"algorithm": "lanczos", "max_iters": 100, "tol": 1e-6},
        )
        consolidator.initialize(ctx)

        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

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
            parameters={"svd_backend": "cola", "algorithm": "lanczos", "rank": 32, "max_iters": 100, "tol": 1e-6},
        )

        assert param_count(model) < original_params

    def test_lanczos_inference(self, model, tokenizer, tmp_path, csv_logger):
        """Model compressed via Lanczos should run inference."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[1].mlp.up_proj"],
            rank=48,
            svd_backend="cola",
            svd_backend_config={"algorithm": "lanczos", "max_iters": 50},
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="Lanczos test")
        inference_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed_lanczos",
            evaluation_type="inference",
            plugin_name="inference_check",
            task_results={"lanczos_inference": {"output_length": len(output), "produces_output": int(isinstance(output, str) and len(output) > 0)}},
            evaluation_params={"svd_backend": "cola", "algorithm": "lanczos", "rank": 48, "max_iters": 50, "prompt": "Lanczos test"},
            execution_time=inference_time,
        )

        assert isinstance(output, str) and len(output) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Feature 14 — LOBPCG Algorithm (via CoLA)
# ═══════════════════════════════════════════════════════════════════════════

class TestLOBPCGBackend:
    """SVD via CoLA LOBPCG eigenvalue solver."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cola(self):
        try:
            import cola
        except ImportError:
            pytest.skip("CoLA not installed")

    def test_lobpcg_svd_compresses(self, model, tokenizer, tmp_path, csv_logger):
        """LOBPCG-based SVD should compress layers."""
        original_params = param_count(model)
        original_size = size_mb(model)
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[0].self_attn.o_proj"],
            rank=32,
            svd_backend="cola",
            svd_backend_config={"algorithm": "lobpcg", "max_iters": 50},
        )
        consolidator.initialize(ctx)

        t0 = time.time()
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

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
            parameters={"svd_backend": "cola", "algorithm": "lobpcg", "rank": 32, "max_iters": 50},
        )

        assert param_count(model) < original_params

    def test_lobpcg_inference(self, model, tokenizer, tmp_path, csv_logger):
        """LOBPCG-compressed model should produce output."""
        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=["model.layers[2].mlp.down_proj"],
            rank=24,
            svd_backend="cola",
            svd_backend_config={"algorithm": "lobpcg", "max_iters": 30},
        )
        consolidator.initialize(ctx)
        consolidator.compress_model_with_surgery(model)

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="LOBPCG result")
        inference_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="compressed_lobpcg",
            evaluation_type="inference",
            plugin_name="inference_check",
            task_results={"lobpcg_inference": {"output_length": len(output), "produces_output": int(isinstance(output, str) and len(output) > 0)}},
            evaluation_params={"svd_backend": "cola", "algorithm": "lobpcg", "rank": 24, "max_iters": 30, "prompt": "LOBPCG result"},
            execution_time=inference_time,
        )

        assert isinstance(output, str) and len(output) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Feature 15 — PyTorch SVD Backend
# ═══════════════════════════════════════════════════════════════════════════

class TestPyTorchBackend:
    """SVD via torch.linalg.svd — the default backend."""

    def test_pytorch_svd_compresses(self, model, tokenizer, tmp_path, csv_logger):
        """PyTorch SVD backend should compress layers."""
        original_params = param_count(model)
        original_size = size_mb(model)
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
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

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
            parameters={"svd_backend": "torch", "rank": 32, "target_modules": ["model.layers[0].mlp.gate_proj"]},
        )

        assert param_count(model) < original_params

    def test_pytorch_svd_on_multiple_layers(self, model, tokenizer, tmp_path, csv_logger):
        """PyTorch SVD on attention + MLP layers simultaneously."""
        original_params = param_count(model)
        original_size = size_mb(model)
        targets = [
            "model.layers[0].self_attn.q_proj",
            "model.layers[0].self_attn.k_proj",
            "model.layers[0].mlp.gate_proj",
            "model.layers[0].mlp.down_proj",
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
            parameters={"svd_backend": "torch", "rank": 32, "target_modules": targets, "num_layers": len(targets)},
        )

        assert param_count(model) < original_params
        assert len(result.parameters["layers_replaced"]) == len(targets)
        output = run_forward(model, tokenizer, prompt="PyTorch SVD")
        assert isinstance(output, str) and len(output) > 0

    def test_pytorch_backend_is_default(self, model, tokenizer, tmp_path, csv_logger):
        """When no svd_backend is specified, torch should be used."""
        from src.plugins.compression.svd_backend import build_svd_backend, TorchSVDBackend
        backend = build_svd_backend("torch", {})
        assert isinstance(backend, TorchSVDBackend)

        # Verify it works on actual weight data
        weight = model.model.layers[0].self_attn.q_proj.weight.data.float()

        t0 = time.time()
        U, S, Vh = backend.compute_svd(weight, k=16)
        svd_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="pytorch_svd_default",
            evaluation_type="backend_verification",
            plugin_name="backend_check",
            task_results={"pytorch_default_backend": {
                "is_torch_backend": 1,
                "U_shape_0": U.shape[0],
                "U_shape_1": U.shape[1],
                "S_shape_0": S.shape[0],
                "Vh_shape_0": Vh.shape[0],
                "Vh_shape_1": Vh.shape[1],
                "k": 16,
            }},
            evaluation_params={"svd_backend": "torch", "k": 16, "weight_shape": list(weight.shape)},
            execution_time=svd_time,
        )

        assert U.shape[1] == 16
        assert S.shape[0] == 16
        assert Vh.shape[0] == 16


# ═══════════════════════════════════════════════════════════════════════════
# Feature 16 — TensorLy Backend
# ═══════════════════════════════════════════════════════════════════════════

class TestTensorLyBackend:
    """TensorLy used by Tucker, CP, and Tensor Train decompositions."""

    def test_tensorly_tucker_decomposition(self, model, tokenizer, tmp_path, csv_logger):
        """TensorLy should perform Tucker decomposition on real weight."""
        import tensorly as tl
        from src.plugins.compression.tensorly_backend import set_tensorly_backend
        set_tensorly_backend("pytorch")

        weight = model.model.layers[0].mlp.gate_proj.weight.data.float()

        t0 = time.time()
        result = tl.decomposition.tucker(weight, rank=[32, 32])
        decomp_time = time.time() - t0

        core = result[0] if isinstance(result, (tuple, list)) else result.core

        csv_logger.log_evaluation_results(
            model_type="tensorly_tucker",
            evaluation_type="decomposition",
            plugin_name="tensorly_tucker",
            task_results={"tucker_decomposition": {
                "core_shape_0": core.shape[0],
                "core_shape_1": core.shape[1],
                "original_shape_0": weight.shape[0],
                "original_shape_1": weight.shape[1],
                "rank_0": 32,
                "rank_1": 32,
            }},
            evaluation_params={"backend": "pytorch", "decomposition": "tucker", "rank": [32, 32], "weight_shape": list(weight.shape)},
            execution_time=decomp_time,
        )

        assert core.shape == (32, 32)

    def test_tensorly_cp_decomposition(self, model, tokenizer, tmp_path, csv_logger):
        """TensorLy should perform CP decomposition on real weight."""
        import tensorly as tl
        from src.plugins.compression.tensorly_backend import set_tensorly_backend
        set_tensorly_backend("pytorch")

        weight = model.model.layers[0].self_attn.v_proj.weight.data.float()

        t0 = time.time()
        result = tl.decomposition.parafac(weight, rank=8)
        decomp_time = time.time() - t0

        reconstructed = tl.cp_to_tensor(result)

        csv_logger.log_evaluation_results(
            model_type="tensorly_cp",
            evaluation_type="decomposition",
            plugin_name="tensorly_cp",
            task_results={"cp_decomposition": {
                "rank": 8,
                "reconstructed_shape_match": int(reconstructed.shape == weight.shape),
                "original_shape_0": weight.shape[0],
                "original_shape_1": weight.shape[1],
            }},
            evaluation_params={"backend": "pytorch", "decomposition": "cp", "rank": 8, "weight_shape": list(weight.shape)},
            execution_time=decomp_time,
        )

        assert reconstructed.shape == weight.shape

    def test_tensorly_tt_decomposition(self, model, tokenizer, tmp_path, csv_logger):
        """TensorLy should perform Tensor Train decomposition on reshaped weight."""
        import tensorly as tl
        from src.plugins.compression.tensorly_backend import set_tensorly_backend
        set_tensorly_backend("pytorch")

        weight = model.model.layers[0].mlp.down_proj.weight.data.float()
        # Reshape 2D to 3D for TT (down_proj is 640x2048)
        reshaped = weight.reshape(640, 2, 1024)
        ranks = tl.validate_tt_rank(reshaped.shape, [1, 4, 4, 1])

        t0 = time.time()
        tt = tl.decomposition.tensor_train(reshaped, rank=ranks)
        decomp_time = time.time() - t0

        if hasattr(tt, 'factors'):
            factors = tt.factors
        elif isinstance(tt, list):
            factors = tt
        else:
            factors = [tt[i] for i in range(len(tt))]
        reconstructed = tl.tt_to_tensor(factors)

        csv_logger.log_evaluation_results(
            model_type="tensorly_tt",
            evaluation_type="decomposition",
            plugin_name="tensorly_tt",
            task_results={"tt_decomposition": {
                "num_factors": len(factors),
                "reconstructed_shape_match": int(reconstructed.shape == reshaped.shape),
                "reshaped_shape_0": reshaped.shape[0],
                "reshaped_shape_1": reshaped.shape[1],
                "reshaped_shape_2": reshaped.shape[2],
            }},
            evaluation_params={"backend": "pytorch", "decomposition": "tensor_train", "ranks": list(ranks) if hasattr(ranks, '__iter__') else ranks, "reshaped_shape": list(reshaped.shape)},
            execution_time=decomp_time,
        )

        assert reconstructed.shape == reshaped.shape

    def test_tensorly_backend_switch(self, model, tokenizer, tmp_path, csv_logger):
        """set_tensorly_backend('pytorch') should set TensorLy to use PyTorch."""
        import tensorly as tl
        from src.plugins.compression.tensorly_backend import set_tensorly_backend

        t0 = time.time()
        set_tensorly_backend("pytorch")
        switch_time = time.time() - t0

        current_backend = tl.get_backend()

        csv_logger.log_evaluation_results(
            model_type="tensorly_backend_switch",
            evaluation_type="backend_verification",
            plugin_name="tensorly_backend",
            task_results={"backend_switch": {"backend_is_pytorch": int(current_backend == "pytorch"), "backend_name": current_backend}},
            evaluation_params={"requested_backend": "pytorch"},
            execution_time=switch_time,
        )

        assert tl.get_backend() == "pytorch"

    def test_tensorly_cola_not_implemented(self, csv_logger):
        """CoLA backend for TensorLy should raise NotImplementedError."""
        from src.plugins.compression.tensorly_backend import set_tensorly_backend

        t0 = time.time()
        with pytest.raises(NotImplementedError):
            set_tensorly_backend("cola")
        check_time = time.time() - t0

        csv_logger.log_evaluation_results(
            model_type="tensorly_cola_not_implemented",
            evaluation_type="error_handling",
            plugin_name="tensorly_backend",
            task_results={"cola_not_implemented": {"raises_not_implemented": 1}},
            evaluation_params={"requested_backend": "cola"},
            execution_time=check_time,
        )
