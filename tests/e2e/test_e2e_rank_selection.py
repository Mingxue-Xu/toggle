"""
E2E tests — Rank Selection Criteria (Features 6–11)
====================================================
Each test applies a rank selection method to actual Gemma-3-270m-it weights
then verifies the downstream SVD compression uses the selected ranks.

Feature 6:  ASVD (activation-aware SVD scaling)
Feature 7:  SVD-LLM (data whitening + closed-form update)
Feature 8:  Information flow metrics (mutual information)
Feature 9:  Entropies (histogram entropy)
Feature 10: Fisher Information
Feature 11: Stable rank

Model: google/gemma-3-270m-it
  hidden=640, intermediate=2048, heads=4, kv_heads=1, head_dim=256, layers=18
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.framework.context import PipelineContext
from src.plugins.compression.consolidator import ModelConsolidator
from tests.e2e.conftest import param_count, run_forward, random_input, size_mb, DEVICE, MODEL_ID


def _make_calibration_loader(model, tokenizer, n_batches: int = 3, include_labels: bool = False):
    """Create a small calibration dataloader from real tokenized text."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be compressed using low-rank factorization.",
        "Singular value decomposition is a fundamental linear algebra technique.",
    ]
    batches = []
    for text in texts[:n_batches]:
        enc = tokenizer(text, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=32)
        batch = {
            "input_ids": enc["input_ids"].to(model.device),
            "attention_mask": enc["attention_mask"].to(model.device),
        }
        if include_labels:
            batch["labels"] = enc["input_ids"].clone().to(model.device)
        batches.append(batch)
    return DataLoader(batches, batch_size=None)


def _flatten_state_tree(tree, prefix: str = ""):
    """Flatten nested PipelineState dictionaries into dotted-key mappings."""
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_state_tree(value, path))
        else:
            flat[path] = value
    return flat


# ═══════════════════════════════════════════════════════════════════════════
# Feature 6 — ASVD (Activation-Aware SVD Scaling)
# ═══════════════════════════════════════════════════════════════════════════

class TestASVD:
    """ASVD computes per-column scaling factors from calibration activations,
    then SVD.compress() applies/reverses the scaling around the decomposition."""

    def test_asvd_pipeline_writes_scaling_factors(self, model, tokenizer, tmp_path, csv_logger):
        """CalibrationCollector → ActivationScaling should populate svd.scaling.* keys."""
        from src.plugins.compression.calibration_collector import CalibrationCollectorPlugin
        from src.plugins.compression.svd_activation_scaling import ActivationScalingPlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        # Step 1: Collect calibration data
        calib = CalibrationCollectorPlugin(n_samples=3, name="calibration")
        calib.initialize(ctx)
        loader = _make_calibration_loader(model, tokenizer)
        calib.execute(dataloader=loader)

        # Step 2: Compute activation scaling
        t0 = time.time()
        scaling = ActivationScalingPlugin(method="abs_mean", alpha=0.5, name="scaling")
        scaling.initialize(ctx)
        scaling.execute()
        elapsed = time.time() - t0

        # Verify scaling factors were written for at least some layers
        scaling = ctx.state.get("svd.scaling", {})
        assert isinstance(scaling, dict) and len(scaling) > 0, (
            "ASVD should write scaling factors under svd.scaling"
        )

        csv_logger.log_evaluation_results(
            "baseline", "profile", "asvd_scaling",
            {"asvd_scaling": {"num_scaling_keys": len(scaling)}},
            {"method": "abs_mean", "alpha": 0.5, "n_samples": 3},
            elapsed,
        )

    def test_asvd_followed_by_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """Full ASVD pipeline: calibrate → scale → SVD compress with use_activation_scaling."""
        from src.plugins.compression.calibration_collector import CalibrationCollectorPlugin
        from src.plugins.compression.svd_activation_scaling import ActivationScalingPlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_params = param_count(model)
        original_size = size_mb(model)

        # Calibration
        calib = CalibrationCollectorPlugin(n_samples=3, name="calibration")
        calib.initialize(ctx)
        calib.execute(dataloader=_make_calibration_loader(model, tokenizer))

        # Scaling
        scaling = ActivationScalingPlugin(method="abs_mean", alpha=0.5, name="scaling")
        scaling.initialize(ctx)
        scaling.execute()

        # SVD with activation scaling enabled
        targets = ["model.layers[0].mlp.gate_proj", "model.layers[0].self_attn.q_proj"]
        t0 = time.time()
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=targets,
            rank=32,
            svd_backend="torch",
            use_activation_scaling=True,
        )
        consolidator.initialize(ctx)
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)

        assert param_count(model) < original_params
        assert len(result.parameters["layers_replaced"]) == len(targets)

        csv_logger.log_model_info(
            MODEL_ID, "baseline", {"params": original_params},
            0.0, DEVICE, "float32",
        )
        csv_logger.log_model_info(
            MODEL_ID + "_asvd_svd", "compressed",
            {"params": param_count(model), "layers_replaced": len(targets)},
            0.0, DEVICE, "float32",
        )
        csv_logger.log_compression_results(
            "svd+asvd",
            {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            {"rank": 32, "targets": targets, "use_activation_scaling": True},
        )

        # Model should still run
        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="ASVD test")
        inference_time = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            "compressed", "inference", "asvd_svd",
            {"generation": {"output_length": len(output), "success": 1}},
            {"prompt": "ASVD test", "max_new_tokens": 4},
            inference_time,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 7 — SVD-LLM (Data Whitening + Closed-Form Update)
# ═══════════════════════════════════════════════════════════════════════════

class TestSVDLLM:
    """SVD-LLM pipeline: calibrate → data whitening → SVD → closed-form U update."""

    def test_data_whitening_writes_cholesky_factors(self, model, tokenizer, tmp_path, csv_logger):
        """DataWhiteningPlugin should write svd.whitening.L/L_inv.* keys."""
        from src.plugins.compression.calibration_collector import CalibrationCollectorPlugin
        from src.plugins.compression.svd_data_whitening import DataWhiteningPlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        calib = CalibrationCollectorPlugin(n_samples=3, name="calibration")
        calib.initialize(ctx)
        calib.execute(dataloader=_make_calibration_loader(model, tokenizer))

        t0 = time.time()
        whitening = DataWhiteningPlugin(regularization=1e-4, name="whitening")
        whitening.initialize(ctx)
        whitening.execute()
        elapsed = time.time() - t0

        whitening = ctx.state.get("svd.whitening", {})
        L_factors = whitening.get("L", {}) if isinstance(whitening, dict) else {}
        L_inv_factors = whitening.get("L_inv", {}) if isinstance(whitening, dict) else {}
        assert len(L_factors) > 0, "Whitening should write L factors"
        assert len(L_inv_factors) > 0, "Whitening should write L_inv factors"

        csv_logger.log_evaluation_results(
            "baseline", "profile", "data_whitening",
            {"whitening": {"num_L_factors": len(L_factors), "num_L_inv_factors": len(L_inv_factors)}},
            {"regularization": 1e-4, "n_samples": 3},
            elapsed,
        )

    def test_svdllm_full_pipeline(self, model, tokenizer, tmp_path, csv_logger):
        """Full SVD-LLM: calibrate → whiten → SVD compress with whitening enabled."""
        from src.plugins.compression.svdllm_pipeline import SVDLLMPipelinePlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        original_params = param_count(model)
        original_size = size_mb(model)

        targets = ["model.layers[0].mlp.down_proj"]
        dataloader = _make_calibration_loader(model, tokenizer)
        t0 = time.time()
        pipeline = SVDLLMPipelinePlugin(
            target_modules=targets,
            rank=32,
            regularization=1e-4,
            svd_backend="torch",
            use_closed_form_update=True,
            name="svdllm_pipeline",
        )
        pipeline.initialize(ctx)
        result = pipeline.execute(model=model, dataloader=dataloader)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)

        assert param_count(model) < original_params
        assert result["failures"] == {}
        assert result["layers_processed"] == 1

        csv_logger.log_compression_results(
            "svd+whitening",
            {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            {"rank": 32, "targets": targets, "use_data_whitening": True},
        )
        csv_logger.log_model_info(
            MODEL_ID, "baseline", {"params": original_params},
            0.0, DEVICE, "float32",
        )
        csv_logger.log_model_info(
            MODEL_ID + "_svdllm", "compressed",
            {"params": param_count(model), "layers_replaced": len(targets)},
            0.0, DEVICE, "float32",
        )

        t0 = time.time()
        output = run_forward(model, tokenizer, prompt="SVD-LLM test")
        inference_time = time.time() - t0
        assert isinstance(output, str) and len(output) > 0

        csv_logger.log_evaluation_results(
            "compressed", "inference", "svdllm",
            {"generation": {"output_length": len(output), "success": 1}},
            {"prompt": "SVD-LLM test", "max_new_tokens": 4},
            inference_time,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 8 — Information Flow Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestInformationFlow:
    """ActivationMetricsPlugin computes mutual information, Frobenius norm, etc."""

    def test_activation_metrics_computes_mi(self, model, tokenizer, tmp_path, csv_logger):
        """Run activation analysis and check that MI values are computed."""
        from src.plugins.analysis.activation_metrics import ActivationMetricsPlugin

        ctx = PipelineContext(
            config={
                "analysis": {
                    "metrics": {"names": ["compute_lda_matrix"]},
                    "selection": {
                        "include_names": ["mlp", "self_attn"],
                        "module_types": ["Gemma3MLP", "Gemma3Attention"],
                    },
                    "aggregation": {"reductions": ["mean", "median"]},
                    "compute": {"move_to_cpu": True, "seed": 42},
                },
            },
            workspace_dir=tmp_path,
        )
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        loader = _make_calibration_loader(model, tokenizer, n_batches=2)

        plugin = ActivationMetricsPlugin(name="activation_metrics")
        plugin.initialize(ctx)
        t0 = time.time()
        result = plugin.execute(dataloader=loader)
        elapsed = time.time() - t0

        # The report should contain per-layer statistics
        report = ctx.state.get("analysis.activation_metrics")
        assert report is not None, "Activation metrics report should be stored in state"
        per_layer = report.get("per_layer", [])
        assert len(per_layer) > 0, "Should have per-layer entries"

        has_frobenius = any(
            entry.get("statistics", {}).get("compute_lda_matrix.frobenius_norm", {}).get("median")
            is not None
            for entry in per_layer
        )
        has_mi = any(
            entry.get("statistics", {}).get("compute_lda_matrix.mutual_information_m0_m1", {}).get("median")
            is not None
            for entry in per_layer
        )
        assert has_frobenius, "Should compute Frobenius statistics for compute_lda_matrix"
        assert has_mi, "Should compute mutual information statistics for compute_lda_matrix"

        # Log per-layer metrics
        layer_metrics = {}
        for entry in per_layer:
            layer_name = entry.get("name", "unknown")
            stats = entry.get("statistics", {})
            frob = stats.get("compute_lda_matrix.frobenius_norm", {}).get("median")
            mi = stats.get("compute_lda_matrix.mutual_information_m0_m1", {}).get("median")
            layer_metrics[layer_name] = {
                "frobenius_norm_median": frob if frob is not None else -1.0,
                "mutual_information_median": mi if mi is not None else -1.0,
            }

        csv_logger.log_evaluation_results(
            "baseline", "profile", "activation_metrics",
            {"activation_analysis": {
                "num_layers_analyzed": len(per_layer),
                "has_frobenius": int(has_frobenius),
                "has_mi": int(has_mi),
            }},
            {"metrics": ["compute_lda_matrix"], "n_batches": 2, "reductions": ["mean", "median"]},
            elapsed,
        )

        # Log per-layer detail
        for layer_name, metrics in layer_metrics.items():
            csv_logger.log_evaluation_results(
                "baseline", "profile", "activation_metrics",
                {layer_name: metrics},
                {"metric_type": "per_layer_detail"},
                elapsed,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 9 — Entropies
# ═══════════════════════════════════════════════════════════════════════════

class TestEntropies:
    """Histogram entropy is computed as part of activation metrics."""

    def test_histogram_entropy_computed(self, model, tokenizer, tmp_path, csv_logger):
        """Activation metrics should include histogram_entropy values."""
        from src.plugins.analysis.activation_metrics import ActivationMetricsPlugin

        ctx = PipelineContext(
            config={
                "analysis": {
                    "metrics": {"names": ["compute_lda_matrix"]},
                    "selection": {"include_names": ["mlp"]},
                    "aggregation": {"reductions": ["mean"]},
                    "compute": {"move_to_cpu": True, "seed": 42},
                },
            },
            workspace_dir=tmp_path,
        )
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        loader = _make_calibration_loader(model, tokenizer, n_batches=2)
        plugin = ActivationMetricsPlugin(name="activation_metrics")
        plugin.initialize(ctx)
        t0 = time.time()
        plugin.execute(dataloader=loader)
        elapsed = time.time() - t0

        report = ctx.state.get("analysis.activation_metrics")
        per_layer = report.get("per_layer", [])

        # Check for histogram_entropy in at least one entry
        has_entropy = any(
            entry.get("statistics", {}).get("compute_lda_matrix.histogram_entropy", {}).get("mean")
            is not None
            for entry in per_layer
        )
        assert has_entropy, "Should compute histogram_entropy for layer activations"

        # Gather entropy values for logging
        entropy_values = {}
        for entry in per_layer:
            layer_name = entry.get("name", "unknown")
            ent = entry.get("statistics", {}).get("compute_lda_matrix.histogram_entropy", {}).get("mean")
            if ent is not None:
                entropy_values[layer_name] = ent

        csv_logger.log_evaluation_results(
            "baseline", "profile", "activation_metrics",
            {"histogram_entropy": {
                "num_layers_with_entropy": len(entropy_values),
                "has_entropy": int(has_entropy),
            }},
            {"metrics": ["compute_lda_matrix"], "selection": "mlp", "reductions": ["mean"]},
            elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 10 — Fisher Information
# ═══════════════════════════════════════════════════════════════════════════

class TestFisherInformation:
    """Fisher information approximation via squared gradients."""

    def test_fisher_computes_per_layer_scores(self, model, tokenizer, tmp_path, csv_logger):
        """FisherInformationPlugin should store calibration.fisher.* in state."""
        from src.plugins.analysis.fisher_information import FisherInformationPlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        fisher = FisherInformationPlugin(n_samples=2, name="fisher")
        fisher.initialize(ctx)

        loader = _make_calibration_loader(model, tokenizer, n_batches=2, include_labels=True)
        t0 = time.time()
        fisher.execute(dataloader=loader)
        elapsed = time.time() - t0

        fisher_data = ctx.state.get("calibration.fisher", {})
        assert isinstance(fisher_data, dict) and len(fisher_data) > 0, (
            "Fisher should write calibration.fisher entries to state"
        )

        csv_logger.log_evaluation_results(
            "baseline", "profile", "fisher_information",
            {"fisher_scores": {"num_layer_entries": len(fisher_data)}},
            {"n_samples": 2},
            elapsed,
        )

    def test_fisher_importance_scores(self, model, tokenizer, tmp_path, csv_logger):
        """Fisher importance scores should be in [0, 1] range."""
        from src.plugins.analysis.fisher_information import FisherInformationPlugin

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        fisher = FisherInformationPlugin(n_samples=2, name="fisher")
        fisher.initialize(ctx)

        loader = _make_calibration_loader(model, tokenizer, n_batches=2, include_labels=True)
        t0 = time.time()
        fisher.execute(dataloader=loader)
        elapsed = time.time() - t0

        fisher_data = _flatten_state_tree(ctx.state.get("calibration.fisher", {}))
        scores = fisher.compute_importance_scores(fisher_data)
        assert len(scores) > 0
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score {name}={score} outside [0,1]"

        csv_logger.log_evaluation_results(
            "baseline", "profile", "fisher_importance",
            {"importance_scores": {
                "num_scores": len(scores),
                "min_score": min(scores.values()),
                "max_score": max(scores.values()),
                "mean_score": sum(scores.values()) / len(scores),
            }},
            {"n_samples": 2},
            elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Feature 11 — Stable Rank
# ═══════════════════════════════════════════════════════════════════════════

class TestStableRank:
    """Stable rank = (frobenius_norm / spectral_norm)^2 drives base rank selection."""

    def test_stable_rank_decides_per_layer_ranks(self, model, tokenizer, tmp_path, csv_logger):
        """LayerSVDRankDecider with stable_rank strategy should produce per-layer ranks."""
        import json
        from src.plugins.analysis.layer_svd_rank_decider import LayerSVDRankDecider

        # Create a plausible activation report
        per_layer = []
        for i in range(18):
            for mod in ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]:
                per_layer.append({
                    "name": f"model.layers.{i}.{mod}",
                    "statistics": {
                        "compute_lda_matrix.frobenius_norm": {"median": float(50 + i * 2)},
                        "compute_lda_matrix.spectral_norm": {"median": float(5 + i * 0.5)},
                    },
                })
        report = {"per_layer": per_layer}

        cfg = {
            "rank_strategy": "stable_rank",
            "selection_metric_type": "compute_lda_matrix.frobenius_norm.median",
            "min_rank": 4,
            "max_rank": 128,
            "alpha_min": 0.5,
            "alpha_max": 1.0,
        }
        decider = LayerSVDRankDecider(cfg)
        t0 = time.time()
        decision = decider.decide(model, report)
        elapsed = time.time() - t0
        ranks = decision.get("layer_ranks", {})

        assert isinstance(ranks, dict)
        assert len(ranks) > 0, "Should produce at least one rank decision"
        for layer_name, rank in ranks.items():
            assert 4 <= rank <= 128, f"Rank for {layer_name} = {rank} outside [4, 128]"

        # Build per-layer rank task_results
        rank_results = {
            "rank_decision": {
                "num_layers_ranked": len(ranks),
                "min_rank": min(ranks.values()),
                "max_rank": max(ranks.values()),
                "mean_rank": sum(ranks.values()) / len(ranks),
            }
        }
        # Include individual layer ranks
        for layer_name, rank_val in ranks.items():
            rank_results[layer_name] = {"rank": rank_val}

        csv_logger.log_evaluation_results(
            "baseline", "profile", "rank_decider",
            rank_results,
            {"rank_strategy": "stable_rank", "min_rank": 4, "max_rank": 128,
             "alpha_min": 0.5, "alpha_max": 1.0},
            elapsed,
        )

    def test_stable_rank_drives_svd_compression(self, model, tokenizer, tmp_path, csv_logger):
        """Use stable-rank–decided ranks to actually SVD-compress the model."""
        from src.plugins.analysis.layer_svd_rank_decider import LayerSVDRankDecider

        per_layer = []
        for i in range(18):
            for mod in ["self_attn.q_proj", "mlp.gate_proj"]:
                per_layer.append({
                    "name": f"model.layers.{i}.{mod}",
                    "statistics": {
                        "compute_lda_matrix.frobenius_norm": {"median": float(40 + i * 3)},
                        "compute_lda_matrix.spectral_norm": {"median": float(4 + i * 0.3)},
                    },
                })
        report = {"per_layer": per_layer}

        cfg = {
            "rank_strategy": "stable_rank",
            "selection_metric_type": "compute_lda_matrix.frobenius_norm.median",
            "min_rank": 8,
            "max_rank": 64,
        }
        decider = LayerSVDRankDecider(cfg)
        t0 = time.time()
        decision = decider.decide(model, report)
        rank_decision_time = time.time() - t0
        ranks = decision.get("layer_ranks", {})
        assert len(ranks) > 0, "Should produce at least one layer rank"

        csv_logger.log_evaluation_results(
            "baseline", "profile", "rank_decider",
            {"rank_decision": {
                "num_layers_ranked": len(ranks),
                "min_rank": min(ranks.values()),
                "max_rank": max(ranks.values()),
                "mean_rank": sum(ranks.values()) / len(ranks),
            }},
            {"rank_strategy": "stable_rank", "min_rank": 8, "max_rank": 64},
            rank_decision_time,
        )

        # Pick first layer with a valid rank and compress it
        target_name = sorted(ranks.keys())[0]
        chosen_rank = ranks[target_name]

        original_params = param_count(model)
        original_size = size_mb(model)

        ctx = PipelineContext(config={}, workspace_dir=tmp_path)
        ctx.state.model = model
        ctx.state.tokenizer = tokenizer

        t0 = time.time()
        consolidator = ModelConsolidator(
            compression_method="svd",
            target_modules=[target_name],
            rank=chosen_rank,
            svd_backend="torch",
        )
        consolidator.initialize(ctx)
        result = consolidator.compress_model_with_surgery(model)
        compression_time = time.time() - t0

        compressed_size = size_mb(model)

        assert param_count(model) < original_params

        csv_logger.log_compression_results(
            "svd+stable_rank",
            {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
                "compression_time": compression_time,
                "memory_usage_mb": compressed_size,
            },
            {"rank": chosen_rank, "target": target_name, "rank_strategy": "stable_rank"},
        )
        csv_logger.log_model_info(
            MODEL_ID, "baseline", {"params": original_params},
            0.0, DEVICE, "float32",
        )
        csv_logger.log_model_info(
            MODEL_ID + "_stable_rank_svd", "compressed",
            {"params": param_count(model), "target": target_name, "rank": chosen_rank},
            0.0, DEVICE, "float32",
        )

        output = run_forward(model, tokenizer, prompt="Stable rank test")
        assert isinstance(output, str)
