from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.plugins import Plugin, PluginRegistry
from goldcrest.orchestration.executor import WorkflowExecutor
from goldcrest.orchestration.workflow import Workflow, WorkflowStep
from goldcrest.plugins.compression import (
    ActivationScalingPlugin,
    BinarySearchRankPlugin,
    CalibrationCollectorPlugin,
    ModelConsolidator,
)
from goldcrest.plugins.evaluation import LMHarness
from goldcrest.plugins.evaluation import lm_eval as lm_eval_module


class ToyTokenizer:
    eos_token = "</s>"
    pad_token = "<pad>"
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 32


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name_or_path = "toy-model"
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            raise AssertionError("attention_mask is required")

        x = input_ids.float()
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=x, loss=x.mean())


def test_toy_workflow_runs_calibration_scaling_rank_and_evaluation(tmp_path, monkeypatch):
    context = PipelineContext(workspace_dir=tmp_path)
    model = ToyModel()
    tokenizer = ToyTokenizer()
    context.state.model = model
    context.state.tokenizer = tokenizer

    calibration_batches = [
        {"input_ids": torch.ones(2, 4), "attention_mask": torch.ones(2, 4)},
        {"input_ids": torch.ones(1, 4), "attention_mask": torch.ones(1, 4)},
    ]
    calibration_loader = DataLoader(calibration_batches, batch_size=None)

    captured_eval = {}

    class FakeAdapter:
        def __init__(self, model, tokenizer, **kwargs):
            self.model = model
            self.tokenizer = tokenizer
            self.kwargs = kwargs

    def fake_simple_evaluate(**kwargs):
        captured_eval["kwargs"] = kwargs
        assert isinstance(kwargs["model"], FakeAdapter)
        assert kwargs["model"].model is model
        return {"results": {"hellaswag": {"acc": 0.9}}, "n-samples": {"hellaswag": 1}}

    def fake_compress_model_with_surgery(self, model, **params):
        model.layers = nn.ModuleList([model.layers[0]])
        return SimpleNamespace(parameters={"remaining_layers": len(model.layers)})

    def safe_model_consolidator_init(
        self,
        compression_method="tensor_train",
        target_modules=None,
        device="cpu",
        backend="pytorch",
        name=None,
        **kwargs,
    ):
        Plugin.__init__(self, name="compression", **kwargs)
        self.compression_method = compression_method
        self.device = device
        self.backend = backend
        self.compression_params = dict(kwargs)
        self.method_overrides = kwargs.get("method_overrides") or kwargs.get("overrides")
        self.target_modules = target_modules or self._get_default_target_modules()
        self.compression_strategy = None
        self.tensorizer = None

    monkeypatch.setattr(lm_eval_module, "LM_EVAL_AVAILABLE", True)
    monkeypatch.setattr(
        lm_eval_module,
        "evaluator",
        SimpleNamespace(simple_evaluate=fake_simple_evaluate, __version__="test"),
    )
    monkeypatch.setattr(lm_eval_module, "LMLMHarnessModelAdapter", FakeAdapter)
    monkeypatch.setattr(ModelConsolidator, "__init__", safe_model_consolidator_init)
    monkeypatch.setattr(ModelConsolidator, "compress_model_with_surgery", fake_compress_model_with_surgery)

    registry = PluginRegistry()
    registry.register(CalibrationCollectorPlugin, "CalibrationCollectorPlugin")
    registry.register(ActivationScalingPlugin, "ActivationScalingPlugin")
    registry.register(BinarySearchRankPlugin, "BinarySearchRankPlugin")
    registry.register(ModelConsolidator, "ModelConsolidator")
    registry.register(LMHarness, "LMHarness")

    workflow = Workflow(
        name="toy",
        steps=[
            WorkflowStep(
                name="calibration",
                plugin="CalibrationCollectorPlugin",
                config={"dataloader": calibration_loader},
            ),
            WorkflowStep(
                name="scaling",
                plugin="ActivationScalingPlugin",
                config={"target_layers": ["layers"]},
                depends_on=["calibration"],
            ),
            WorkflowStep(
                name="rank",
                plugin="BinarySearchRankPlugin",
                config={"target_layers": ["layers"]},
                depends_on=["scaling"],
            ),
            WorkflowStep(
                name="consolidate",
                plugin="ModelConsolidator",
                config={"target_modules": ["layers"], "compression_method": "tensor_train"},
                depends_on=["rank"],
            ),
            WorkflowStep(
                name="eval",
                plugin="LMHarness",
                config={"tasks": ["hellaswag"]},
                depends_on=["consolidate"],
            ),
        ],
        plugin_configs={
            "CalibrationCollectorPlugin": {"n_samples": 3},
            "ActivationScalingPlugin": {"method": "abs_mean"},
            "BinarySearchRankPlugin": {"target_mode": "param_ratio", "param_ratio_target": 0.5},
            "ModelConsolidator": {"device": "cpu"},
            "LMHarness": {"backend": "auto", "device": "cpu"},
        },
    )

    before_params = sum(p.numel() for p in model.parameters())
    results = WorkflowExecutor(context, registry).execute_workflow(workflow)
    after_params = sum(p.numel() for p in context.state.model.parameters())

    assert results["calibration"]["samples"] == 3
    assert context.state.get("calibration.sample_count") == 3
    assert context.state.get("svd.scaling.layers.0") is not None
    assert context.state.get("svd.ranks.layers.0") is not None
    assert after_params < before_params
    assert results["eval"]["hellaswag"].metrics["acc"] == 0.9
    assert captured_eval["kwargs"]["model"].model is context.state.model
