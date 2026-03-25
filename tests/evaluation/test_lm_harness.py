from types import SimpleNamespace

import pytest
import torch

from goldcrest.plugins.evaluation import lm_eval as lm_eval_module


class FakeTokenizer:
    eos_token = "</s>"
    pad_token = "<pad>"
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 16


def _install_fake_lm_eval(monkeypatch, captured, *, results=None):
    results = results or {"results": {"hellaswag": {"acc": 0.75}}, "n-samples": {"hellaswag": 2}}

    class FakeAdapter:
        def __init__(self, model, tokenizer, **kwargs):
            self.model = model
            self.tokenizer = tokenizer
            self.kwargs = kwargs

    def fake_simple_evaluate(**kwargs):
        captured["kwargs"] = kwargs
        return results

    monkeypatch.setattr(lm_eval_module, "LM_EVAL_AVAILABLE", True)
    monkeypatch.setattr(
        lm_eval_module,
        "evaluator",
        SimpleNamespace(simple_evaluate=fake_simple_evaluate, __version__="test"),
    )
    monkeypatch.setattr(lm_eval_module, "LMLMHarnessModelAdapter", FakeAdapter)
    return FakeAdapter


def test_default_backend_uses_in_memory_adapter(monkeypatch):
    captured = {}
    adapter_cls = _install_fake_lm_eval(monkeypatch, captured)
    model = torch.nn.Linear(4, 4)
    model.name_or_path = "stale-checkpoint-name"
    tokenizer = FakeTokenizer()

    harness = lm_eval_module.LMHarness(tasks=["hellaswag"], device="cpu")
    assert harness.backend == "auto"

    result = harness.evaluate_task(model, tokenizer, "hellaswag")

    assert isinstance(captured["kwargs"]["model"], adapter_cls)
    assert captured["kwargs"]["model"].model is model
    assert captured["kwargs"]["model"].tokenizer is tokenizer
    assert captured["kwargs"]["tasks"] == "hellaswag"
    assert result.metrics["acc"] == pytest.approx(0.75)


def test_explicit_auto_backend_uses_in_memory_adapter(monkeypatch):
    captured = {}
    adapter_cls = _install_fake_lm_eval(monkeypatch, captured)
    model = torch.nn.Linear(4, 4)
    tokenizer = FakeTokenizer()

    harness = lm_eval_module.LMHarness(tasks=["hellaswag"], backend="auto", device="cpu")
    result = harness.evaluate_task(model, tokenizer, "hellaswag", limit=1)

    assert isinstance(captured["kwargs"]["model"], adapter_cls)
    assert captured["kwargs"]["model"].model is model
    assert captured["kwargs"]["limit"] == 1
    assert result.metrics["acc"] == pytest.approx(0.75)


def test_checkpoint_backend_requires_explicit_opt_in(monkeypatch):
    captured = {}
    _install_fake_lm_eval(monkeypatch, captured)
    model = torch.nn.Linear(4, 4)
    model.name_or_path = "mutated-in-memory-model"
    tokenizer = FakeTokenizer()

    harness = lm_eval_module.LMHarness(tasks=["hellaswag"], backend="hf", hf_model_name="gpt2", device="cpu")

    with pytest.raises(ValueError, match="allow_checkpoint_reload"):
        harness.evaluate_task(model, tokenizer, "hellaswag")


def test_checkpoint_backend_with_opt_in_uses_hf_path(monkeypatch):
    captured = {}
    _install_fake_lm_eval(monkeypatch, captured)
    model = torch.nn.Linear(4, 4)
    tokenizer = FakeTokenizer()

    harness = lm_eval_module.LMHarness(
        tasks=["hellaswag"],
        backend="hf",
        hf_model_name="gpt2",
        allow_checkpoint_reload=True,
        device="cpu",
    )

    result = harness.evaluate_task(model, tokenizer, "hellaswag")

    assert captured["kwargs"]["model"] == "hf"
    assert captured["kwargs"]["model_args"]["pretrained"] == "gpt2"
    assert result.metrics["acc"] == pytest.approx(0.75)
