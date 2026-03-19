import inspect

import pytest

from toggle.src.framework.context import PipelineContext
from toggle.src.plugins.compression import (
    ActivationScalingPlugin,
    BinarySearchRankPlugin,
    CalibrationCollectorPlugin,
    ModelConsolidator,
    PPLSensitivityPlugin,
)
from toggle.src.plugins.evaluation import CompressedModelProfile, LMHarness, UncompressedModelProfile
from toggle.src.plugins.models.loader import HuggingFaceModelLoader, LocalModelLoader


@pytest.mark.parametrize(
    "plugin_cls, kwargs",
    [
        (HuggingFaceModelLoader, {"model_name": "toy", "device": "cpu"}),
        (LocalModelLoader, {"model_name": "toy", "device": "cpu"}),
        (UncompressedModelProfile, {}),
        (CompressedModelProfile, {"compression_info": {}}),
        (LMHarness, {"tasks": ["hellaswag"], "backend": "auto", "device": "cpu"}),
        (CalibrationCollectorPlugin, {"n_samples": 1}),
        (ActivationScalingPlugin, {}),
        (BinarySearchRankPlugin, {}),
        (PPLSensitivityPlugin, {}),
        (ModelConsolidator, {"device": "cpu"}),
    ],
)
def test_concrete_plugins_are_instantiable(plugin_cls, kwargs):
    plugin = plugin_cls(**kwargs)

    assert isinstance(plugin, plugin_cls)
    assert not inspect.isabstract(plugin_cls)


def test_plugin_execute_uses_common_framework_path(tmp_path, monkeypatch):
    context = PipelineContext(workspace_dir=tmp_path)
    plugin = CalibrationCollectorPlugin(n_samples=1)
    plugin.initialize(context)

    def fake_do_execute(**kwargs):
        return {"seen": kwargs}

    monkeypatch.setattr(plugin, "do_execute", fake_do_execute)

    result = plugin.execute(answer=42)

    assert result == {"seen": {"answer": 42}}
    assert plugin.get_execution_stats()["executions"] == 1
    assert context.state.get("plugin_results.CalibrationCollectorPlugin") == result
