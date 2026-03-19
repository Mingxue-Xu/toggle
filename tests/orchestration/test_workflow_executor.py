from types import SimpleNamespace

from toggle.src.framework.context import PipelineContext
from toggle.src.framework.plugins import Plugin, PluginRegistry
from toggle.src.orchestration.executor import WorkflowExecutor
from toggle.src.orchestration.workflow import Workflow, WorkflowStep


class TrackingPlugin(Plugin):
    last_instance = None

    def __init__(self, constructor_value=None, **kwargs):
        super().__init__(**kwargs)
        self.constructor_value = constructor_value
        self.constructor_kwargs = dict(kwargs)
        self.calls = []
        TrackingPlugin.last_instance = self

    def do_execute(self, context, model=None, tokenizer=None, **kwargs):
        payload = {
            "context": context,
            "model": model,
            "tokenizer": tokenizer,
            "kwargs": dict(kwargs),
            "constructor_value": self.constructor_value,
            "constructor_kwargs": dict(self.constructor_kwargs),
        }
        self.calls.append(payload)
        return payload


def test_workflow_executor_separates_constructor_and_runtime_config(tmp_path):
    context = PipelineContext(workspace_dir=tmp_path)
    context.state.model = SimpleNamespace(name="toy-model")
    context.state.tokenizer = SimpleNamespace(name="toy-tokenizer")

    registry = PluginRegistry()
    registry.register(TrackingPlugin, "TrackingPlugin")

    step = WorkflowStep(
        name="track",
        plugin="TrackingPlugin",
        config={"runtime_value": "runtime", "target_layers": ["layers.0"]},
    )
    workflow = Workflow(
        name="workflow",
        steps=[step],
        plugin_configs={"TrackingPlugin": {"constructor_value": "constructor", "constructor_only": True}},
    )

    executor = WorkflowExecutor(context, registry)
    results = executor.execute_workflow(workflow)

    plugin = TrackingPlugin.last_instance
    assert plugin is not None
    assert plugin.constructor_value == "constructor"
    assert plugin.constructor_kwargs["constructor_only"] is True
    assert plugin.constructor_kwargs["name"] == "TrackingPlugin"
    assert len(plugin.calls) == 1

    call = plugin.calls[0]
    assert call["context"] is context
    assert call["model"] is context.state.model
    assert call["tokenizer"] is context.state.tokenizer
    assert call["kwargs"]["runtime_value"] == "runtime"
    assert call["kwargs"]["target_layers"] == ["layers.0"]
    assert "constructor_only" not in call["kwargs"]
    assert results["track"]["constructor_value"] == "constructor"
