from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from toggle.src.framework.context import PipelineContext
from toggle.src.plugins.compression.calibration_collector import CalibrationCollectorPlugin


class TwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            raise AssertionError("attention_mask must be preserved")
        return self.layer(input_ids.float())


def test_calibration_counts_examples_in_dict_batches(tmp_path):
    context = PipelineContext(workspace_dir=tmp_path)
    model = TwoInputModel()
    context.state.model = model

    dataset = [
        {"input_ids": torch.ones(4), "attention_mask": torch.ones(4)},
        {"input_ids": torch.ones(4), "attention_mask": torch.ones(4)},
        {"input_ids": torch.ones(4), "attention_mask": torch.ones(4)},
    ]
    dataloader = DataLoader(dataset, batch_size=2)

    plugin = CalibrationCollectorPlugin(n_samples=3)
    plugin.initialize(context)

    result = plugin.execute(model=model, dataloader=dataloader)

    assert result["samples"] == 3
    assert context.state.get("calibration.sample_count") == 3
    assert context.state.get("calibration.collected") is True
    assert context.state.get("calibration.xtx.layer") is not None


def test_calibration_tuple_batches_preserve_all_inputs(tmp_path):
    context = PipelineContext(workspace_dir=tmp_path)
    model = TwoInputModel()
    context.state.model = model

    dataset = [
        (torch.ones(4), torch.ones(4)),
        (torch.ones(4), torch.ones(4)),
    ]
    dataloader = DataLoader(dataset, batch_size=2)

    plugin = CalibrationCollectorPlugin(n_samples=2)
    plugin.initialize(context)

    result = plugin.execute(model=model, dataloader=dataloader)

    assert result["samples"] == 2
    assert context.state.get("calibration.sample_count") == 2
