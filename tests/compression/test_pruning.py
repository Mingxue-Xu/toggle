from pathlib import Path

import pytest

from goldcrest.plugins.compression import pruning as pruning_module


def test_load_pruning_config_rejects_missing_report_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pruning_module.ConfigurationLoader,
        "load_config",
        lambda self, path: {"pruning": {"selection_metric_type": "l2_norm.median"}},
    )

    with pytest.raises(ValueError, match="pruning.report_path"):
        pruning_module.load_pruning_config(tmp_path / "config.yaml")


def test_load_pruning_config_rejects_empty_report_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pruning_module.ConfigurationLoader,
        "load_config",
        lambda self, path: {"pruning": {"report_path": "", "selection_metric_type": "l2_norm.median"}},
    )

    with pytest.raises(ValueError, match="pruning.report_path"):
        pruning_module.load_pruning_config(tmp_path / "config.yaml")


def test_load_pruning_config_accepts_valid_report_path(monkeypatch, tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")

    monkeypatch.setattr(
        pruning_module.ConfigurationLoader,
        "load_config",
        lambda self, path: {
            "pruning": {
                "report_path": str(report_path),
                "selection_metric_type": "l2_norm.median",
                "group_prefix": "layers",
            }
        },
    )

    cfg = pruning_module.load_pruning_config(tmp_path / "config.yaml")

    assert cfg.report_path == Path(report_path)
    assert cfg.selection_metric_type == "l2_norm.median"
