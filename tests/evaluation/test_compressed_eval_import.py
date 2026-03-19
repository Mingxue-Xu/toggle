import importlib


def test_compressed_eval_imports_without_name_errors():
    module = importlib.import_module("toggle.src.plugins.evaluation.compressed_eval")

    assert hasattr(module, "CompressedToggleAdapter")
    assert hasattr(module, "CompressedModelProfile")


def test_compressed_eval_package_exports_profile_class():
    from toggle.src.plugins.evaluation import CompressedModelProfile

    assert CompressedModelProfile.__name__ == "CompressedModelProfile"
