import importlib


def test_compressed_eval_imports_without_name_errors():
    module = importlib.import_module("goldcrest.plugins.evaluation.compressed_eval")

    assert hasattr(module, "CompressedGoldcrestAdapter")
    assert hasattr(module, "CompressedModelProfile")


def test_compressed_eval_package_exports_profile_class():
    from goldcrest.plugins.evaluation import CompressedModelProfile

    assert CompressedModelProfile.__name__ == "CompressedModelProfile"
