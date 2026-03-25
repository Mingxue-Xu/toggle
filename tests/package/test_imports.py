import importlib


def test_package_imports_work_without_src_module_errors():
    modules = [
        "goldcrest.plugins",
        "goldcrest.framework",
        "goldcrest.orchestration",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_evaluation_and_loader_exports_import_cleanly():
    from goldcrest.plugins.evaluation import CompressedModelProfile
    from goldcrest.plugins.models.loader import HuggingFaceModelLoader

    assert CompressedModelProfile.__name__ == "CompressedModelProfile"
    assert HuggingFaceModelLoader.__name__ == "HuggingFaceModelLoader"
