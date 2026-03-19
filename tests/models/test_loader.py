import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from toggle.src.plugins.models.loader import HuggingFaceModelLoader


class FakeTokenizer:
    def __init__(self):
        self.seen_kwargs = None


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def test_huggingface_loader_normalizes_auth_and_uses_causal_lm(monkeypatch):
    tokenizer_calls = {}
    model_calls = {}

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        tokenizer_calls["model_name"] = model_name
        tokenizer_calls["kwargs"] = dict(kwargs)
        return FakeTokenizer()

    def fake_causal_lm_from_pretrained(model_name, **kwargs):
        model_calls["model_name"] = model_name
        model_calls["kwargs"] = dict(kwargs)
        return FakeCausalLM()

    def fail_if_auto_model_used(*args, **kwargs):
        raise AssertionError("AutoModel should not be used for the main loader path")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_causal_lm_from_pretrained)
    monkeypatch.setattr(AutoModel, "from_pretrained", fail_if_auto_model_used)

    loader = HuggingFaceModelLoader(model_name="toy-model", device="cpu")
    model, tokenizer = loader.load_model(
        "toy-model",
        hf_token="hf-secret",
        token="legacy-token",
        use_auth_token="old-token",
        cache_dir="/tmp/cache",
        trust_remote_code=False,
    )

    assert isinstance(tokenizer, FakeTokenizer)
    assert tokenizer_calls["model_name"] == "toy-model"
    assert model_calls["model_name"] == "toy-model"
    assert "hf_token" not in tokenizer_calls["kwargs"]
    assert "hf_token" not in model_calls["kwargs"]
    assert "cache_dir" in tokenizer_calls["kwargs"]
    assert "cache_dir" in model_calls["kwargs"]
    assert not ("token" in tokenizer_calls["kwargs"] and "use_auth_token" in tokenizer_calls["kwargs"])
    assert not ("token" in model_calls["kwargs"] and "use_auth_token" in model_calls["kwargs"])
    assert model_calls["kwargs"]["torch_dtype"] == torch.float32
    assert model.moved_to == "cpu"
