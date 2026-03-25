"""
Shared fixtures for end-to-end tests on google/gemma-3-270m-it.

All tests in this directory perform ACTUAL model compression on a real
HuggingFace model.  The model is loaded once per session to save time.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldcrest.plugins.evaluation.csv_logger import CSVLogger, ResultComparator

MODEL_ID = "google/gemma-3-270m-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # float32 for deterministic CPU decomposition
LOG_DIR = Path("logging/logs")


# ---------------------------------------------------------------------------
# Session-scoped: load model + tokenizer once, reused across ALL tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def hf_token():
    return os.environ.get("HF_TOKEN")


@pytest.fixture(scope="session")
def base_tokenizer(hf_token):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=hf_token,
    )


@pytest.fixture(scope="session")
def base_model(hf_token):
    """Load the model once and keep it pristine for per-test deep copies."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=DTYPE, token=hf_token,
    )
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Function-scoped: fresh model copy for each test
# ---------------------------------------------------------------------------
@pytest.fixture()
def model_and_tokenizer(base_model, base_tokenizer):
    """Yield an isolated model/tokenizer pair for each test."""
    model = copy.deepcopy(base_model)
    model.to(DEVICE).eval()
    return model, base_tokenizer


@pytest.fixture()
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture()
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture()
def pipeline_context(tmp_path, model_and_tokenizer):
    """Return a PipelineContext with model + tokenizer pre-loaded."""
    from goldcrest.framework.context import PipelineContext
    model, tokenizer = model_and_tokenizer
    ctx = PipelineContext(config={}, workspace_dir=tmp_path)
    ctx.state.model = model
    ctx.state.tokenizer = tokenizer
    return ctx


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------
def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def run_forward(model, tokenizer, prompt: str = "Hello", max_new: int = 4):
    """Run a quick forward pass / generate to verify the model still works."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def random_input(model, batch: int = 1, seq_len: int = 16):
    """Generate random token-id inputs for quick forward pass."""
    vocab = getattr(getattr(model, "config", None), "vocab_size", 32000)
    ids = torch.randint(0, vocab, (batch, seq_len), device=model.device)
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask}


def size_mb(model) -> float:
    """Compute model size in MB from parameters."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


# ---------------------------------------------------------------------------
# CSV Logging fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def csv_logger():
    """Session-scoped CSV logger writing to the workspace log directory."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(output_dir=str(LOG_DIR))
    return logger


@pytest.fixture(scope="session")
def result_comparator(csv_logger):
    """Session-scoped result comparator for cross-experiment analysis."""
    return ResultComparator(csv_logger)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test pass/fail/skip status on the item for fixture teardown."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _experiment_session(csv_logger, request):
    """Auto-wrap each test in an experiment session for CSV logging."""
    test_name = request.node.name
    csv_logger.start_experiment(test_name, config_file="e2e_test")
    yield
    status = "completed"
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        status = "failed"
    elif hasattr(request.node, "rep_setup") and request.node.rep_setup.skipped:
        status = "skipped"
    csv_logger.end_experiment(status=status)
