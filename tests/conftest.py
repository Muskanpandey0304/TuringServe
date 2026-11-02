"""Shared fixtures and marker handling (Build Guide 0.6)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))
        if "model" in item.keywords and not (os.environ.get("TS_MODEL_05")
                                             or os.environ.get("TS_MODEL_15")):
            item.add_marker(pytest.mark.skip(reason="needs TS_MODEL_05 / TS_MODEL_15"))
        if "kaggle" in item.keywords and os.environ.get("TS_KAGGLE") != "1":
            item.add_marker(pytest.mark.skip(reason="needs the Kaggle T4 (TS_KAGGLE=1)"))


def _tiny(variant: str) -> str:
    out = os.path.join(ROOT, "tests", "assets", f"tiny_{variant}")
    if not os.path.exists(os.path.join(out, "model.safetensors")):
        from make_tiny_model import make

        make(out, variant)
    return out


@pytest.fixture(scope="session")
def tiny_model_dir() -> str:
    return _tiny("g2")


@pytest.fixture(scope="session")
def tiny_model_g7_dir() -> str:
    return _tiny("g7")


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)
