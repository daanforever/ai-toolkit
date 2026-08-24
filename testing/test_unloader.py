"""Unit tests for text-encoder unload stash / reload."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from toolkit.unloader import FakeTextEncoder, reload_text_encoder, unload_text_encoder


def _param_device(module: nn.Module) -> torch.device:
    p = next(module.parameters())
    return p.device


def _make_model(*, te_on_cuda: bool = False, as_list: bool = False, with_pipeline: bool = False):
    device = torch.device("cuda" if te_on_cuda and torch.cuda.is_available() else "cpu")
    te = nn.Linear(4, 4)
    te2 = nn.Linear(4, 4)
    te = te.to(device)
    te2 = te2.to(device)
    pipe = None
    if with_pipeline:
        pipe = SimpleNamespace(text_encoder=te, text_encoder_2=te2)
    model = SimpleNamespace(
        text_encoder=[te, te2] if as_list else te,
        pipeline=pipe,
        device_torch=device,
        torch_dtype=torch.float32,
    )
    return model, te, te2


def test_unload_stashes_cpu_module_and_installs_fake():
    model, te, _ = _make_model(te_on_cuda=torch.cuda.is_available(), as_list=False)
    unload_text_encoder(model)
    assert isinstance(model.text_encoder, FakeTextEncoder)
    assert model._real_text_encoder is te
    assert _param_device(model._real_text_encoder).type == "cpu"


def test_unload_list_and_pipeline_become_fake():
    model, te, te2 = _make_model(
        te_on_cuda=torch.cuda.is_available(), as_list=True, with_pipeline=True
    )
    unload_text_encoder(model)
    assert isinstance(model.text_encoder, list)
    assert all(isinstance(x, FakeTextEncoder) for x in model.text_encoder)
    assert isinstance(model.pipeline.text_encoder, FakeTextEncoder)
    assert isinstance(model.pipeline.text_encoder_2, FakeTextEncoder)
    assert model._real_text_encoder[0] is te
    assert model._real_text_encoder[1] is te2
    assert _param_device(te).type == "cpu"
    assert _param_device(te2).type == "cpu"
    stash = model._real_pipeline_text_encoders
    assert stash["text_encoder"] is te
    assert stash["text_encoder_2"] is te2


def test_second_unload_does_not_replace_stash_with_fakes():
    model, te, _ = _make_model(as_list=False)
    unload_text_encoder(model)
    first_stash = model._real_text_encoder
    unload_text_encoder(model)
    assert model._real_text_encoder is first_stash
    assert model._real_text_encoder is te
    assert not isinstance(model._real_text_encoder, FakeTextEncoder)


def test_reload_restores_identity_on_model_and_pipeline():
    model, te, te2 = _make_model(as_list=True, with_pipeline=True)
    unload_text_encoder(model)
    reload_text_encoder(model)
    assert model.text_encoder[0] is te
    assert model.text_encoder[1] is te2
    assert model.pipeline.text_encoder is te
    assert model.pipeline.text_encoder_2 is te2


def test_reload_then_unload_back_to_cpu_fake():
    model, te, _ = _make_model(
        te_on_cuda=torch.cuda.is_available(), as_list=False, with_pipeline=True
    )
    unload_text_encoder(model)
    reload_text_encoder(model)
    if torch.cuda.is_available():
        te.to("cuda")
        assert _param_device(te).type == "cuda"
    unload_text_encoder(model)
    assert isinstance(model.text_encoder, FakeTextEncoder)
    assert model._real_text_encoder is te
    assert _param_device(te).type == "cpu"
