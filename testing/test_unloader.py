"""Unit tests for text-encoder unload stash / reload."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from toolkit.unloader import (
    FakeTextEncoder,
    park_main_transformer_for_text_cache,
    reload_text_encoder,
    restore_main_transformer_after_text_cache,
    unload_text_encoder,
)


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


def test_park_calls_te_to_cpu_before_unet_to_cpu():
    order = []
    unet = MagicMock()
    unet.to = MagicMock(side_effect=lambda *a, **k: order.append(("unet.to", a, k)) or unet)

    def text_encoder_to(*a, **k):
        order.append(("text_encoder_to", a, k))

    model = SimpleNamespace(
        text_encoder=nn.Linear(2, 2),
        text_encoder_to=text_encoder_to,
        unet=unet,
        device_torch=torch.device("cpu"),
    )
    park_main_transformer_for_text_cache(model)
    assert [x[0] for x in order] == ["text_encoder_to", "unet.to"]
    assert order[0][1] == ("cpu",)
    assert order[1][1] == ("cpu",)


def test_park_uses_place_training_dit_when_present():
    order = []

    def text_encoder_to(*a, **k):
        order.append("te")

    def place(device):
        order.append(("place", str(device)))
        return True

    model = SimpleNamespace(
        text_encoder=nn.Linear(2, 2),
        text_encoder_to=text_encoder_to,
        _place_training_dit=place,
        unet=MagicMock(),
        device_torch=torch.device("cpu"),
    )
    park_main_transformer_for_text_cache(model)
    assert order == ["te", ("place", "cpu")]
    model.unet.to.assert_not_called()


def test_restore_uses_move_main_network_when_present():
    move = MagicMock()
    unet = MagicMock()
    model = SimpleNamespace(
        _move_main_network=move,
        unet=unet,
        device_torch=torch.device("cuda:0"),
    )
    restore_main_transformer_after_text_cache(model, torch.device("cuda:0"))
    move.assert_called_once_with(torch.device("cuda:0"))
    unet.to.assert_not_called()


def test_restore_falls_back_to_unet_to():
    unet = MagicMock()
    model = SimpleNamespace(unet=unet, device_torch=torch.device("cpu"))
    restore_main_transformer_after_text_cache(model, "cuda")
    unet.to.assert_called_once_with("cuda")


def test_restore_only_when_caching_text_embeddings_gate():
    """Mirror SDTrainer: restore runs only if is_caching_text_embeddings."""
    move = MagicMock()
    model = SimpleNamespace(_move_main_network=move, device_torch=torch.device("cpu"))

    is_caching_text_embeddings = False
    unload_text_encoder_flag = True
    if is_caching_text_embeddings:
        restore_main_transformer_after_text_cache(model, model.device_torch)
    move.assert_not_called()

    is_caching_text_embeddings = True
    if unload_text_encoder_flag or is_caching_text_embeddings:
        if is_caching_text_embeddings:
            restore_main_transformer_after_text_cache(model, model.device_torch)
    move.assert_called_once_with(model.device_torch)


def test_restore_uses_turbo_teacher_mode_when_train_on_turbo():
    apply = MagicMock()
    move = MagicMock()
    model = SimpleNamespace(
        _train_on_turbo=True,
        apply_turbo_teacher_mode=apply,
        _move_main_network=move,
        device_torch=torch.device("cuda:0"),
    )
    restore_main_transformer_after_text_cache(model, torch.device("cuda:0"))
    apply.assert_called_once_with(True)
    move.assert_not_called()


def test_restore_uses_prefer_turbo_flag_without_train_on_turbo():
    apply = MagicMock()
    move = MagicMock()
    model = SimpleNamespace(
        _train_on_turbo=False,
        _prefer_turbo_restore_after_te=True,
        apply_turbo_teacher_mode=apply,
        _move_main_network=move,
        device_torch=torch.device("cuda:0"),
    )
    restore_main_transformer_after_text_cache(model, torch.device("cuda:0"))
    apply.assert_called_once_with(True)
    move.assert_not_called()
    assert model._prefer_turbo_restore_after_te is False
