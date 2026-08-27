"""Unit tests for text-encoder unload stash / reload and text-cache residency."""

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
    enter_text_cache_residency,
    exit_text_cache_residency,
    reload_text_encoder,
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


def _make_residency_model(*, train_on_turbo: bool = False):
    device = torch.device("cpu")
    te = nn.Linear(2, 2)
    unet = nn.Linear(2, 2)
    network = nn.Linear(2, 2)
    sampling = nn.Linear(2, 2)
    order = []

    def text_encoder_to(dev, *a, **k):
        order.append(("text_encoder_to", str(torch.device(dev))))
        te.to(dev)

    def place(dev):
        order.append(("_place_training_dit", str(torch.device(dev))))
        unet.to(dev)
        return True

    def move_main(dev):
        order.append(("_move_main_network", str(torch.device(dev))))
        unet.to(dev)
        network.to(dev)

    def move_sampling(dev):
        order.append(("_move_sampling_transformer", str(torch.device(dev))))
        sampling.to(dev)

    apply = MagicMock(
        side_effect=lambda enabled: order.append(("apply_turbo_teacher_mode", str(bool(enabled))))
    )

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=unet,
        network=network,
        vae=nn.Linear(2, 2),
        adapter=None,
        refiner_unet=None,
        _sampling_transformer=sampling,
        _sampling_network=None,
        text_encoder_to=text_encoder_to,
        _place_training_dit=place,
        _move_main_network=move_main,
        _move_sampling_transformer=move_sampling,
        apply_turbo_teacher_mode=apply,
        _train_on_turbo=train_on_turbo,
        _order=order,
    )
    return model


def test_enter_calls_te_to_cpu_before_non_te_owners():
    model = _make_residency_model()
    enter_text_cache_residency(model)
    labels = [x[0] for x in model._order]
    assert labels[0] == "text_encoder_to"
    assert model._order[0][1] == "cpu"
    assert "_place_training_dit" in labels
    te_cpu = next(i for i, x in enumerate(model._order) if x[0] == "text_encoder_to" and x[1] == "cpu")
    place_idx = next(i for i, x in enumerate(model._order) if x[0] == "_place_training_dit")
    assert place_idx > te_cpu


def test_enter_uses_place_training_dit_when_present():
    model = _make_residency_model()
    enter_text_cache_residency(model)
    assert ("_place_training_dit", "cpu") in model._order


def test_exit_uses_move_main_network_when_present():
    model = _make_residency_model(train_on_turbo=False)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    model._order.clear()
    exit_text_cache_residency(model)
    assert ("_move_main_network", "cpu") in model._order
    assert not any(x[0] == "apply_turbo_teacher_mode" for x in model._order)


def test_exit_falls_back_to_unet_move_without_move_main_network():
    unet = nn.Linear(2, 2)
    te = nn.Linear(2, 2)
    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=torch.device("cpu"),
        torch_dtype=torch.float32,
        unet=unet,
        network=None,
        vae=None,
        adapter=None,
        refiner_unet=None,
        _sampling_transformer=None,
        _sampling_network=None,
        text_encoder_to=lambda dev, *a, **k: te.to(dev),
        _train_on_turbo=False,
    )
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    model.device_torch = torch.device("cpu")
    exit_text_cache_residency(model)
    assert _param_device(unet).type == "cpu"


def test_exit_uses_turbo_teacher_mode_when_train_on_turbo():
    model = _make_residency_model(train_on_turbo=True)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    model._order.clear()
    model.apply_turbo_teacher_mode.reset_mock()
    exit_text_cache_residency(model)
    model.apply_turbo_teacher_mode.assert_called_once_with(True)
    assert not any(x[0] == "_move_main_network" for x in model._order)


def test_enter_offloads_network_and_sampling():
    """Enter must offload network and sampling transformer (not TE+DiT only)."""
    model = _make_residency_model()
    network = model.network
    sampling = model._sampling_transformer
    enter_text_cache_residency(model)
    assert _param_device(network).type == "cpu"
    assert _param_device(sampling).type == "cpu"
    assert ("_move_sampling_transformer", "cpu") in model._order


def test_unload_second_te_move_failure_propagates_no_fake(monkeypatch):
    """Multi-TE: second-owner move failure propagates; no Fake replacement."""
    import pytest
    from toolkit.util import device as device_mod

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to force a real TE device transition")

    te1 = nn.Linear(4, 4).to("cuda")
    te2 = nn.Linear(4, 4).to("cuda")
    model = SimpleNamespace(
        text_encoder=[te1, te2],
        pipeline=None,
        device_torch=torch.device("cuda"),
        torch_dtype=torch.float32,
    )
    real_safe = device_mod.safe_module_to_device

    def boom(module, device, dtype=None):
        if module is te2:
            raise RuntimeError("simulated te2 move failure")
        return real_safe(module, device, dtype)

    monkeypatch.setattr("toolkit.unloader.safe_module_to_device", boom)

    with pytest.raises(RuntimeError) as ei:
        unload_text_encoder(model)
    msg = str(ei.value)
    assert "text_encoder[1]" in msg
    assert "cuda" in msg.lower()
    assert "cpu" in msg.lower()
    assert "simulated te2 move failure" in msg
    assert model.text_encoder[0] is te1
    assert model.text_encoder[1] is te2
    assert not isinstance(model.text_encoder[0], FakeTextEncoder)
    assert not isinstance(model.text_encoder[1], FakeTextEncoder)
    assert getattr(model, "_real_text_encoder", None) is None


def test_unload_pipeline_only_te_move_failure_propagates(monkeypatch):
    """Pipeline TE distinct from model TE: move failure propagates before Fake."""
    import pytest
    from toolkit.util import device as device_mod

    if not torch.cuda.is_available():
        pytest.skip("CUDA required to force a real TE device transition")

    model_te = nn.Linear(4, 4).to("cpu")
    pipe_te = nn.Linear(4, 4).to("cuda")
    pipe = SimpleNamespace(text_encoder=pipe_te)
    model = SimpleNamespace(
        text_encoder=model_te,
        pipeline=pipe,
        device_torch=torch.device("cuda"),
        torch_dtype=torch.float32,
    )
    real_safe = device_mod.safe_module_to_device

    def boom(module, device, dtype=None):
        if module is pipe_te:
            raise RuntimeError("simulated pipeline te move failure")
        return real_safe(module, device, dtype)

    monkeypatch.setattr("toolkit.unloader.safe_module_to_device", boom)

    with pytest.raises(RuntimeError) as ei:
        unload_text_encoder(model)
    msg = str(ei.value)
    assert "pipeline.text_encoder" in msg
    assert "simulated pipeline te move failure" in msg
    assert model.text_encoder is model_te
    assert not isinstance(model.text_encoder, FakeTextEncoder)
    assert pipe.text_encoder is pipe_te
    assert getattr(model, "_real_text_encoder", None) is None
    assert getattr(model, "_real_pipeline_text_encoders", None) is None


def test_unload_dedupes_model_and_pipeline_alias():
    """Shared model/pipeline TE object is moved once (still ends Fake+stash)."""
    model, te, te2 = _make_model(
        te_on_cuda=torch.cuda.is_available(), as_list=True, with_pipeline=True
    )
    # Aliases: pipeline refs are the same objects as list entries.
    assert model.pipeline.text_encoder is te
    unload_text_encoder(model)
    assert isinstance(model.text_encoder[0], FakeTextEncoder)
    assert _param_device(te).type == "cpu"
    assert model._real_pipeline_text_encoders["text_encoder"] is te
