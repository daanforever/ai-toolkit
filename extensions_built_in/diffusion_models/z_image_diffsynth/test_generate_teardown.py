"""CPU tests: post-sample VAE park, sampling D2H, train DiT remount.

No 6B weights. Stubs with device_torch=cuda mock _flush_cuda (no empty_cache/synchronize).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from toolkit.models.base_model import BaseModel


def _zimage():
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    return ZImageDiffSynthModel, ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)


def _vae_stub(*, on_cuda=True):
    class _Vae:
        def __init__(self):
            self.device = torch.device("cuda:0") if on_cuda else torch.device("cpu")
            self.to_calls = []
            self._p = SimpleNamespace(device=self.device)

        def parameters(self):
            return iter([self._p])

        def to(self, device):
            self.to_calls.append(device)
            target = device if isinstance(device, torch.device) else torch.device(device)
            self.device = target
            self._p.device = target

    return _Vae()


def _order_flush_and_moves(model, order):
    model._flush_cuda = lambda: order.append("flush")
    model._move_sampling_transformer = lambda device: order.append(
        ("move_sampling", str(device) if not isinstance(device, str) else device)
    )
    model._place_training_dit = lambda device: order.append(("place_dit", device))


def _cuda_generate_stub(*, train_on_turbo=False, batch=False):
    cls, model = _zimage()
    model.device_torch = torch.device("cuda")
    model.torch_dtype = torch.float32
    model._train_on_turbo = train_on_turbo
    model._sampling_transformer = object()
    model._sampling_network = None
    model._sampling_in_batch_generate = batch
    model.network = None
    model.model = MagicMock()
    model.print_and_status_update = lambda *a, **k: None
    model._log_device_state = lambda *a, **k: None
    model._flush_cuda = lambda: None
    return cls, model


def _pipeline_and_config():
    img = object()
    pipeline = MagicMock()
    pipeline.return_value = SimpleNamespace(images=[img])
    gen_config = SimpleNamespace(
        width=64,
        height=64,
        num_inference_steps=1,
        guidance_scale=1.0,
        latents=None,
    )
    embeds = SimpleNamespace(text_embeds=torch.randn(4, 8), attention_mask=None)
    return pipeline, gen_config, embeds, img


def _call_generate_single(cls, model):
    pipeline, gen_config, embeds, img = _pipeline_and_config()
    out = cls.generate_single_image(
        model,
        pipeline,
        gen_config,
        embeds,
        embeds,
        torch.Generator(),
        {},
    )
    return out, img


def test_release_generate_workspace_flush_vae_cpu_flush():
    cls, model = _cuda_generate_stub()
    order = []
    vae = _vae_stub(on_cuda=True)
    orig_to = vae.to

    def _to(device):
        order.append(("vae.to", device))
        orig_to(device)

    vae.to = _to
    model.vae = vae
    _order_flush_and_moves(model, order)

    cls._release_generate_workspace(model)

    assert order == ["flush", ("vae.to", "cpu"), "flush"]


def test_release_generate_workspace_second_flush_when_vae_already_cpu():
    cls, model = _cuda_generate_stub()
    order = []
    vae = _vae_stub(on_cuda=False)
    orig_to = vae.to

    def _to(device):
        order.append(("vae.to", device))
        orig_to(device)

    vae.to = _to
    model.vae = vae
    _order_flush_and_moves(model, order)

    cls._release_generate_workspace(model)

    assert order == ["flush", "flush"]
    assert vae.to_calls == []


def test_unload_parks_vae_then_sampling_cpu_then_flush():
    cls, model = _cuda_generate_stub(train_on_turbo=False)
    order = []
    vae = _vae_stub(on_cuda=True)
    orig_to = vae.to

    def _to(device):
        order.append(("vae.to", device))
        orig_to(device)

    vae.to = _to
    model.vae = vae
    model.print_and_status_update = lambda msg: order.append(("status", str(msg)))
    _order_flush_and_moves(model, order)

    cls._unload_sampling_transformer_after_generate(model)

    assert order == [
        "flush",
        ("vae.to", "cpu"),
        "flush",
        ("move_sampling", "cpu"),
        "flush",
        ("status", "\nUnloaded sampling transformer to CPU"),
    ]


def test_unload_skips_when_train_on_turbo():
    cls, model = _cuda_generate_stub(train_on_turbo=True)
    order = []
    vae = _vae_stub(on_cuda=True)
    orig_to = vae.to

    def _to(device):
        order.append(("vae.to", device))
        orig_to(device)

    vae.to = _to
    model.vae = vae
    model.print_and_status_update = lambda msg: order.append(("status", str(msg)))
    _order_flush_and_moves(model, order)

    cls._unload_sampling_transformer_after_generate(model)

    assert order == []
    assert vae.to_calls == []


def test_batch_generate_flush_then_apply_false(monkeypatch):
    monkeypatch.setattr(BaseModel, "generate_images", lambda self, configs, sampler=None: [])
    cls, model = _cuda_generate_stub(train_on_turbo=False)
    order = []
    model._flush_cuda = lambda: order.append("flush")
    model.apply_turbo_teacher_mode = lambda enabled: order.append(("apply", enabled))
    model._move_sampling_transformer = lambda device: order.append(
        ("move_sampling", device)
    )

    cls.generate_images(model, [])

    assert order == ["flush", ("apply", False)]


def test_batch_generate_train_on_turbo_apply_true_no_sampling_cpu(monkeypatch):
    monkeypatch.setattr(BaseModel, "generate_images", lambda self, configs, sampler=None: [])
    cls, model = _cuda_generate_stub(train_on_turbo=True)
    order = []
    model._flush_cuda = lambda: order.append("flush")
    model.apply_turbo_teacher_mode = lambda enabled: order.append(("apply", enabled))
    model._move_sampling_transformer = lambda device: order.append(
        ("move_sampling", device)
    )

    cls.generate_images(model, [])

    assert order == [("apply", True)]


def test_generate_single_image_batch_no_sampling_moves_or_flush():
    cls, model = _cuda_generate_stub(batch=True)
    moves = []
    flushes = []
    model._move_sampling_transformer = lambda device: moves.append(device)
    model._flush_cuda = lambda: flushes.append(1)
    model.apply_turbo_teacher_mode = lambda enabled: None

    out1, img1 = _call_generate_single(cls, model)
    out2, img2 = _call_generate_single(cls, model)

    assert out1 is img1 and out2 is img2
    assert moves == []
    assert flushes == []


def test_move_main_network_logs_gpu_when_need_move():
    cls, model = _cuda_generate_stub()
    statuses = []
    model.print_and_status_update = lambda msg: statuses.append(str(msg))
    model._place_training_dit = lambda device: True
    model.network = None
    dit = nn.Linear(2, 2, bias=False)
    dit.weight.requires_grad_(False)
    model._raw_dit = dit

    cls._move_main_network(model, torch.device("cuda"))

    assert any("Moving main transformer to GPU" in s for s in statuses)


def test_move_main_network_silent_when_already_cuda():
    cls, model = _cuda_generate_stub()
    statuses = []
    model.print_and_status_update = lambda msg: statuses.append(str(msg))
    model._place_training_dit = lambda device: False
    model.network = None
    model._raw_dit = object()
    model._first_frozen_base_param = lambda module: SimpleNamespace(
        device=torch.device("cuda:0")
    )

    cls._move_main_network(model, torch.device("cuda:0"))

    assert not any("Moving main transformer to GPU" in s for s in statuses)


def test_standalone_finally_parks_vae_then_restores_prev():
    cls, model = _cuda_generate_stub(train_on_turbo=False, batch=False)
    order = []
    vae = _vae_stub(on_cuda=True)
    prev = vae.device
    orig_to = vae.to

    def _to(device):
        order.append(("vae.to", device))
        orig_to(device)

    vae.to = _to
    model.vae = vae
    model._flush_cuda = lambda: order.append("flush")
    model._move_sampling_transformer = lambda device: order.append(
        ("move_sampling", device)
    )
    model.apply_turbo_teacher_mode = lambda enabled: order.append(("apply", enabled))

    _call_generate_single(cls, model)

    vae_cpu = next(i for i, c in enumerate(order) if c == ("vae.to", "cpu"))
    apply_false = next(i for i, c in enumerate(order) if c == ("apply", False))
    vae_prev = next(i for i, c in enumerate(order) if c == ("vae.to", prev))
    assert vae_cpu < apply_false < vae_prev
    assert sum(1 for c in order if c[0] == "apply") == 1
