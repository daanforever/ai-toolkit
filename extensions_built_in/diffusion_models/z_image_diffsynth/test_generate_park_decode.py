"""CPU tests: wrapper latent/pil contract and generate park/flush/decode order."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from PIL import Image

from toolkit.config_modules import GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds


STANDALONE_BEGIN = (
    "[zimage_diffsynth] standalone sampling: moving main transformer to "
    "CPU and sampling transformer to GPU"
)
STANDALONE_END = (
    "[zimage_diffsynth] standalone sampling: restoring train DiT residency"
)


def _fake_run_forward(dit, latents, t, cond_emb, model_dtype=None, **kwargs):
    return torch.zeros_like(latents)


class _DummyDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.in_channels = 16


class _DummyVae(nn.Module):
    def __init__(self, *, scaling_factor=1.0, shift_factor=0.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )
        self.decode_calls = 0
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def decode(self, z):
        self.decode_calls += 1
        b, _, h, w = z.shape
        return SimpleNamespace(sample=torch.zeros(b, 3, h, w, dtype=z.dtype))


def _wrapper(vae=None, device=None, dtype=torch.float32):
    from extensions_built_in.diffusion_models.z_image_diffsynth.sampling import (
        ZImageDiffSynthPipelineWrapper,
    )

    return ZImageDiffSynthPipelineWrapper(
        dit=_DummyDit(),
        vae=vae if vae is not None else _DummyVae(),
        tokenizer=None,
        text_encoder=None,
        device=device if device is not None else torch.device("cpu"),
        dtype=dtype,
    )


def test_wrapper_output_type_latent_is_rank4_tensor(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod.fwd_mod, "run_forward", _fake_run_forward)
    vae = _DummyVae()
    pipe = _wrapper(vae=vae)
    init = torch.full((1, 16, 4, 4), 0.25)
    out = pipe(
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=0.0,
        latents=init.clone(),
        output_type="latent",
    )
    assert torch.is_tensor(out.images)
    assert out.images.dim() == 4
    assert out.images.shape == init.shape
    assert torch.allclose(out.images, init)
    assert vae.decode_calls == 0
    assert vae.to_calls == []


def test_wrapper_output_type_pil_default_is_pil(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod.fwd_mod, "run_forward", _fake_run_forward)
    vae = _DummyVae()
    pipe = _wrapper(vae=vae)
    init = torch.zeros(1, 16, 4, 4)
    out = pipe(
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=0.0,
        latents=init,
    )
    assert isinstance(out.images, list)
    assert isinstance(out.images[0], Image.Image)
    assert vae.decode_calls == 1
    # zeros → (x/2+0.5).clamp → 0.5 → round(127.5)=128
    assert out.images[0].getpixel((0, 0)) == (128, 128, 128)


def test_wrapper_output_type_pil_explicit(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth import sampling as sampling_mod

    monkeypatch.setattr(sampling_mod.fwd_mod, "run_forward", _fake_run_forward)
    pipe = _wrapper()
    out = pipe(
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=0.0,
        latents=torch.zeros(1, 16, 4, 4),
        output_type="pil",
    )
    assert isinstance(out.images[0], Image.Image)


def _stub_model(*, batch=True, sampling_on="cuda"):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    events = []
    logs = []
    residency = {"dev": sampling_on}

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cuda")
    model.torch_dtype = torch.float32
    model._sampling_transformer = object()
    model._sampling_in_batch_generate = batch
    model._train_on_turbo = False
    model.model = SimpleNamespace(to=MagicMock())
    model.print_and_status_update = lambda msg: logs.append(str(msg))

    def _move(device):
        target = device if isinstance(device, torch.device) else torch.device(device)
        moved = residency["dev"] != target.type
        events.append(("move", target.type, moved))
        residency["dev"] = target.type
        return moved

    def _flush():
        events.append(("flush",))

    model._move_sampling_transformer = _move
    model._flush_cuda = _flush
    model.apply_turbo_teacher_mode = MagicMock()
    return model, events, logs, residency


def _embeds():
    return PromptEmbeds(torch.zeros(4, 8)), PromptEmbeds(torch.zeros(4, 8))


def _gen_config():
    return GenerateImageConfig(
        width=64,
        height=64,
        num_inference_steps=1,
        guidance_scale=0.0,
        output_path="park_decode.png",
    )


class _HfPipeline:
    def __init__(self, events, images=None):
        self.events = events
        self.kwargs_list = []
        self._images = images
        self.vae = SimpleNamespace(
            dtype=torch.float32,
            config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
            decode=self._decode,
        )
        self.image_processor = SimpleNamespace(postprocess=self._postprocess)

    def _decode(self, latents, return_dict=False):
        self.events.append(("decode",))
        b = latents.shape[0]
        img = torch.zeros(b, 3, 8, 8)
        if return_dict:
            return SimpleNamespace(sample=img)
        return (img,)

    def _postprocess(self, image, output_type="pil"):
        self.events.append(("postprocess", output_type))
        return [Image.new("RGB", (8, 8))]

    def __call__(self, output_type="pil", **kwargs):
        kwargs["output_type"] = output_type
        self.events.append(("pipeline",))
        self.kwargs_list.append(kwargs)
        images = self._images
        if images is None:
            images = torch.zeros(1, 16, 2, 2)
        return SimpleNamespace(images=images)


class _WrapperPipeline:
    def __init__(self, events):
        self.events = events
        self.kwargs_list = []

    def decode_latents_to_pil(self, latents):
        self.events.append(("wrapper_decode",))
        return [Image.new("RGB", (8, 8))]

    def __call__(self, output_type="pil", **kwargs):
        kwargs["output_type"] = output_type
        self.events.append(("pipeline",))
        self.kwargs_list.append(kwargs)
        return SimpleNamespace(images=torch.zeros(1, 16, 2, 2))


def _run_single(model, pipeline, extra=None):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    cond, uncond = _embeds()
    return ZImageDiffSynthModel.generate_single_image(
        model,
        pipeline,
        _gen_config(),
        cond,
        uncond,
        torch.Generator(),
        extra if extra is not None else {},
    )


def test_batch_two_image_park_flush_decode_order():
    model, events, logs, residency = _stub_model(batch=True, sampling_on="cuda")
    pipe = _HfPipeline(events)

    img0 = _run_single(model, pipe, extra={"output_type": "pil"})
    assert isinstance(img0, Image.Image)
    assert pipe.kwargs_list[0]["output_type"] == "latent"
    assert residency["dev"] == "cpu"
    assert events == [
        ("move", "cuda", False),
        ("pipeline",),
        ("move", "cpu", True),
        ("flush",),
        ("decode",),
        ("postprocess", "pil"),
        ("flush",),
    ]

    events.clear()
    img1 = _run_single(model, pipe)
    assert isinstance(img1, Image.Image)
    assert residency["dev"] == "cpu"
    assert events == [
        ("move", "cuda", True),
        ("pipeline",),
        ("move", "cpu", True),
        ("flush",),
        ("decode",),
        ("postprocess", "pil"),
        ("flush",),
    ]
    assert all(STANDALONE_BEGIN not in msg and STANDALONE_END not in msg for msg in logs)
    assert all("standalone sampling" not in msg for msg in logs)


def test_batch_list_images_takes_first_tensor():
    model, events, logs, _ = _stub_model(batch=True, sampling_on="cuda")
    pipe = _HfPipeline(events, images=[torch.zeros(1, 16, 2, 2), torch.ones(1, 16, 2, 2)])
    img = _run_single(model, pipe)
    assert isinstance(img, Image.Image)
    assert ("decode",) in events


def test_wrapper_pipeline_decode_branch():
    model, events, logs, _ = _stub_model(batch=True, sampling_on="cuda")
    pipe = _WrapperPipeline(events)
    img = _run_single(model, pipe)
    assert isinstance(img, Image.Image)
    assert pipe.kwargs_list[0]["output_type"] == "latent"
    assert ("wrapper_decode",) in events
    assert ("decode",) not in events
    assert all("standalone sampling" not in msg for msg in logs)


def test_park_unmoved_skips_pre_decode_flush():
    model, events, _, _ = _stub_model(batch=True, sampling_on="cuda")

    def _move(device):
        target = device if isinstance(device, torch.device) else torch.device(device)
        events.append(("move", target.type, False))
        return False

    model._move_sampling_transformer = _move
    pipe = _HfPipeline(events)
    img = _run_single(model, pipe)
    assert isinstance(img, Image.Image)
    assert events == [
        ("move", "cuda", False),
        ("pipeline",),
        ("move", "cpu", False),
        ("decode",),
        ("postprocess", "pil"),
        ("flush",),
    ]
