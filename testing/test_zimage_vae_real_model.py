"""
Real-model checks for Z-Image VAE parity and wrapper behavior.

Run (repo root, venv):
  venv\\Scripts\\python.exe -m pytest testing/test_zimage_vae_real_model.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKL

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth.model import (  # noqa: E402
    ZImageDiffSynthModel,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.vae_wrapper import (  # noqa: E402
    DiffSynthVAEWrapper,
)


def _resolve_model_path() -> str:
    return os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH


@pytest.fixture(scope="module")
def vae_bundle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this real-model test")

    model_path = _resolve_model_path()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path is missing. Set ZIMAGE_DIFFSYNTH_MODEL_PATH or update DEFAULT_ZIMAGE_MODEL_PATH."
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.to(device)
    vae.eval()

    vae_wrapper = DiffSynthVAEWrapper(vae, None)
    vae_wrapper.to(device)
    vae_wrapper.eval()

    try:
        yield {
            "vae": vae,
            "vae_wrapper": vae_wrapper,
            "device": device,
            "dtype": dtype,
        }
    finally:
        del vae_wrapper
        del vae
        torch.cuda.empty_cache()


def _make_vae_model(bundle) -> ZImageDiffSynthModel:
    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = bundle["device"]
    model.device = bundle["device"]
    model.torch_dtype = bundle["dtype"]
    model.vae_device_torch = bundle["device"]
    model.vae_torch_dtype = bundle["dtype"]
    model.vae = bundle["vae_wrapper"]
    return model


def test_vae_wrapper_properties(vae_bundle):
    wrapper = vae_bundle["vae_wrapper"]
    device = vae_bundle["device"]
    dtype = vae_bundle["dtype"]

    # Basic metadata
    assert wrapper.device.type == device.type
    assert wrapper.dtype == dtype

    # Config delegation
    assert hasattr(wrapper, "config")
    assert "block_out_channels" in wrapper.config
    assert "scaling_factor" in wrapper.config
    assert "shift_factor" in wrapper.config

    # nn.Module interface tests
    # parameters access
    params = list(wrapper.parameters())
    assert len(params) > 0
    assert all(p.device.type == device.type for p in params)
    assert all(p.dtype == dtype for p in params)

    # test moving to CPU and back
    wrapper.cpu()
    assert wrapper.device.type == "cpu"
    wrapper.to(device)
    assert wrapper.device.type == device.type


def test_vae_direct_parity_encode_decode(vae_bundle):
    vae = vae_bundle["vae"]
    wrapper = vae_bundle["vae_wrapper"]
    device = vae_bundle["device"]
    dtype = vae_bundle["dtype"]

    # Create dummy image in [0, 1] range
    torch.manual_seed(42)
    img = torch.rand((1, 3, 256, 256), device=device, dtype=dtype)
    # VAE usually expects images normalized to [-1, 1]
    normalized_img = 2 * img - 1

    with torch.inference_mode():
        # Encode parity
        res_vae = vae.encode(normalized_img)
        res_wrap = wrapper.encode(normalized_img)

        # Compare latent distribution mode
        latents_vae = res_vae.latent_dist.mode()
        latents_wrap = res_wrap.latent_dist.mode()
        assert torch.allclose(latents_vae, latents_wrap, rtol=1e-5, atol=1e-5)

        # Decode parity
        dec_vae = vae.decode(latents_vae).sample
        dec_wrap = wrapper.decode(latents_wrap).sample
        assert torch.allclose(dec_vae, dec_wrap, rtol=1e-5, atol=1e-5)


def test_zimage_model_encode_images_divisible(vae_bundle):
    bundle = vae_bundle
    device = bundle["device"]
    dtype = bundle["dtype"]
    model = _make_vae_model(bundle)

    # Create images of divisible size (512x512)
    torch.manual_seed(42)
    images = [torch.rand((3, 512, 512), device=device, dtype=dtype) for _ in range(2)]

    with torch.inference_mode():
        latents = model.encode_images(images)

    # 8x spatial downsampling, 16 latent channels
    assert latents.shape == (2, 16, 64, 64)
    assert latents.device.type == device.type
    assert latents.dtype == dtype


def test_zimage_model_encode_images_non_divisible(vae_bundle):
    bundle = vae_bundle
    device = bundle["device"]
    dtype = bundle["dtype"]
    model = _make_vae_model(bundle)

    # Create images of non-divisible size (515x509)
    # VAE scaling factor is 8 (block_out_channels length is 4, 2^(4-1) = 8)
    torch.manual_seed(42)
    images = [torch.rand((3, 515, 509), device=device, dtype=dtype)]

    with torch.inference_mode():
        latents = model.encode_images(images)

    # Divisible by 8: 515 -> 512, 509 -> 504. Divided by 8 -> 64, 63
    assert latents.shape == (1, 16, 64, 63)
    assert latents.device.type == device.type
    assert latents.dtype == dtype


def test_zimage_model_decode_latents_cases(vae_bundle):
    bundle = vae_bundle
    device = bundle["device"]
    dtype = bundle["dtype"]
    model = _make_vae_model(bundle)

    # Case A: True latent tensor (16 channels) -> should use VAE
    latents = torch.randn((1, 16, 32, 32), device=device, dtype=dtype)
    with torch.inference_mode():
        decoded = model.decode_latents(latents)

    # 8x upsampling -> 32*8 = 256, 3 output channels (RGB)
    assert decoded.shape == (1, 3, 256, 256)
    assert decoded.device.type == device.type
    assert decoded.dtype == dtype

    # Case B: RGB preview bypass (3 channels) -> should bypass VAE and preserve shape/device/dtype
    rgb_preview = torch.rand((1, 3, 256, 256), device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        decoded_preview = model.decode_latents(rgb_preview, device=device, dtype=dtype)

    assert decoded_preview.shape == (1, 3, 256, 256)
    assert decoded_preview.device.type == device.type
    assert decoded_preview.dtype == dtype
    assert torch.allclose(decoded_preview, rgb_preview.to(device, dtype=dtype))


def test_vae_determinism(vae_bundle):
    bundle = vae_bundle
    device = bundle["device"]
    dtype = bundle["dtype"]
    model = _make_vae_model(bundle)

    # Single image encoding
    image = torch.rand((3, 256, 256), device=device, dtype=dtype)

    with torch.inference_mode():
        torch.manual_seed(100)
        latents_1 = model.encode_images([image])
        torch.manual_seed(100)
        latents_2 = model.encode_images([image])

    assert torch.equal(latents_1, latents_2)
