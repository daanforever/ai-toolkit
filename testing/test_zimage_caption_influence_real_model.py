"""
Real-model checks that Z-Image DiffSynth forward reacts to caption changes.

Unlike text-encoder-only tests, this file loads the real DiT (quantized) and checks
`get_noise_prediction` with fixed latents/timestep.

Run (repo root, venv):
  venv\\Scripts\\python.exe -m pytest testing/test_zimage_caption_influence_real_model.py -v -s

Env:
  ZIMAGE_DIFFSYNTH_MODEL_PATH (optional, falls back to DEFAULT_ZIMAGE_MODEL_PATH from test_smoke)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from toolkit.config_modules import ModelConfig  # noqa: E402
from toolkit.util.get_model import get_model_class  # noqa: E402


def _resolve_model_path() -> str:
    return os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH


@pytest.fixture(scope="module")
def loaded_zimage_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this real-model test")

    model_path = _resolve_model_path()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path is missing. Set ZIMAGE_DIFFSYNTH_MODEL_PATH or update DEFAULT_ZIMAGE_MODEL_PATH."
        )

    model_config = ModelConfig(
        name_or_path=model_path,
        arch="zimage_diffsynth",
        quantize=True,
        quantize_te=True,
        model_kwargs={"use_diffsynth_training_loop": False},
    )
    model_class = get_model_class(model_config)
    device = torch.device("cuda")
    sd = model_class(device, model_config, dtype="bf16")
    sd.load_model()

    # Keep forward deterministic for repeated calls with the same inputs.
    sd.model.eval()
    sd._raw_dit.eval()

    try:
        yield sd
    finally:
        del sd
        torch.cuda.empty_cache()


def _predict_for_prompt(sd, prompt: str, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    # get_noise_prediction swallows .to() failures for safety; enforce the move here
    # so this test fails loudly if quantized DiT cannot be placed on CUDA.
    sd.model.to(sd.device_torch)
    embeds = sd.get_prompt_embeds(prompt)
    with torch.inference_mode():
        pred = sd.get_noise_prediction(latent, timestep, embeds)
    return pred.detach().float()


def test_toolkit_prompt_encoding_forward_changes_with_caption(loaded_zimage_model):
    sd = loaded_zimage_model
    sd.model_config.model_kwargs["use_diffsynth_prompt_encoding"] = False

    torch.manual_seed(123)
    latent = torch.randn(1, 16, 64, 64, device=sd.device_torch, dtype=sd.torch_dtype)
    timestep = torch.tensor([500.0], device=sd.device_torch, dtype=torch.float32)

    pred_a = _predict_for_prompt(sd, "a red cube on white table", latent, timestep)
    pred_a_repeat = _predict_for_prompt(sd, "a red cube on white table", latent, timestep)
    pred_b = _predict_for_prompt(sd, "a green sphere on white table", latent, timestep)

    same_prompt_delta = (pred_a - pred_a_repeat).abs().max().item()
    different_prompt_delta = (pred_a - pred_b).abs().max().item()

    assert same_prompt_delta == pytest.approx(0.0, abs=1e-7)
    assert different_prompt_delta > 1e-6


def test_diffsynth_literal_prompt_encoding_forward_changes_with_caption(loaded_zimage_model):
    sd = loaded_zimage_model
    sd.model_config.model_kwargs["use_diffsynth_prompt_encoding"] = True

    torch.manual_seed(321)
    latent = torch.randn(1, 16, 64, 64, device=sd.device_torch, dtype=sd.torch_dtype)
    timestep = torch.tensor([500.0], device=sd.device_torch, dtype=torch.float32)

    pred_a = _predict_for_prompt(sd, "close-up portrait of an orange fox", latent, timestep)
    pred_a_repeat = _predict_for_prompt(sd, "close-up portrait of an orange fox", latent, timestep)
    pred_b = _predict_for_prompt(sd, "close-up portrait of a blue robot", latent, timestep)

    same_prompt_delta = (pred_a - pred_a_repeat).abs().max().item()
    different_prompt_delta = (pred_a - pred_b).abs().max().item()

    assert same_prompt_delta == pytest.approx(0.0, abs=1e-7)
    assert different_prompt_delta > 1e-6
