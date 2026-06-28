"""
Real-model check: LoRA gradients should depend on caption for zimage_diffsynth.

This test targets the user's real setup:
- use_diffsynth_training_loop: false
- quantize: true
- quantize_te: true

Run (repo root, venv):
  venv\\Scripts\\python.exe -m pytest testing/test_zimage_lora_caption_grad_real_model.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from toolkit.config_modules import ModelConfig, NetworkConfig  # noqa: E402
from toolkit.lora_special import LoRASpecialNetwork  # noqa: E402
from toolkit.util.get_model import get_model_class  # noqa: E402


def _resolve_model_path() -> str:
    return os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH


def _build_lora_network(sd) -> LoRASpecialNetwork:
    network_config = NetworkConfig(
        type="lora",
        linear=8,
        linear_alpha=8,
        transformer_only=True,
        network_kwargs={},
    )
    network_kwargs = dict(network_config.network_kwargs)
    if hasattr(sd, "target_lora_modules"):
        network_kwargs["target_lin_modules"] = sd.target_lora_modules

    network = LoRASpecialNetwork(
        text_encoder=sd.text_encoder,
        unet=sd.get_model_to_train(),
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        conv_lora_dim=network_config.conv,
        conv_alpha=network_config.conv_alpha,
        is_sdxl=False,
        is_v2=False,
        is_v3=False,
        is_pixart=False,
        is_auraflow=False,
        is_flux=False,
        is_lumina2=False,
        is_ssd=False,
        is_vega=False,
        dropout=network_config.dropout,
        rank_dropout=network_config.rank_dropout,
        module_dropout=network_config.module_dropout,
        use_text_encoder_1=True,
        use_text_encoder_2=True,
        use_bias=False,
        is_lorm=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=sd.is_transformer,
        base_model=sd,
        **network_kwargs,
    )
    network.force_to(sd.device_torch, dtype=torch.float32)
    sd.network = network
    network._update_torch_multiplier()
    network.apply_to(sd.text_encoder, sd.get_model_to_train(), False, True)
    network.prepare_grad_etc(sd.text_encoder, sd.get_model_to_train())
    network.train()
    return network


@pytest.fixture(scope="module")
def zimage_with_lora():
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
        model_kwargs={
            "use_diffsynth_training_loop": False,
            "use_diffsynth_prompt_encoding": False,
        },
    )
    model_class = get_model_class(model_config)
    sd = model_class(torch.device("cuda"), model_config, dtype="bf16")
    sd.load_model()

    # Match real trainer setup: freeze backbone modules, only LoRA trainable.
    if isinstance(sd.text_encoder, list):
        for te in sd.text_encoder:
            te.requires_grad_(False)
            te.eval()
    else:
        sd.text_encoder.requires_grad_(False)
        sd.text_encoder.eval()
    sd.get_model_to_train().requires_grad_(False)
    sd.get_model_to_train().eval()
    sd.vae.to("cpu", dtype=sd.torch_dtype)
    sd.vae.requires_grad_(False)
    sd.vae.eval()

    network = _build_lora_network(sd)

    with torch.inference_mode():
        cached_embeds = {
            "a red cube on white table": sd.get_prompt_embeds("a red cube on white table")
            .detach()
            .to("cpu", dtype=torch.float32),
            "a green sphere on white table": sd.get_prompt_embeds("a green sphere on white table")
            .detach()
            .to("cpu", dtype=torch.float32),
        }
    sd.text_encoder_to("cpu")
    torch.cuda.empty_cache()

    try:
        yield sd, network, cached_embeds
    finally:
        del network
        del sd
        torch.cuda.empty_cache()


def _collect_lora_grad_vector(
    sd,
    network: LoRASpecialNetwork,
    cached_embeds,
    latent,
    timestep,
    target,
) -> torch.Tensor:
    network.zero_grad(set_to_none=True)
    # get_noise_prediction can ignore failed .to(); enforce move here for quantized DiT.
    sd.model.to(sd.device_torch)
    with network:
        embeds = cached_embeds.clone().to(sd.device_torch, dtype=sd.torch_dtype)
        pred = sd.get_noise_prediction(latent, timestep, embeds)
        loss = F.mse_loss(pred.float(), target.float())
        loss.backward()

    grad_chunks: list[torch.Tensor] = []
    for _, param in network.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_chunks.append(param.grad.detach().float().reshape(-1).cpu())
    if not grad_chunks:
        raise AssertionError("no LoRA gradients were produced")
    return torch.cat(grad_chunks, dim=0)


def test_lora_gradients_change_when_caption_changes(zimage_with_lora):
    sd, network, cached_embeds = zimage_with_lora

    torch.manual_seed(777)
    latent = torch.randn(1, 16, 64, 64, device=sd.device_torch, dtype=sd.torch_dtype)
    target = torch.randn(1, 16, 64, 64, device=sd.device_torch, dtype=sd.torch_dtype)
    timestep = torch.tensor([500.0], device=sd.device_torch, dtype=torch.float32)

    grad_same_1 = _collect_lora_grad_vector(
        sd, network, cached_embeds["a red cube on white table"], latent, timestep, target
    )
    grad_same_2 = _collect_lora_grad_vector(
        sd, network, cached_embeds["a red cube on white table"], latent, timestep, target
    )
    grad_diff = _collect_lora_grad_vector(
        sd, network, cached_embeds["a green sphere on white table"], latent, timestep, target
    )

    same_norm = torch.norm(grad_same_1).item()
    assert same_norm > 0.0

    same_delta_rel = torch.norm(grad_same_1 - grad_same_2).item() / same_norm
    diff_delta_rel = torch.norm(grad_same_1 - grad_diff).item() / same_norm

    # Quantized kernels are not strictly bitwise deterministic between calls.
    # We expect small drift for same prompt and a significantly larger shift for different prompts.
    assert same_delta_rel < 0.05
    assert diff_delta_rel > same_delta_rel * 2.0
