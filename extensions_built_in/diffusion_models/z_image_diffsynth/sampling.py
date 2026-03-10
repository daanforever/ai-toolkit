# Sampling/generation pipeline for Z-Image DiffSynth (wrapper with .images for compatibility).
# When sampling_name_or_path points to a diffusers-format checkpoint, we use ZImagePipeline
# (same as z_image) so the same checkpoint gives the same quality.

import os
import sys
from typing import List, Optional
import torch
import numpy as np
from PIL import Image

from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.debug import memory_debug, is_debug_enabled

from . import forward as fwd_mod

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


def _get_diffsynth_scheduler(num_inference_steps: int, denoising_strength: float = 1.0):
    """Z-Image timestep schedule (DiffSynth set_timesteps_z_image style)."""
    sigma_min = 0.0
    sigma_max = 1.0
    shift = 3.0
    num_train_timesteps = 1000
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps


def _step_scheduler(sigmas, timesteps, model_output, timestep, sample, device):
    """Euler step: prev_sample = sample + model_output * (sigma_next - sigma)."""
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.cpu()
    timestep_id = torch.argmin((timesteps - timestep).abs())
    sigma = sigmas[timestep_id].to(device=device, dtype=sample.dtype)
    if timestep_id + 1 >= len(timesteps):
        sigma_next = torch.tensor(0.0, device=sigmas.device, dtype=sigmas.dtype)
    else:
        sigma_next = sigmas[timestep_id + 1].to(device=device, dtype=sample.dtype)
    prev_sample = sample + model_output * (sigma_next - sigma)
    return prev_sample


class ZImageDiffSynthPipelineWrapper:
    """Wrapper that mimics pipeline(prompt_embeds=..., ...).images for toolkit compatibility."""

    def __init__(self, dit, vae, tokenizer, text_encoder, device, dtype):
        self.dit = dit
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[List[torch.Tensor]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        device = self.device
        dtype = self.dtype
        dit = self.dit
        if next(dit.parameters()).device != device:
            dit = dit.to(device)

        if prompt_embeds is None:
            prompt_embeds = []
        if negative_prompt_embeds is None:
            negative_prompt_embeds = []
        if isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = [prompt_embeds[i] for i in range(prompt_embeds.shape[0])]
        if isinstance(negative_prompt_embeds, torch.Tensor):
            negative_prompt_embeds = [negative_prompt_embeds[i] for i in range(negative_prompt_embeds.shape[0])]

        batch = 1
        if prompt_embeds:
            batch = len(prompt_embeds)
        ch = 16
        h, w = height // 8, width // 8
        # Ensure that the random latents and the RNG generator share the same
        # device. Torch requires that a CUDA generator is only used when the
        # tensor is also allocated on CUDA; otherwise it raises
        # "Expected a 'cuda' device type for generator but found 'cpu'".
        if latents is None:
            rand_device = device
            gen_device = None
            if generator is not None and hasattr(generator, "device"):
                try:
                    gen_device = torch.device(generator.device)
                except Exception:
                    gen_device = None
            if gen_device is not None and gen_device.type != device.type:
                rand_device = gen_device
            latents = torch.randn(
                (batch, ch, h, w),
                device=rand_device,
                dtype=dtype,
                generator=generator,
            )
            if rand_device != device:
                latents = latents.to(device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        sigmas, timesteps = _get_diffsynth_scheduler(num_inference_steps)
        timesteps = timesteps.to(device)

        # When debug logging is enabled, wrap the full sampling loop in a
        # memory_debug context so that peak VRAM for z_image_diffsynth
        # generation can be compared directly against the baseline z_image
        # pipeline.
        def _run_sampling_loop():
            nonlocal latents
            for progress_id in range(len(timesteps)):
                t = timesteps[progress_id].unsqueeze(0).expand(latents.shape[0])
                if guidance_scale <= 1.0 or not prompt_embeds or not negative_prompt_embeds:
                    cond_emb = prompt_embeds[0] if prompt_embeds else None
                    noise_pred = fwd_mod.run_forward(dit, latents, t, cond_emb)
                else:
                    cond_emb = prompt_embeds[0]
                    uncond_emb = negative_prompt_embeds[0]
                    pred_cond = fwd_mod.run_forward(dit, latents, t, cond_emb)
                    pred_uncond = fwd_mod.run_forward(dit, latents, t, uncond_emb)
                    noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                latents = _step_scheduler(sigmas, timesteps, noise_pred, t[0], latents, device)

        if is_debug_enabled():
            with memory_debug(
                lambda msg: print(msg),
                "zimage_diffsynth sampling loop",
            ):
                _run_sampling_loop()
        else:
            _run_sampling_loop()

        vae = self.vae
        decoder = getattr(vae, "vae_decoder", vae)
        if hasattr(decoder, "parameters") and next(decoder.parameters(), None) is not None:
            if next(decoder.parameters()).device != device:
                decoder.to(device)
        if hasattr(vae, "decode"):
            out = vae.decode(latents)
            image = out.sample if hasattr(out, "sample") else out
        else:
            image = vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().numpy()
        image = (image.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
        images = [Image.fromarray(img) for img in image]
        return _ImagesOutput(images)


class _ImagesOutput:
    def __init__(self, images: List[Image.Image]):
        self.images = images


def get_generation_pipeline(sd_model):
    """Build pipeline for sd_model (ZImageDiffSynthModel). Uses sampling transformer if set.
    When sampling is a diffusers-format checkpoint, returns ZImagePipeline (same as z_image)
    so quality matches. Otherwise returns ZImageDiffSynthPipelineWrapper (DiffSynth DiT)."""
    from toolkit.accelerator import unwrap_model

    # Same sampling path as z_image: diffusers ZImageTransformer2DModel + ZImagePipeline
    if getattr(sd_model, "_sampling_is_diffusers", False) and getattr(
        sd_model, "_sampling_transformer", None
    ) is not None:
        from diffusers import ZImagePipeline
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        vae = getattr(sd_model.vae, "vae_decoder", sd_model.vae)
        te = sd_model.text_encoder[0] if isinstance(sd_model.text_encoder, list) else sd_model.text_encoder
        tok = sd_model.tokenizer[0] if isinstance(sd_model.tokenizer, list) else sd_model.tokenizer
        if te is None:
            from toolkit.unloader import FakeTextEncoder
            te = FakeTextEncoder(device=sd_model.device_torch, dtype=sd_model.torch_dtype)
        # Sampling transformer is wrapped in _DiTUnetWrapper for LoRA name match; ZImagePipeline needs raw transformer
        tr = sd_model._sampling_transformer
        tr = getattr(tr, "_inner_dit", getattr(tr, "dit", tr))
        return ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(te),
            tokenizer=tok,
            vae=unwrap_model(vae),
            transformer=unwrap_model(tr),
        )

    # DiffSynth path: ZImageDiT + model_fn_z_image_turbo
    sampling_dit = getattr(sd_model, "_sampling_transformer", None)
    raw_dit = getattr(sd_model, "_raw_dit", None)
    dit = sampling_dit if sampling_dit is not None else raw_dit
    if dit is None:
        dit = sd_model.model
    if isinstance(dit, torch.nn.Module) and "dit" in getattr(dit, "_modules", {}):
        dit = dit._modules["dit"]
    vae = sd_model.vae
    vae_decoder = vae if hasattr(vae, "decode") else (vae.vae_decoder if hasattr(vae, "vae_decoder") else vae)
    tokenizer = sd_model.tokenizer[0] if isinstance(sd_model.tokenizer, list) else sd_model.tokenizer
    if isinstance(sd_model.text_encoder, list):
        text_encoder = sd_model.text_encoder[0] if len(sd_model.text_encoder) > 0 else None
    else:
        text_encoder = sd_model.text_encoder
    if text_encoder is None:
        from toolkit.unloader import FakeTextEncoder
        text_encoder = FakeTextEncoder(device=sd_model.device_torch, dtype=sd_model.torch_dtype)
    return ZImageDiffSynthPipelineWrapper(
        dit=unwrap_model(dit),
        vae=unwrap_model(vae_decoder),
        tokenizer=tokenizer,
        text_encoder=unwrap_model(text_encoder),
        device=sd_model.device_torch,
        dtype=sd_model.torch_dtype,
    )
