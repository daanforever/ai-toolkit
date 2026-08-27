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
from .scheduler_config import build_scheduler_config
from .turbo_schedule import get_turbo_sigmas_and_timesteps


def _get_diffsynth_scheduler(
    num_inference_steps: int,
    denoising_strength: float = 1.0,
    use_dynamic_shifting: bool = False,
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
):
    """Z-Image timestep schedule (DiffSynth static or Flux-style dynamic shift)."""
    return get_turbo_sigmas_and_timesteps(
        num_inference_steps,
        denoising_strength=denoising_strength,
        use_dynamic_shifting=use_dynamic_shifting,
        latent_h=latent_h,
        latent_w=latent_w,
    )


def _step_scheduler(sigmas, timesteps, model_output, timestep, sample, device):
    """Euler step: prev_sample = sample + model_output * (sigma_next - sigma)."""
    timestep_id = torch.argmin((timesteps - timestep).abs())
    sigma = sigmas[timestep_id].to(device=device, dtype=sample.dtype)
    if timestep_id + 1 >= len(timesteps):
        sigma_next = torch.tensor(0.0, device=device, dtype=sample.dtype)
    else:
        sigma_next = sigmas[timestep_id + 1].to(device=device, dtype=sample.dtype)
    prev_sample = sample + model_output * (sigma_next - sigma)
    return prev_sample


class ZImageDiffSynthPipelineWrapper:
    """Wrapper that mimics pipeline(prompt_embeds=..., ...).images for toolkit compatibility."""

    def __init__(self, dit, vae, tokenizer, text_encoder, device, dtype, use_dynamic_shifting=False):
        self.dit = dit
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype
        self.use_dynamic_shifting = use_dynamic_shifting

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

        sigmas, timesteps = _get_diffsynth_scheduler(
            num_inference_steps,
            use_dynamic_shifting=self.use_dynamic_shifting,
            latent_h=h,
            latent_w=w,
        )
        sigmas = sigmas.to(device)
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
                    noise_pred = fwd_mod.run_forward(
                        dit, latents, t, cond_emb, model_dtype=self.dtype
                    )
                else:
                    cond_emb = prompt_embeds[0]
                    uncond_emb = negative_prompt_embeds[0]
                    pred_cond = fwd_mod.run_forward(
                        dit, latents, t, cond_emb, model_dtype=self.dtype
                    )
                    pred_uncond = fwd_mod.run_forward(
                        dit, latents, t, uncond_emb, model_dtype=self.dtype
                    )
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
        # Same as diffusers ZImagePipeline: scale latents before decode (missing this caused noisy/grainy output)
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
        shift_factor = getattr(getattr(vae, "config", None), "shift_factor", 0.0)
        decode_dtype = getattr(vae, "dtype", latents.dtype)
        latents = latents.to(decode_dtype)
        latents = (latents / scaling_factor) + shift_factor
        if hasattr(vae, "decode"):
            out = vae.decode(latents)
            image = out.sample if hasattr(out, "sample") else out
        else:
            image = vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float().cpu().numpy()
        image = (image.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
        images = [Image.fromarray(img) for img in image]
        return _ImagesOutput(images)


class _ImagesOutput:
    def __init__(self, images: List[Image.Image]):
        self.images = images


def _resolve_use_dynamic_shifting_from_sd_model(sd_model) -> bool:
    try:
        mk = getattr(getattr(sd_model, "model_config", None), "model_kwargs", None) or {}
        return bool(mk.get("use_dynamic_shifting", False))
    except Exception:
        return False


def get_generation_pipeline(sd_model):
    """Build pipeline for sd_model (ZImageDiffSynthModel). Uses sampling transformer if set.
    When loader=diffusers (or auto loaded as Diffusers), use ZImagePipeline with the
    Diffusers transformer. Otherwise use DiffSynth DiT + model_fn_z_image_turbo wrapper.

    Never ``ZImagePipeline.from_pretrained`` here: that reloads a second Qwen TE
    (~8GB) plus a bf16 DiT from disk on every sample().
    """
    from toolkit.accelerator import unwrap_model

    sampling_is_diffusers = getattr(sd_model, "_sampling_is_diffusers", False)
    main_is_diffusers = getattr(sd_model, "_main_is_diffusers", False)
    sampling_transformer = getattr(sd_model, "_sampling_transformer", None)

    use_diffusers_pipeline = (
        (sampling_is_diffusers and sampling_transformer is not None)
        or (main_is_diffusers and sampling_transformer is None)
        or (main_is_diffusers and sampling_is_diffusers)
    )

    # Diffusers path: ZImagePipeline with already-loaded transformer / VAE / TE.
    if use_diffusers_pipeline:
        from diffusers import ZImagePipeline

        if sampling_transformer is not None and (
            sampling_is_diffusers or main_is_diffusers
        ):
            tr_source = sampling_transformer
        else:
            tr_source = getattr(sd_model, "model", None)

        use_dynamic_shifting = _resolve_use_dynamic_shifting_from_sd_model(sd_model)
        scheduler = CustomFlowMatchEulerDiscreteScheduler(
            **build_scheduler_config(use_dynamic_shifting=use_dynamic_shifting)
        )
        vae = getattr(sd_model.vae, "vae_decoder", sd_model.vae)
        te = (
            sd_model.text_encoder[0]
            if isinstance(sd_model.text_encoder, list)
            else sd_model.text_encoder
        )
        tok = (
            sd_model.tokenizer[0]
            if isinstance(sd_model.tokenizer, list)
            else sd_model.tokenizer
        )
        if te is None:
            from toolkit.unloader import FakeTextEncoder

            te = FakeTextEncoder(
                device=sd_model.device_torch, dtype=sd_model.torch_dtype
            )
        tr = getattr(tr_source, "_inner_dit", getattr(tr_source, "dit", tr_source))
        return ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(te),
            tokenizer=tok,
            vae=unwrap_model(vae),
            transformer=unwrap_model(tr),
        )

    # DiffSynth path: ZImageDiT + model_fn_z_image_turbo
    sampling_dit = sampling_transformer
    raw_dit = getattr(sd_model, "_raw_dit", None)
    dit = sampling_dit if sampling_dit is not None else raw_dit
    if dit is None:
        dit = sd_model.model
    if isinstance(dit, torch.nn.Module) and "dit" in getattr(dit, "_modules", {}):
        dit = dit._modules["dit"]
    dit = getattr(dit, "_inner_dit", dit)
    vae = sd_model.vae
    vae_decoder = (
        vae
        if hasattr(vae, "decode")
        else (vae.vae_decoder if hasattr(vae, "vae_decoder") else vae)
    )
    tokenizer = (
        sd_model.tokenizer[0]
        if isinstance(sd_model.tokenizer, list)
        else sd_model.tokenizer
    )
    if isinstance(sd_model.text_encoder, list):
        text_encoder = (
            sd_model.text_encoder[0] if len(sd_model.text_encoder) > 0 else None
        )
    else:
        text_encoder = sd_model.text_encoder
    if text_encoder is None:
        from toolkit.unloader import FakeTextEncoder

        text_encoder = FakeTextEncoder(
            device=sd_model.device_torch, dtype=sd_model.torch_dtype
        )
    return ZImageDiffSynthPipelineWrapper(
        dit=unwrap_model(dit),
        vae=unwrap_model(vae_decoder),
        tokenizer=tokenizer,
        text_encoder=unwrap_model(text_encoder),
        device=sd_model.device_torch,
        dtype=sd_model.torch_dtype,
        use_dynamic_shifting=_resolve_use_dynamic_shifting_from_sd_model(sd_model),
    )
