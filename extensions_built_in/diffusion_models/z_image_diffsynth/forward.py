# Forward pass via DiffSynth model_fn_z_image_turbo (or direct DiT when spatial mask set).

from typing import Optional, Sequence
import torch


def run_forward(
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds,
    model_dtype: torch.dtype,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    train_dtype: Optional[torch.dtype] = None,
    image_valid_patches: Optional[Sequence[torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Run one forward pass: latents BCHW, timestep 0..1000, prompt_embeds as expected by DiffSynth.
    Returns prediction tensor in same convention as DiffSynth-Studio (flow matching).

    When image_valid_patches is provided, routes B=1 and B>1 through dit.forward under
    the spatial-attention adapter (bypassing turbo B=1 attn_mask=None hardcode).
    Unmasked / inference calls keep model_fn_z_image_turbo.

    Activations are aligned to train_dtype (fallback: latents.dtype). If that differs from
    model_dtype (YAML model.dtype), a compute gate casts inputs to model_dtype and casts
    the output back. CUDA autocast uses model_dtype for fp16/bf16.
    """
    import sys
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(this_dir, "DiffSynth-Studio")
    if ds_dir not in sys.path:
        sys.path.insert(0, ds_dir)
    from diffsynth.pipelines.z_image import model_fn_z_image_turbo

    from .spatial_attention import run_diffsynth_forward_with_spatial_mask

    # model_fn_z_image_turbo expects: latents (BCHW or list), timestep 0..1000, prompt_embeds (list or tensor)
    # It unwraps list to single; timestep is 1000 - timestep inside. We pass timestep in 0..1000.
    #
    # Before calling into the DiffSynth code, enforce that the incoming latents
    # have the channel count the DiT was configured for. If, for example, 3‑channel
    # RGB tensors reach this point instead of 16‑channel VAE latents, patchify
    # will produce tokens of size 12 and hit dit.all_x_embedder[\"2-1\"] with a
    # 64x3840 weight matrix, causing a confusing Quanto matmul error. Failing
    # here with a clear message makes the problem debuggable.
    if isinstance(latents, torch.Tensor) and latents.dim() == 4:
        expected_c = getattr(dit, "in_channels", None)
        if expected_c is not None:
            _, c, _, _ = latents.shape
            if c != expected_c:
                raise RuntimeError(
                    (
                        f"Z-Image DiffSynth DiT expected latents with {expected_c} channels "
                        f"(B x {expected_c} x H x W), but got B x {c} x H x W. "
                        "This typically indicates that cached latents were generated with a different "
                        "model/latent_space_version or that non-latent RGB tensors were cached. "
                        "Regenerate the dataset latents for this model or disable latent caching for this run."
                    )
                )

    # Align embeddings to train.dtype (explicit) or latents.dtype
    act_dtype = train_dtype if train_dtype is not None else latents.dtype
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(act_dtype)
    elif isinstance(prompt_embeds, list):
        prompt_embeds = [p.to(act_dtype) if isinstance(p, torch.Tensor) else p for p in prompt_embeds]
    if isinstance(latents, torch.Tensor) and latents.dtype != act_dtype:
        latents = latents.to(act_dtype)

    # Compute gate: model.dtype may differ from train.dtype activations
    out_dtype = act_dtype
    if latents.dtype != model_dtype:
        latents = latents.to(model_dtype)
        if isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = prompt_embeds.to(model_dtype)
        elif isinstance(prompt_embeds, list):
            prompt_embeds = [
                p.to(model_dtype) if isinstance(p, torch.Tensor) else p for p in prompt_embeds
            ]

    use_autocast = (
        latents.device.type == "cuda"
        and model_dtype in (torch.float16, torch.bfloat16)
    )
    with torch.autocast(
        device_type="cuda",
        dtype=model_dtype if use_autocast else torch.float32,
        enabled=use_autocast,
    ):
        if image_valid_patches:
            out = run_diffsynth_forward_with_spatial_mask(
                dit,
                latents,
                timestep,
                prompt_embeds,
                image_valid_patches,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        else:
            out = model_fn_z_image_turbo(
                dit,
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                **kwargs,
            )
    if isinstance(out, torch.Tensor) and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out
