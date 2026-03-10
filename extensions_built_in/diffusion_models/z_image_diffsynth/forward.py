# Forward pass via DiffSynth model_fn_z_image_turbo.

from typing import Optional
import torch

from einops import rearrange


def run_forward(
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Run one forward pass: latents BCHW, timestep 0..1000, prompt_embeds as expected by DiffSynth.
    Returns prediction tensor in same convention as DiffSynth-Studio (flow matching).
    """
    import sys
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(this_dir, "DiffSynth-Studio")
    if ds_dir not in sys.path:
        sys.path.insert(0, ds_dir)
    from diffsynth.pipelines.z_image import model_fn_z_image_turbo

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

    out = model_fn_z_image_turbo(
        dit,
        latents=latents,
        timestep=timestep,
        prompt_embeds=prompt_embeds,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        **kwargs,
    )
    return out
