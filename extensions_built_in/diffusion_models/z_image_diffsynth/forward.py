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
