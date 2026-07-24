"""Instance-local spatial attention adapter for Z-Image DiT (DiffSynth / Diffusers).

Does not modify DiffSynth-Studio submodule or site-packages. Temporarily binds
patched `_prepare_sequence` / `_build_unified_sequence` on a single DiT instance
for the duration of one forward call.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Sequence
import types

import torch
import torch.nn.functional as F


def pixel_valid_to_patch_valid(
    image_valid_mask: torch.Tensor,
    latent_h: int,
    latent_w: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Convert pixel (or any-res) validity mask to patch-grid validity.

    Args:
        image_valid_mask: (B, 1, H, W) or (B, H, W); True/1 = content.
        latent_h, latent_w: actual latent spatial size matching DiT input.
        patch_size: DiT patch size (Z-Image = 2).

    Returns:
        Bool tensor (B, Hp, Wp). A patch is valid if it contains any content pixel
        (only fully padded patches are masked out).
    """
    if image_valid_mask.dim() == 3:
        image_valid_mask = image_valid_mask.unsqueeze(1)
    if image_valid_mask.dim() != 4:
        raise ValueError(
            f"image_valid_mask expected BCHW or BHW, got shape {tuple(image_valid_mask.shape)}"
        )
    v = image_valid_mask.float()
    if v.shape[-2] != latent_h or v.shape[-1] != latent_w:
        v = F.interpolate(v, size=(latent_h, latent_w), mode="nearest")
    if latent_h % patch_size != 0 or latent_w % patch_size != 0:
        raise ValueError(
            f"latent size ({latent_h}, {latent_w}) must be divisible by patch_size={patch_size}"
        )
    # Max-pool: patch valid if any sub-pixel is content
    pooled = F.max_pool2d(v, kernel_size=patch_size, stride=patch_size)
    return pooled.squeeze(1) > 0.5


def _flatten_patch_grid(valid_hp_wp: torch.Tensor) -> torch.Tensor:
    """(Hp, Wp) or (B, Hp, Wp) → flattened (Hp*Wp,) / (B, Hp*Wp) in (h, w) order."""
    return valid_hp_wp.reshape(*valid_hp_wp.shape[:-2], -1)


def _assert_dit_compatible(dit) -> None:
    missing = [
        name
        for name in ("_prepare_sequence", "_build_unified_sequence", "x_pad_token")
        if not hasattr(dit, name)
    ]
    if missing:
        raise RuntimeError(
            "Z-Image spatial attention adapter requires DiT methods/attrs "
            f"{missing}; incompatible Diffusers/DiffSynth build. "
            "Update diffusers or use loader=diffsynth."
        )


@contextmanager
def spatial_attention_context(
    dit,
    valid_patches: Optional[Sequence[torch.Tensor]],
):
    """
    Temporarily patch dit instance so fully padded patches are:
      - replaced with x_pad_token (OR'd into x_pad_mask)
      - marked False in noise-refiner and unified attn masks

    valid_patches: list length B of bool tensors (Hp, Wp) or (Hp*Wp,), True = valid.
                   None / empty → no-op (yields without patching).
    """
    if not valid_patches:
        yield
        return

    _assert_dit_compatible(dit)

    # Precompute flattened invalid masks (True = invalid / pad)
    spatial_invalid: List[torch.Tensor] = []
    for vp in valid_patches:
        if not isinstance(vp, torch.Tensor):
            raise TypeError(f"valid_patches items must be Tensor, got {type(vp)}")
        flat = _flatten_patch_grid(vp.bool()) if vp.dim() >= 2 else vp.bool().flatten()
        spatial_invalid.append(~flat)

    orig_prepare = dit._prepare_sequence
    orig_build = dit._build_unified_sequence
    state = {
        "x_item_masks": None,  # List[Tensor] bool, length = x_seqlens[i]
        "cap_item_masks": None,
    }

    def prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask=None,
        device: torch.device = None,
    ):
        is_x = pad_token is self.x_pad_token
        pads = list(inner_pad_mask)

        if is_x:
            if len(spatial_invalid) != len(pads):
                raise ValueError(
                    f"spatial_invalid length {len(spatial_invalid)} != batch {len(pads)}"
                )
            new_pads = []
            for i, pad in enumerate(pads):
                inv = spatial_invalid[i].to(device=pad.device, dtype=torch.bool).flatten()
                ori = inv.numel()
                if pad.numel() < ori:
                    raise ValueError(
                        f"x_pad_mask len {pad.numel()} < spatial patches {ori}"
                    )
                combined = pad.clone()
                combined[:ori] = combined[:ori] | inv
                new_pads.append(combined)
            pads = new_pads

        feats_out, freqs, attn_mask, seqlens, noise = orig_prepare(
            feats, pos_ids, pads, pad_token, noise_mask, device
        )

        # Override attn_mask: valid keys = ~combined_pad (never use equal-len → None)
        bsz = len(seqlens)
        max_seqlen = max(seqlens) if seqlens else 0
        device = device or feats_out.device
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        item_masks = []
        for i, seq_len in enumerate(seqlens):
            valid_i = ~pads[i][:seq_len]
            attn_mask[i, :seq_len] = valid_i
            item_masks.append(valid_i.clone())

        if is_x:
            state["x_item_masks"] = item_masks
        else:
            state["cap_item_masks"] = item_masks

        return feats_out, freqs, attn_mask, seqlens, noise

    def build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: List[int],
        x_noise_mask,
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: List[int],
        cap_noise_mask,
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[torch.Tensor],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask,
        omni_mode: bool,
        device: torch.device,
    ):
        unified, unified_freqs, attn_mask, noise_mask_tensor = orig_build(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        x_masks = state.get("x_item_masks")
        cap_masks = state.get("cap_item_masks")
        if x_masks is None:
            return unified, unified_freqs, attn_mask, noise_mask_tensor

        bsz = len(x_seqlens)
        if omni_mode:
            # Omni: [cap, x, (siglip)] — rebuild with our x/cap masks
            parts = []
            for i in range(bsz):
                pieces = []
                if cap_masks is not None:
                    pieces.append(cap_masks[i])
                else:
                    pieces.append(
                        torch.ones(cap_seqlens[i], dtype=torch.bool, device=device)
                    )
                pieces.append(x_masks[i])
                if siglip is not None and siglip_seqlens is not None:
                    pieces.append(
                        torch.ones(siglip_seqlens[i], dtype=torch.bool, device=device)
                    )
                parts.append(torch.cat(pieces))
            unified_seqlens = [p.numel() for p in parts]
        else:
            # Basic: [x, cap]
            parts = []
            for i in range(bsz):
                if cap_masks is not None:
                    cap_m = cap_masks[i]
                else:
                    cap_m = torch.ones(cap_seqlens[i], dtype=torch.bool, device=device)
                parts.append(torch.cat([x_masks[i], cap_m]))
            unified_seqlens = [p.numel() for p in parts]

        max_seqlen = max(unified_seqlens)
        new_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, m in enumerate(parts):
            new_mask[i, : m.numel()] = m

        return unified, unified_freqs, new_mask, noise_mask_tensor

    dit._prepare_sequence = types.MethodType(prepare_sequence, dit)
    dit._build_unified_sequence = types.MethodType(build_unified_sequence, dit)
    try:
        yield
    finally:
        dit._prepare_sequence = orig_prepare
        dit._build_unified_sequence = orig_build
        state["x_item_masks"] = None
        state["cap_item_masks"] = None


def run_diffsynth_forward_with_spatial_mask(
    dit,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds,
    valid_patches: Sequence[torch.Tensor],
    *,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
) -> torch.Tensor:
    """
    DiffSynth DiT forward for B>=1 with spatial mask, matching B>1 turbo conventions:
      timestep = (1000 - t) / 1000, output negated, latents as list of C×1×H×W.
    """
    from einops import rearrange

    B = latents.shape[0]
    t = (1000 - timestep) / 1000.0
    all_image = [rearrange(latents[b : b + 1], "1 C H W -> C 1 H W") for b in range(B)]
    if isinstance(prompt_embeds, torch.Tensor) and prompt_embeds.dim() == 3:
        all_cap = [prompt_embeds[b] for b in range(B)]
    elif isinstance(prompt_embeds, list):
        all_cap = list(prompt_embeds)
    else:
        raise TypeError(f"Unsupported prompt_embeds type: {type(prompt_embeds)}")

    with spatial_attention_context(dit, valid_patches):
        out_list = dit.forward(
            all_image,
            t,
            all_cap,
            patch_size=2,
            f_patch_size=1,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    x = torch.stack([out_list[b].squeeze(1) for b in range(B)], dim=0)
    return -x
