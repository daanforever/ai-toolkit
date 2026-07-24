"""Unit tests for spatial attention adapter and batch plumbing (no real weights)."""

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from extensions_built_in.diffusion_models.z_image_diffsynth.spatial_attention import (
    pixel_valid_to_patch_valid,
    spatial_attention_context,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.diffsynth_training import (
    aggregate_flow_matching_mse_diffsynth,
)


def test_pixel_valid_to_patch_valid_any_content():
    # Latent 4x4, patch 2 → 2x2 patches. Bottom-right patch fully pad.
    valid = torch.zeros(1, 1, 4, 4)
    valid[0, 0, :3, :3] = 1.0
    patches = pixel_valid_to_patch_valid(valid, 4, 4, patch_size=2)
    assert patches.shape == (1, 2, 2)
    assert bool(patches[0, 0, 0])
    assert bool(patches[0, 0, 1])
    assert bool(patches[0, 1, 0])
    # Bottom-right 2x2 has only (2,2) as content → still valid (any content)
    assert bool(patches[0, 1, 1])

    valid2 = torch.zeros(1, 1, 4, 4)
    valid2[0, 0, :2, :2] = 1.0
    patches2 = pixel_valid_to_patch_valid(valid2, 4, 4, patch_size=2)
    assert bool(patches2[0, 0, 0])
    assert not bool(patches2[0, 0, 1])
    assert not bool(patches2[0, 1, 0])
    assert not bool(patches2[0, 1, 1])


def test_pixel_valid_nearest_resize_to_latent():
    # Pixel mask 8x8 → latent 4x4
    valid = torch.zeros(1, 1, 8, 8)
    valid[0, 0, :, :4] = 1.0  # left half
    patches = pixel_valid_to_patch_valid(valid, 4, 4, patch_size=2)
    assert bool(patches[0, 0, 0])
    assert not bool(patches[0, 0, 1])


class _FakeDiT(nn.Module):
    """Minimal stand-in exposing _prepare_sequence / _build_unified_sequence."""

    def __init__(self):
        super().__init__()
        self.x_pad_token = nn.Parameter(torch.ones(1, 4) * 7.0)
        self.cap_pad_token = nn.Parameter(torch.ones(1, 4) * 3.0)
        self._calls = {"prepare": 0, "build": 0}

    def _prepare_sequence(
        self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None
    ):
        self._calls["prepare"] += 1
        from torch.nn.utils.rnn import pad_sequence

        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token.to(
            dtype=feats_cat.dtype, device=feats_cat.device
        )
        feats = list(feats_cat.split(item_seqlens, dim=0))
        freqs = [torch.zeros(len(p), 2) for p in pos_ids]
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs = pad_sequence(freqs, batch_first=True, padding_value=0.0)
        # Equal-length shortcut like some Diffusers builds
        if all(s == max_seqlen for s in item_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(item_seqlens):
                attn_mask[i, :seq_len] = 1
        return feats, freqs, attn_mask, item_seqlens, None

    def _build_unified_sequence(
        self,
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
    ):
        self._calls["build"] += 1
        from torch.nn.utils.rnn import pad_sequence

        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        for i in range(bsz):
            unified.append(torch.cat([x[i][: x_seqlens[i]], cap[i][: cap_seqlens[i]]]))
            unified_freqs.append(
                torch.cat([x_freqs[i][: x_seqlens[i]], cap_freqs[i][: cap_seqlens[i]]])
            )
        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)
        if all(s == max_seqlen for s in unified_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_seqlens):
                attn_mask[i, :seq_len] = 1
        return unified, unified_freqs, attn_mask, None


def test_adapter_masks_invalid_and_replaces_embeds():
    dit = _FakeDiT()
    # 4 content patches + 0 SEQ pad (len already multiple of nothing special)
    # Make x_pad_mask: last 0 of 4 are SEQ; mark patch 3 spatially invalid
    feats = [torch.randn(4, 4)]
    pos = [torch.zeros(4, 3)]
    # No SEQ pad; spatial invalid on index 3
    pad_mask = [torch.tensor([False, False, False, False])]
    valid = torch.tensor([[True, True], [True, False]])  # Hp=2,Wp=2

    with spatial_attention_context(dit, [valid]):
        fout, freqs, attn_mask, seqlens, _ = dit._prepare_sequence(
            feats, pos, pad_mask, dit.x_pad_token, device=torch.device("cpu")
        )
    assert attn_mask is not None
    assert attn_mask.shape == (1, 4)
    assert bool(attn_mask[0, 0]) and bool(attn_mask[0, 1]) and bool(attn_mask[0, 2])
    assert not bool(attn_mask[0, 3])
    # Invalid embed replaced with pad token value 7
    assert torch.allclose(fout[0, 3], torch.ones(4) * 7.0)


def test_adapter_unified_mask_order_x_then_cap():
    dit = _FakeDiT()
    valid = torch.tensor([[True, False], [True, True]])
    x_feats = [torch.randn(4, 4)]
    x_pos = [torch.zeros(4, 3)]
    x_pad = [torch.zeros(4, dtype=torch.bool)]
    cap_feats = [torch.randn(2, 4)]
    cap_pos = [torch.zeros(2, 3)]
    cap_pad = [torch.zeros(2, dtype=torch.bool)]

    with spatial_attention_context(dit, [valid]):
        x, x_f, x_m, xs, _ = dit._prepare_sequence(
            x_feats, x_pos, x_pad, dit.x_pad_token, device=torch.device("cpu")
        )
        cap, c_f, c_m, cs, _ = dit._prepare_sequence(
            cap_feats, cap_pos, cap_pad, dit.cap_pad_token, device=torch.device("cpu")
        )
        unified, uf, um, _ = dit._build_unified_sequence(
            x,
            x_f,
            xs,
            None,
            cap,
            c_f,
            cs,
            None,
            None,
            None,
            None,
            None,
            False,
            torch.device("cpu"),
        )
    assert um is not None
    # [x(4), cap(2)] — patch 1 invalid
    assert um.shape == (1, 6)
    assert list(um[0].tolist()) == [True, False, True, True, True, True]


def test_adapter_restores_methods():
    dit = _FakeDiT()
    orig_p = dit._prepare_sequence
    orig_b = dit._build_unified_sequence
    valid = torch.ones(2, 2, dtype=torch.bool)
    with spatial_attention_context(dit, [valid]):
        assert dit._prepare_sequence is not orig_p
    assert dit._prepare_sequence == orig_p
    assert dit._build_unified_sequence == orig_b


def test_get_noise_prediction_batch_param_uses_validity_not_user_mask():
    """Named batch= plumbing: attention uses image_valid_mask_tensor only."""
    from extensions_built_in.diffusion_models.z_image_diffsynth import model as model_mod
    from toolkit.prompt_utils import PromptEmbeds
    import inspect

    sig = inspect.signature(model_mod.ZImageDiffSynthModel.get_noise_prediction)
    assert "batch" in sig.parameters

    # Build a minimal model instance without loading weights
    m = object.__new__(model_mod.ZImageDiffSynthModel)
    m._main_is_diffusers = False
    m._raw_dit = SimpleNamespace(in_channels=16)
    m.model = MagicMock()
    m.device_torch = torch.device("cpu")
    m.torch_dtype = torch.float32
    m.train_torch_dtype = torch.float32
    m.gradient_checkpointing = False

    captured = {}

    def fake_run_forward(dit, latents, timestep, text_embeds, **kwargs):
        captured["image_valid_patches"] = kwargs.get("image_valid_patches")
        return torch.zeros_like(latents)

    import extensions_built_in.diffusion_models.z_image_diffsynth.forward as forward_mod

    orig = forward_mod.run_forward
    forward_mod.run_forward = fake_run_forward
    try:
        # Validity: left half content on 8x8 latent → patch grid 4x4
        valid = torch.zeros(1, 1, 8, 8)
        valid[0, 0, :, :4] = 1.0
        user_mask = torch.ones(1, 1, 8, 8) * 0.5  # should NOT drive attention
        batch = SimpleNamespace(
            image_valid_mask_tensor=valid,
            mask_tensor=user_mask,
        )
        te = PromptEmbeds(torch.randn(1, 4, 8))
        latents = torch.randn(1, 16, 8, 8)
        ts = torch.tensor([500.0])
        out = model_mod.ZImageDiffSynthModel.get_noise_prediction(
            m, latents, ts, te, batch=batch
        )
        assert out.shape == latents.shape
        patches = captured["image_valid_patches"]
        assert patches is not None
        assert patches[0].shape == (4, 4)
        # Right half patches invalid
        assert not bool(patches[0][0, 3])
        assert bool(patches[0][0, 0])
    finally:
        forward_mod.run_forward = orig


def test_all_valid_skips_adapter_kwargs():
    from extensions_built_in.diffusion_models.z_image_diffsynth import model as model_mod
    from toolkit.prompt_utils import PromptEmbeds

    m = object.__new__(model_mod.ZImageDiffSynthModel)
    m._main_is_diffusers = False
    m._raw_dit = SimpleNamespace(in_channels=16)
    m.model = MagicMock()
    m.device_torch = torch.device("cpu")
    m.torch_dtype = torch.float32
    m.train_torch_dtype = torch.float32
    m.gradient_checkpointing = False

    captured = {}

    def fake_run_forward(dit, latents, timestep, text_embeds, **kwargs):
        captured["image_valid_patches"] = kwargs.get("image_valid_patches")
        return torch.zeros_like(latents)

    import extensions_built_in.diffusion_models.z_image_diffsynth.forward as forward_mod

    orig = forward_mod.run_forward
    forward_mod.run_forward = fake_run_forward
    try:
        valid = torch.ones(1, 1, 8, 8)
        batch = SimpleNamespace(image_valid_mask_tensor=valid, mask_tensor=valid)
        te = PromptEmbeds(torch.randn(1, 4, 8))
        out = model_mod.ZImageDiffSynthModel.get_noise_prediction(
            m, torch.randn(1, 16, 8, 8), torch.tensor([100.0]), te, batch=batch
        )
        assert captured["image_valid_patches"] is None
        assert out.shape[1] == 16
    finally:
        forward_mod.run_forward = orig


def test_loss_pad_region_zero_both_aggregators():
    """Pad zeros in mask → zero contribution; mean-norm stays finite."""
    B, C, H, W = 2, 4, 8, 8
    pred = torch.ones(B, C, H, W)
    target = torch.zeros(B, C, H, W)
    # Validity-style mask: left half 1, right half 0
    mm = torch.zeros(B, C, H, W)
    mm[:, :, :, : W // 2] = 1.0
    # Simulate SDTrainer mean normalization
    mm_norm = mm / mm.mean()
    assert torch.isfinite(mm_norm).all()

    # DiffSynth aggregator
    ts = torch.tensor([100.0, 200.0])
    w = torch.ones(B)
    out_ds = aggregate_flow_matching_mse_diffsynth(
        pred,
        target,
        ts,
        w,
        mm_norm,
        pred,
        train_turbo=False,
        log_writer=None,
        step_num=0,
        is_main_process=False,
        log_every=None,
    )
    assert out_ds.shape == (B,)
    assert torch.isfinite(out_ds).all()

    # Toolkit-style: loss * mask then mean
    sq = F.mse_loss(pred, target, reduction="none") * mm_norm
    # Right half must be exactly 0 before norm scaling... after norm, right is still 0
    assert float(sq[:, :, :, W // 2 :].abs().max()) == 0.0
    toolkit_loss = sq.mean(dim=(1, 2, 3))
    assert torch.isfinite(toolkit_loss).all()
