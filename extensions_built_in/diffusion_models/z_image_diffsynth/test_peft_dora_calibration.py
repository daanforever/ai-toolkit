"""DoRA magnitude-calibration test for `PeftNetwork.share_parameters_with`.

Reproduces the bug where a sampling `PeftNetwork` (peft_dora) built on a
different base DiT than the main network produces non-identity step-0 output
because its DoRA magnitude is sourced from the main network's ||W_main||.

After the calibration fix, the sampling network's magnitude Parameter stays
shared by reference with the main network (only main trains), and a per-module
calibration ratio = ||W_sampling|| / ||W_main|| is applied at read time so
PEFT's DoRA forward sees a calibrated magnitude equal to ||W_sampling|| at init
-> identity at step 0, and proportionally rescaled during training. This
mirrors toolkit.models.DoRA.DoRAModule.apply_dora.

Runs on CPU and does not require the full Z-Image model.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d)])

    def forward(self, x):
        # Exercise every targeted linear so all DoRA magnitude modules are
        # forwarded (and their calibration is applied) during a pass.
        return self.to_out[0](self.to_v(x) + self.to_q(x) + self.to_k(x))


class FeedForward(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.w1 = nn.Linear(d, d * 2)
        self.w2 = nn.Linear(d * 2, d)
        self.w3 = nn.Linear(d, d * 2)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) + self.w3(x))


class _BlockStub(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.attention = Attention(d)
        self.feed_forward = FeedForward(d)


class _InnerDiTStub(nn.Module):
    def __init__(self, d: int = 8, n_blocks: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_BlockStub(d) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.layers:
            x = blk.attention(x) + blk.feed_forward(x)
        return x


class _UnetWrapperStub(nn.Module):
    def __init__(self, dit: nn.Module):
        super().__init__()
        self._inner_dit = dit

    def forward(self, *args, **kwargs):
        return self._inner_dit(*args, **kwargs)


class _StubBaseModel:
    arch = "zimage_diffsynth"
    target_lora_modules = ["Attention", "FeedForward"]

    def convert_lora_weights_before_save(self, sd):
        return sd

    def convert_lora_weights_before_load(self, sd):
        return sd


def _build_dit_wrapper(seed: int, d: int = 8):
    torch.manual_seed(seed)
    dit = _InnerDiTStub(d=d, n_blocks=2)
    return _UnetWrapperStub(dit)


def _build_peft_dora_network(wrapper, base):
    from toolkit.peft_network import PeftNetwork

    return PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type="peft_dora",
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )


def _collect_magnitude_modules(net):
    mods = []
    for module in net.peft_model.modules():
        mag = getattr(module, "lora_magnitude_vector", None)
        if mag is None:
            continue
        for adapter_name in mag:
            if mag[adapter_name] is not None:
                mods.append(mag[adapter_name])
    return mods


def test_peft_dora_share_parameters_is_identity_at_step0():
    """A sampling peft_dora network built on a different base DiT than the main
    network must produce identity output on step 0 after share_parameters_with
    (lora_B is still zero, so the DoRA term must vanish for every layer)."""
    main_wrapper = _build_dit_wrapper(seed=1)
    sampling_wrapper = _build_dit_wrapper(seed=2)
    base = _StubBaseModel()

    main_net = _build_peft_dora_network(main_wrapper, base)
    sampling_net = _build_peft_dora_network(sampling_wrapper, base)

    # Precondition: the two base DiTs differ, so ||W_main|| != ||W_sampling||.
    main_mags = _collect_magnitude_modules(main_net)
    sampling_mags = _collect_magnitude_modules(sampling_net)
    assert main_mags and sampling_mags
    assert not torch.allclose(main_mags[0].weight, sampling_mags[0].weight)

    sampling_net.share_parameters_with(main_net)

    x = torch.randn(2, 8, 8, dtype=torch.float32)

    # Base-only reference: is_active=False makes the multiplier wrapper set
    # _disable_adapters=True on each LoraLayer, so PEFT returns the base output.
    sampling_net.is_active = False
    with torch.no_grad():
        base_out = sampling_net.peft_model(x)

    # DoRA forward with calibrated magnitude at multiplier=1.0.
    sampling_net.is_active = True
    with torch.no_grad():
        adapter_out = sampling_net.peft_model(x)

    assert torch.allclose(adapter_out, base_out, atol=1e-4), (
        "peft_dora sampling forward is not identity at step 0 after "
        f"share_parameters_with: max diff = {(adapter_out - base_out).abs().max().item()}"
    )


def test_peft_dora_magnitude_is_shared_by_reference():
    """The sampling network's DoRA magnitude Parameter must be the same object
    as the main network's (shared by reference), so only the main network
    trains and updates propagate live. The calibration is applied at read time,
    not by keeping a separate sampling Parameter."""
    main_wrapper = _build_dit_wrapper(seed=1)
    sampling_wrapper = _build_dit_wrapper(seed=2)
    base = _StubBaseModel()

    main_net = _build_peft_dora_network(main_wrapper, base)
    sampling_net = _build_peft_dora_network(sampling_wrapper, base)
    sampling_net.share_parameters_with(main_net)

    main_mags = _collect_magnitude_modules(main_net)
    sampling_mags = _collect_magnitude_modules(sampling_net)

    for main_m, sampling_m in zip(main_mags, sampling_mags):
        assert sampling_m.weight is main_m.weight, (
            "sampling DoRA magnitude must be shared by reference with the main network"
        )

    # Sharing survives a sampling forward (swap-restore must restore the shared
    # Parameter after each calibrated forward call).
    sampling_net.is_active = True
    with torch.no_grad():
        sampling_net.peft_model(torch.randn(2, 8, 8, dtype=torch.float32))
    for main_m, sampling_m in zip(main_mags, sampling_mags):
        assert sampling_m.weight is main_m.weight, (
            "sampling DoRA magnitude must still be shared after a sampling forward"
        )


def test_peft_dora_calibration_tracks_main_magnitude_update():
    """After a magnitude update on the main network, the sampling forward must
    reflect the calibrated magnitude (main_magnitude * ratio) at read time.

    Uses a single-linear DiT so the DoRA scaling does not compound across
    layers: with lora_B = 0 and mag_norm_scale = s, DoRA output = s * base, so
    after main magnitude *= 1.5 the calibrated mag_norm_scale becomes 1.5 and
    adapter_out = 1.5 * base_out.
    """

    class Attention(nn.Module):  # name matches target_lora_modules
        def __init__(self, d: int = 8):
            super().__init__()
            # bias=False so DoRA output = mag_norm_scale * (W@x) exactly
            # (PEFT subtracts bias before scaling base_result in DoRA forward,
            # which would otherwise break a clean mag_norm_scale * base check).
            self.to_q = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.to_q(x)

    class _SingleDiT(nn.Module):
        def __init__(self, d: int = 8):
            super().__init__()
            self.attention = Attention(d)

        def forward(self, x):
            return self.attention(x)

    class _SingleBase:
        arch = "zimage_diffsynth"
        target_lora_modules = ["Attention"]

    torch.manual_seed(1)
    main_wrapper = _UnetWrapperStub(_SingleDiT())
    torch.manual_seed(2)
    sampling_wrapper = _UnetWrapperStub(_SingleDiT())
    base = _SingleBase()

    main_net = _build_peft_dora_network(main_wrapper, base)
    sampling_net = _build_peft_dora_network(sampling_wrapper, base)
    sampling_net.share_parameters_with(main_net)

    x = torch.randn(2, 8, 8, dtype=torch.float32)

    # Step 0: identity (calibrated magnitude == ||W_sampling||).
    sampling_net.is_active = False
    with torch.no_grad():
        base_out = sampling_net.peft_model(x)
    sampling_net.is_active = True
    with torch.no_grad():
        step0_out = sampling_net.peft_model(x)
    assert torch.allclose(step0_out, base_out, atol=1e-4)

    # Simulate a training update on main's magnitude.
    main_mags = _collect_magnitude_modules(main_net)
    with torch.no_grad():
        for m in main_mags:
            m.weight.mul_(1.5)

    # After the update, calibrated mag_norm_scale = 1.5 -> adapter_out = 1.5 * base.
    sampling_net.is_active = True
    with torch.no_grad():
        updated_out = sampling_net.peft_model(x)

    assert torch.allclose(updated_out, 1.5 * base_out, atol=1e-3), (
        "sampling forward did not reflect the calibrated main magnitude update: "
        f"max diff vs 1.5*base = {(updated_out - 1.5 * base_out).abs().max().item()}"
    )


def test_peft_non_dora_share_parameters_unchanged():
    """A plain peft (non-DoRA) network must be unaffected by the calibration
    change: no magnitude vector exists, so share_parameters_with shares only
    lora_A/lora_B and step-0 output stays identity (lora_B = 0)."""
    main_wrapper = _build_dit_wrapper(seed=1)
    sampling_wrapper = _build_dit_wrapper(seed=2)
    base = _StubBaseModel()

    from toolkit.peft_network import PeftNetwork

    main_net = PeftNetwork(
        text_encoder=None,
        unet=main_wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type="peft",
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    sampling_net = PeftNetwork(
        text_encoder=None,
        unet=sampling_wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type="peft",
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )

    sampling_net.share_parameters_with(main_net)

    x = torch.randn(2, 8, 8, dtype=torch.float32)
    sampling_net.is_active = False
    with torch.no_grad():
        base_out = sampling_net.peft_model(x)
    sampling_net.is_active = True
    with torch.no_grad():
        adapter_out = sampling_net.peft_model(x)

    assert torch.allclose(adapter_out, base_out, atol=1e-5)


if __name__ == "__main__":
    test_peft_dora_share_parameters_is_identity_at_step0()
    test_peft_dora_magnitude_is_shared_by_reference()
    test_peft_dora_calibration_tracks_main_magnitude_update()
    test_peft_non_dora_share_parameters_unchanged()
    print("All peft_dora calibration tests passed.")
