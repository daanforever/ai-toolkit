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
import pytest


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


class ZImageTransformerBlock(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.attention = Attention(d)
        self.feed_forward = FeedForward(d)


class _InnerDiTStub(nn.Module):
    def __init__(self, d: int = 8, n_blocks: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([ZImageTransformerBlock(d) for _ in range(n_blocks)])

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
    target_lora_modules = ["ZImageTransformerBlock"]

    def convert_lora_weights_before_save(self, sd):
        return sd

    def convert_lora_weights_before_load(self, sd):
        return sd


def _build_dit_wrapper(seed: int, d: int = 8):
    torch.manual_seed(seed)
    dit = _InnerDiTStub(d=d, n_blocks=2)
    return _UnetWrapperStub(dit)


def _build_peft_dora_network(wrapper, base, multiplier=1.0):
    from toolkit.peft_network import PeftNetwork

    return PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=multiplier,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peft_dora_shared_magnitude_survives_safe_device_move():
    """Shared DoRA magnitude Parameter ids must survive safe_module_to_device.

    Magnitude is a Parameter (identity path), not a buffer. _PeftLoraAdapter
    caches Parameter refs only — no buffer cache to invalidate on move.
    """
    from toolkit.util.device import safe_module_to_device

    device = torch.device("cuda")
    main_wrapper = _build_dit_wrapper(seed=1)
    sampling_wrapper = _build_dit_wrapper(seed=2)
    base = _StubBaseModel()

    main_net = _build_peft_dora_network(main_wrapper, base)
    sampling_net = _build_peft_dora_network(sampling_wrapper, base)
    sampling_net.share_parameters_with(main_net)

    # Move peft_model to CUDA the same way training does before sample offload.
    safe_module_to_device(main_net.peft_model, device)
    safe_module_to_device(sampling_net.peft_model, device)

    main_mags = _collect_magnitude_modules(main_net)
    sampling_mags = _collect_magnitude_modules(sampling_net)
    assert main_mags and sampling_mags

    mag_params = [m.weight for m in main_mags]
    opt = torch.optim.AdamW(mag_params, lr=1e-3)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    shared_ids = [id(m.weight) for m in main_mags]

    for main_m, sampling_m in zip(main_mags, sampling_mags):
        assert sampling_m.weight is main_m.weight

    safe_module_to_device(main_net.peft_model, torch.device("cpu"))
    safe_module_to_device(main_net.peft_model, device)

    main_mags_after = _collect_magnitude_modules(main_net)
    sampling_mags_after = _collect_magnitude_modules(sampling_net)
    for main_m, sampling_m in zip(main_mags_after, sampling_mags_after):
        assert sampling_m.weight is main_m.weight, (
            "shared DoRA magnitude must remain the same object after device move"
        )
        assert id(main_m.weight) in opt_ids, (
            "DoRA magnitude Parameter must stay in the optimizer after device move"
        )

    assert [id(m.weight) for m in main_mags_after] == shared_ids


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


def _quantize_stub(model: nn.Module) -> nn.Module:
    """Quantize all Linear leaves to qfloat8 in-place via optimum.quanto and freeze."""
    from toolkit.util.quantize import quantize
    from optimum.quanto import freeze
    quantize(model, weights="qfloat8")
    freeze(model)
    return model


def test_peft_dora_quantized_main_is_identity_at_step0():
    """Reproduces the real production mismatch: the main DiT is quantized
    (qfloat8 via optimum.quanto -> QLinear base layers) while the sampling DiT
    stays fp32. PEFT's DoRA ``update_layer`` initializes the main magnitude from
    ``dequantize_module_weight(QLinear)`` (the qfloat8-dequantized norm) and the
    sampling magnitude from the fp32 base norm. These differ, so without
    calibration ``mag_norm_scale = ||W_main||_dequant / ||W_sampling||_fp32 != 1``
    and the step-0 DoRA output is corrupted. The calibration ratio
    ``||W_sampling|| / ||W_main||`` applied at read time restores identity.

    Uses the same seed for both DiTs so the only source of the norm mismatch is
    quantization precision (isolating the quantized-base code path).
    """
    torch.manual_seed(7)
    main_wrapper = _build_dit_wrapper(seed=7)
    sampling_wrapper = _build_dit_wrapper(seed=7)
    # Quantize the main base in-place; sampling stays fp32.
    _quantize_stub(main_wrapper)

    base = _StubBaseModel()
    main_net = _build_peft_dora_network(main_wrapper, base)
    sampling_net = _build_peft_dora_network(sampling_wrapper, base)

    # Precondition: main base layers are actually quantized to QLinear.
    main_qlinears = [m for m in main_wrapper.modules() if m.__class__.__name__ == "QLinear"]
    assert main_qlinears, "main stub was not quantized to QLinear"

    # Precondition: the qfloat8-dequant magnitude differs from the fp32 sampling
    # magnitude (quantization introduces a norm mismatch the calibration must fix).
    main_mags = _collect_magnitude_modules(main_net)
    sampling_mags = _collect_magnitude_modules(sampling_net)
    assert main_mags and sampling_mags
    assert not torch.allclose(main_mags[0].weight, sampling_mags[0].weight, atol=1e-6), (
        "quantized main magnitude must differ from fp32 sampling magnitude; "
        f"main={main_mags[0].weight.flatten()[:4].tolist()} "
        f"sampling={sampling_mags[0].weight.flatten()[:4].tolist()}"
    )

    sampling_net.share_parameters_with(main_net)

    x = torch.randn(2, 8, 8, dtype=torch.float32)

    sampling_net.is_active = False
    with torch.no_grad():
        base_out = sampling_net.peft_model(x)

    sampling_net.is_active = True
    with torch.no_grad():
        adapter_out = sampling_net.peft_model(x)

    assert torch.allclose(adapter_out, base_out, atol=1e-4), (
        "peft_dora sampling forward on a quantized main / fp32 sampling base is "
        "not identity at step 0 after share_parameters_with: "
        f"max diff = {(adapter_out - base_out).abs().max().item()}"
    )


def test_peft_dora_multiplier_neq_1_scales_calibrated_delta():
    """With multiplier != 1.0 the multiplier wrapper does a double forward:
    ``base_out`` (adapters disabled) and ``full_out`` (DoRA calibrated), then
    returns ``base_out + mult * (full_out - base_out)``.

    For a single-linear ``bias=False`` stub with ``lora_B = 0`` and a main
    magnitude update of ``*= 1.5``: at ``mult=1.0`` the calibrated DoRA output is
    ``1.5 * base`` (proven by ``test_peft_dora_calibration_tracks_main_magnitude_update``),
    so at ``mult=0.5`` the expected output is
    ``base + 0.5 * (1.5*base - base) = 1.25 * base``. This verifies the
    calibration is applied during the ``full_out`` sub-pass of the multiplier
    double-forward (not skipped or double-applied) and that the scaled delta is
    correct.
    """

    class Attention(nn.Module):  # name matches target_lora_modules
        def __init__(self, d: int = 8):
            super().__init__()
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
    sampling_net = _build_peft_dora_network(sampling_wrapper, base, multiplier=0.5)
    sampling_net.share_parameters_with(main_net)

    # Simulate a main magnitude update so DoRA is no longer identity (delta != 0).
    main_mags = _collect_magnitude_modules(main_net)
    with torch.no_grad():
        for m in main_mags:
            m.weight.mul_(1.5)

    x = torch.randn(2, 8, 8, dtype=torch.float32)

    sampling_net.is_active = False
    with torch.no_grad():
        base_out = sampling_net.peft_model(x)

    sampling_net.is_active = True
    with torch.no_grad():
        mult_half_out = sampling_net.peft_model(x)

    expected = 1.25 * base_out  # base + 0.5 * (1.5*base - base)
    assert torch.allclose(mult_half_out, expected, atol=1e-3), (
        "multiplier=0.5 did not produce base + 0.5*delta for calibrated DoRA: "
        f"max diff vs 1.25*base = {(mult_half_out - expected).abs().max().item()}"
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
    test_peft_dora_quantized_main_is_identity_at_step0()
    test_peft_dora_multiplier_neq_1_scales_calibrated_delta()
    test_peft_non_dora_share_parameters_unchanged()
    print("All peft_dora calibration tests passed.")
