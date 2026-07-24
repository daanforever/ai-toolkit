"""
PEFT LoRA forward smoke test on a quanto-quantized base.

Verifies the core phase-1 proof-of-concept for the `peft`/`peft_dora` network
types: a `PeftNetwork` built on top of an `optimum.quanto`-quantized base
(``qfloat8`` QLinear leaves) runs a forward pass without invoking any
user-side ``dequantize()`` helper. PEFT's ``LoraLayer.forward`` calls
``self.base_layer(x)`` and quanto's ``QBytesTensor`` handles dequantization
internally, so the toolkit's manual ``dequantize_parameter`` path should not
be reached during this forward.

Runs on CPU and does not require the full Z-Image model.
"""

import torch
import torch.nn as nn
import pytest

from toolkit.dequantize import dequantize_parameter


class Attention(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d)])

    def forward(self, x):
        return self.to_out[0](self.to_v(x))


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
        return {k.replace("transformer.", "diffusion_model."): v for k, v in sd.items()}

    def convert_lora_weights_before_load(self, sd):
        return {k.replace("diffusion_model.", "transformer."): v for k, v in sd.items()}


def _quantize_stub(model: nn.Module) -> nn.Module:
    """Quantize all Linear leaves to qfloat8 in-place via optimum.quanto."""
    from toolkit.util.quantize import quantize
    quantize(model, weights="qfloat8")
    from optimum.quanto import freeze
    freeze(model)
    return model


def _build_quantized_peft_network(network_type: str):
    from toolkit.peft_network import PeftNetwork

    dit = _InnerDiTStub(d=8, n_blocks=2)
    wrapper = _UnetWrapperStub(dit)
    _quantize_stub(wrapper)
    base = _StubBaseModel()
    net = PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type=network_type,
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    return net, wrapper


@pytest.mark.parametrize("network_type", ["peft", "peft_dora"])
def test_peft_quantized_base_forward_no_dequantize(network_type):
    """A PEFT LoRA/DoRA forward through a quanto-quantized base must produce the
    expected output shape and must not call toolkit.dequantize.dequantize_parameter
    (PEFT + quanto handle dequant internally)."""
    net, wrapper = _build_quantized_peft_network(network_type)

    # Confirm the base layers were actually quantized.
    base_linears = [
        m for m in wrapper.modules()
        if m.__class__.__name__ == "QLinear"
    ]
    assert base_linears, "expected at least one QLinear base layer after quantize()"

    # Guard: dequantize_parameter must not be called during the forward pass.
    import toolkit.dequantize as deq_mod
    original = deq_mod.dequantize_parameter
    called = {"count": 0}

    def _spy(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    deq_mod.dequantize_parameter = _spy
    try:
        x = torch.randn(2, 8, 8, dtype=torch.float32)
        out = net.peft_model(x)
    finally:
        deq_mod.dequantize_parameter = original

    assert called["count"] == 0, (
        f"toolkit.dequantize.dequantize_parameter was called {called['count']} "
        f"time(s) during PEFT forward — PEFT+quanto should dequant internally"
    )
    assert out.shape == (2, 8, 8), f"unexpected output shape: {out.shape}"


@pytest.mark.parametrize("network_type", ["peft", "peft_dora"])
def test_peft_force_to_preserves_base_dtype(network_type):
    """force_to(network.dtype) must cast only adapters; frozen base stays model.dtype."""
    from toolkit.peft_network import PeftNetwork

    model_dtype = torch.bfloat16
    network_dtype = torch.float32

    dit = _InnerDiTStub(d=8, n_blocks=1).to(dtype=model_dtype)
    wrapper = _UnetWrapperStub(dit)
    base = _StubBaseModel()
    net = PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type=network_type,
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    net.force_to(torch.device("cpu"), network_dtype)

    adapter_ids = {id(p) for p in net.parameters()}
    assert adapter_ids, "expected adapter parameters from network.parameters()"
    assert all(p.dtype == network_dtype for p in net.parameters())

    frozen = [p for p in net.peft_model.parameters() if id(p) not in adapter_ids]
    assert frozen, "expected frozen base parameters"
    assert all(p.dtype == model_dtype for p in frozen), (
        f"frozen base must stay {model_dtype}, got "
        f"{sorted({str(p.dtype) for p in frozen})}"
    )
    assert all(id(p) in adapter_ids for p in net.parameters())


@pytest.mark.parametrize("network_type", ["peft", "peft_dora"])
def test_peft_to_dtype_preserves_base_dtype(network_type):
    """network.to(..., dtype=fp32) must not cast frozen base weights."""
    from toolkit.peft_network import PeftNetwork

    model_dtype = torch.bfloat16
    network_dtype = torch.float32

    dit = _InnerDiTStub(d=8, n_blocks=1).to(dtype=model_dtype)
    wrapper = _UnetWrapperStub(dit)
    base = _StubBaseModel()
    net = PeftNetwork(
        text_encoder=None,
        unet=wrapper,
        multiplier=1.0,
        lora_dim=2,
        alpha=2.0,
        train_unet=True,
        train_text_encoder=False,
        network_type=network_type,
        base_model=base,
        target_lin_modules=base.target_lora_modules,
    )
    net.to(device=torch.device("cpu"), dtype=network_dtype)

    adapter_ids = {id(p) for p in net.parameters()}
    assert adapter_ids
    assert all(p.dtype == network_dtype for p in net.parameters())
    frozen = [p for p in net.peft_model.parameters() if id(p) not in adapter_ids]
    assert frozen
    assert all(p.dtype == model_dtype for p in frozen)


@pytest.mark.parametrize("network_type", ["peft", "peft_dora"])
def test_peft_force_to_preserves_quantized_base(network_type):
    """force_to(fp32) must keep quanto qfloat8 base quantized; adapters become fp32."""
    net, wrapper = _build_quantized_peft_network(network_type)
    network_dtype = torch.float32
    net.force_to(torch.device("cpu"), network_dtype)

    qlinears = [m for m in wrapper.modules() if m.__class__.__name__ == "QLinear"]
    assert qlinears, "expected QLinear base layers after quantize()"

    adapter_ids = {id(p) for p in net.parameters()}
    assert adapter_ids
    assert all(p.dtype == network_dtype for p in net.parameters())

    # Quantized weight payloads must not become plain float32 Parameters.
    for linear in qlinears:
        weight = linear.weight
        assert type(weight).__name__ in ("QBytesTensor", "QTensor", "Float8Tensor") or hasattr(
            weight, "_data"
        ), f"base weight dequantized to {type(weight)}"
        assert weight.dtype != torch.float32 or hasattr(weight, "_data"), (
            "quantized base weight unexpectedly became plain float32"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
