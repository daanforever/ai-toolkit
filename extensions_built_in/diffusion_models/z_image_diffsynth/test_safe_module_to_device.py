"""CUDA tests: safe_module_to_device relocates quanto payloads and keeps LoRA identity."""

import pytest
import torch
import torch.nn as nn


def _assert_qweight_on_device(module: nn.Module, device: torch.device) -> None:
    """Assert quantized weight payload tensors match device (quanto or torchao)."""
    from toolkit.util.device import devices_equal

    checked = 0
    for _, param in module.named_parameters():
        payloads = []
        if hasattr(param, "_data") and hasattr(param, "_scale"):
            payloads = [param._data, param._scale]
        elif hasattr(param, "qdata") and hasattr(param, "scale"):
            payloads = [param.qdata, param.scale]
        else:
            continue
        assert devices_equal(param.device, device), f"param.device={param.device}"
        for t in payloads:
            assert devices_equal(t.device, device), f"payload.device={t.device}"
        checked += 1
    assert checked > 0, "expected at least one quantized weight with payload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_safe_module_to_device_moves_qbytes_payload():
    """safe_module_to_device must relocate QBytesTensor _data/_scale, not only param.device."""
    from toolkit.util.quantize import quantize, get_qtype
    from toolkit.util.device import safe_module_to_device
    from optimum.quanto import freeze

    device = torch.device("cuda")
    mod = nn.Linear(8, 8).to(device=device, dtype=torch.bfloat16)
    try:
        quantize(mod, weights=get_qtype("float8"))
        freeze(mod)
    except Exception as e:
        pytest.skip(f"quanto float8 quantize failed: {e}")

    safe_module_to_device(mod, torch.device("cpu"))
    _assert_qweight_on_device(mod, torch.device("cpu"))

    safe_module_to_device(mod, device)
    _assert_qweight_on_device(mod, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_safe_module_to_device_preserves_lora_param_identity_and_grads():
    """LoRA Parameter ids must survive cpu/cuda move (optimizer/PEFT cache); mimic sample restore."""
    from toolkit.util.quantize import quantize, get_qtype
    from toolkit.util.device import safe_module_to_device
    from optimum.quanto import freeze

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Module, rank: int = 4):
            super().__init__()
            self.base = base
            in_f = base.in_features
            out_f = base.out_features
            self.lora_A = nn.Linear(in_f, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_f, bias=False)
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.base(x) + self.lora_B(
                self.lora_A(x.to(dtype=self.lora_A.weight.dtype))
            ).to(dtype=x.dtype)

    device = torch.device("cuda")
    base = nn.Linear(8, 8).to(device=device, dtype=torch.bfloat16)
    try:
        quantize(base, weights=get_qtype("float8"))
        freeze(base)
    except Exception as e:
        pytest.skip(f"quanto float8 quantize failed: {e}")

    for p in base.parameters():
        p.requires_grad = False

    mod = _LoRALinear(base).to(device=device, dtype=torch.bfloat16)
    for n, p in mod.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # Cache refs as optimizer / _PeftLoraAdapter._param_cache would.
    cached = [p for n, p in mod.named_parameters() if "lora_" in n and p.requires_grad]
    assert cached, "expected trainable LoRA params"
    cached_ids = [id(p) for p in cached]

    safe_module_to_device(mod, torch.device("cpu"))
    _assert_qweight_on_device(mod.base, torch.device("cpu"))
    live = [p for n, p in mod.named_parameters() if "lora_" in n]
    assert [id(p) for p in live] == cached_ids, "LoRA Parameter identity must be preserved"

    safe_module_to_device(mod, device)
    _assert_qweight_on_device(mod.base, device)
    live = [p for n, p in mod.named_parameters() if "lora_" in n]
    assert [id(p) for p in live] == cached_ids, "LoRA Parameter identity must be preserved"

    # Mimic restore_device_state (unet.requires_grad_(False)) + ensure_params_requires_grad.
    mod.requires_grad_(False)
    for p in cached:
        p.requires_grad_(True)

    x = torch.randn(2, 8, device=device, dtype=torch.bfloat16)
    out = mod(x)
    out.sum().backward()
    assert any(p.grad is not None for p in cached), "expected LoRA grads after restore mimic"
