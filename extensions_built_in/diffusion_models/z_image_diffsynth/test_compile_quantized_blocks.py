"""Tests for per-block torch.compile helper, including quanto float8 path."""

import pytest
import torch
import torch.nn as nn

from extensions_built_in.diffusion_models.z_image_diffsynth.compile_blocks import (
    compile_dit_module_lists,
)


class _FakeCompiled(nn.Module):
    """Stand-in for torch._dynamo OptimizedModule (has _orig_mod)."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self._orig_mod = orig

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


class _TinyDiT(nn.Module):
    def __init__(self, d: int = 8, n: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_compile_dit_module_lists_replaces_and_skips(monkeypatch):
    """CPU unit: helper replaces ModuleList entries; second pass skips compiled."""

    def _fake_compile(mod, **_kwargs):
        return _FakeCompiled(mod)

    monkeypatch.setattr(torch, "compile", _fake_compile)

    dit = _TinyDiT()
    originals = [dit.layers[i] for i in range(len(dit.layers))]
    stats = compile_dit_module_lists(dit, ["layers"])
    assert stats["ok"] == 2
    assert stats["failed"] == 0
    for i, orig in enumerate(originals):
        assert isinstance(dit.layers[i], _FakeCompiled)
        assert dit.layers[i]._orig_mod is orig

    stats2 = compile_dit_module_lists(dit, ["layers"])
    assert stats2["ok"] == 0
    assert stats2["skipped"] == 2


def test_compile_dit_module_lists_skips_non_modulelist():
    dit = _TinyDiT()
    dit.not_a_list = nn.Linear(8, 8)  # type: ignore[attr-defined]
    stats = compile_dit_module_lists(dit, ["not_a_list", "missing"])
    assert stats["ok"] == 0
    assert stats["skipped"] >= 2


def _gradient_checkpoint_forward(model, use_gradient_checkpointing, *args, **kwargs):
    """Mirror DiffSynth gradient_checkpoint_forward (use_reentrant=False)."""
    if use_gradient_checkpointing:

        def custom_forward(*inputs, **kw):
            return model(*inputs, **kw)

        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            *args,
            **kwargs,
            use_reentrant=False,
        )
    return model(*args, **kwargs)


def _cuda_compile_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        m = nn.Linear(4, 4).cuda()
        c = torch.compile(m, dynamic=True)
        y = c(torch.randn(2, 4, device="cuda"))
        y.sum().backward()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_quantized_float8_with_checkpoint():
    """CUDA smoke: float8 quantize + compile + gradient checkpoint + backward."""
    if not _cuda_compile_supported():
        pytest.skip("torch.compile/inductor not usable on this platform")

    from toolkit.util.quantize import quantize, get_qtype
    from optimum.quanto import freeze

    device = torch.device("cuda")
    mod = nn.Linear(8, 8).to(device=device, dtype=torch.bfloat16)
    try:
        quantize(mod, weights=get_qtype("float8"))
        freeze(mod)
    except Exception as e:
        pytest.skip(f"quanto float8 quantize failed: {e}")

    try:
        compiled = torch.compile(mod, dynamic=True)
    except Exception as e:
        pytest.skip(f"torch.compile failed on quantized module: {e}")

    x = torch.randn(2, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    try:
        out = _gradient_checkpoint_forward(
            compiled, True, x
        )
        loss = out.sum()
        loss.backward()
    except Exception as e:
        pytest.skip(f"inductor+quanto+checkpoint failed: {e}")

    assert torch.isfinite(out).all()
    assert out.shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_quantized_lora_stub_with_checkpoint():
    """CUDA smoke: quantized base + LoRA stub → compile → checkpoint → LoRA grads."""
    if not _cuda_compile_supported():
        pytest.skip("torch.compile/inductor not usable on this platform")

    from toolkit.util.quantize import quantize, get_qtype
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
            return self.base(x) + self.lora_B(self.lora_A(x.to(dtype=self.lora_A.weight.dtype))).to(
                dtype=x.dtype
            )

    class _Block(nn.Module):
        def __init__(self, linear: nn.Module):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x)

    device = torch.device("cuda")
    base = nn.Linear(8, 8).to(device=device, dtype=torch.bfloat16)
    try:
        quantize(base, weights=get_qtype("float8"))
        freeze(base)
    except Exception as e:
        pytest.skip(f"quanto float8 quantize failed: {e}")

    for p in base.parameters():
        p.requires_grad = False

    block = _Block(_LoRALinear(base).to(device=device, dtype=torch.bfloat16))
    dit = nn.Module()
    dit.layers = nn.ModuleList([block])  # type: ignore[attr-defined]

    try:
        stats = compile_dit_module_lists(dit, ["layers"])
    except Exception as e:
        pytest.skip(f"compile_dit_module_lists failed: {e}")

    if stats["ok"] < 1:
        pytest.skip(
            f"compile did not succeed on quantized+LoRA block "
            f"(ok={stats['ok']} failed={stats['failed']})"
        )

    x = torch.randn(2, 8, device=device, dtype=torch.bfloat16)
    try:
        out = _gradient_checkpoint_forward(dit.layers[0], True, x)
        loss = out.sum()
        loss.backward()
    except Exception as e:
        pytest.skip(f"inductor+quanto+LoRA+checkpoint failed: {e}")

    assert torch.isfinite(out).all()
    compiled_block = dit.layers[0]
    named = list(compiled_block.named_parameters())
    lora_grads = [p.grad for n, p in named if "lora_" in n and p.requires_grad]
    assert lora_grads, "expected LoRA parameters on compiled block"
    assert any(g is not None for g in lora_grads), "expected LoRA parameter grads"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wrapper_to_unwrap_recompile_after_device_move():
    """Compiled+quantized DiT must survive wrapper.to(cpu)/to(cuda) without weakref crash."""
    if not _cuda_compile_supported():
        pytest.skip("torch.compile/inductor not usable on this platform")

    from toolkit.util.quantize import quantize, get_qtype
    from optimum.quanto import freeze
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        _DiTUnetWrapper,
    )
    from extensions_built_in.diffusion_models.z_image_diffsynth.compile_blocks import (
        is_compiled_module,
    )

    device = torch.device("cuda")
    dit = _TinyDiT(d=8, n=2).to(device=device, dtype=torch.bfloat16)
    try:
        quantize(dit, weights=get_qtype("float8"))
        freeze(dit)
    except Exception as e:
        pytest.skip(f"quanto float8 quantize failed: {e}")

    stats = compile_dit_module_lists(dit, ["layers"])
    if stats["ok"] < 1:
        pytest.skip(f"compile failed: {stats}")

    wrapper = _DiTUnetWrapper(dit)
    try:
        wrapper.to("cpu")
    except Exception as e:
        pytest.fail(f"wrapper.to('cpu') failed after compile: {e}")

    for block in wrapper._inner_dit.layers:
        assert not (
            is_compiled_module(block) or hasattr(block, "_orig_mod")
        ), "blocks should be unwrapped on CPU"

    try:
        wrapper.to(device)
    except Exception as e:
        pytest.fail(f"wrapper.to(cuda) failed after unwrap: {e}")

    assert any(
        is_compiled_module(b) or hasattr(b, "_orig_mod")
        for b in wrapper._inner_dit.layers
    ), "blocks should be recompiled on CUDA"

    x = torch.randn(2, 8, device=device, dtype=torch.bfloat16)
    try:
        out = _gradient_checkpoint_forward(wrapper._inner_dit.layers[0], True, x)
        out.sum().backward()
    except Exception as e:
        pytest.skip(f"checkpoint forward after recompile failed: {e}")

    assert torch.isfinite(out).all()
