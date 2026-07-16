"""Regression: LoRA weights must stay trainable across sample/offload device moves.

a89d216 replaced every Parameter in safe_module_to_device; optimizer kept orphaned
refs while save_weights wrote frozen module tensors — resume looked like a progress wipe.
9d82a69 keeps LoRA identity via param.data=; this test closes the corrupt-save path.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from toolkit.util.device import safe_module_to_device


class _LoRALinear(nn.Module):
    def __init__(self, in_f: int = 8, out_f: int = 8, rank: int = 4):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x))


def _trainable_lora(mod: nn.Module):
    return [p for n, p in mod.named_parameters() if "lora_" in n and p.requires_grad]


def _train_steps(mod: nn.Module, opt: torch.optim.Optimizer, device: torch.device, n: int) -> None:
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 8, device=device, dtype=torch.float32)
        loss = mod(x).pow(2).mean()
        loss.backward()
        opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_train_device_move_save_load_roundtrip():
    """train → cpu/cuda move → train → save/load must restore post-train LoRA weights."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    mod = _LoRALinear().to(device=device, dtype=torch.float32)
    for p in _trainable_lora(mod):
        p.requires_grad_(True)

    opt = torch.optim.AdamW(_trainable_lora(mod), lr=1e-2)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}

    _train_steps(mod, opt, device, n=3)
    pre_move = {n: p.detach().float().cpu().clone() for n, p in mod.named_parameters() if "lora_" in n}

    safe_module_to_device(mod, torch.device("cpu"))
    live = _trainable_lora(mod)
    assert {id(p) for p in live} == opt_ids, "LoRA Parameter ids must stay in optimizer after cpu move"

    safe_module_to_device(mod, device)
    live = _trainable_lora(mod)
    assert {id(p) for p in live} == opt_ids, "LoRA Parameter ids must stay in optimizer after cuda move"

    _train_steps(mod, opt, device, n=3)
    post_train = {n: p.detach().float().cpu().clone() for n, p in mod.named_parameters() if "lora_" in n}

    # Post-move training must have changed weights (not frozen at pre-move).
    assert any(not torch.allclose(post_train[k], pre_move[k]) for k in post_train)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lora.pt")
        torch.save({n: p.detach().cpu() for n, p in mod.named_parameters() if "lora_" in n}, path)

        torch.manual_seed(1)
        fresh = _LoRALinear().to(device=device, dtype=torch.float32)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = fresh.load_state_dict(loaded, strict=False)
        assert not unexpected
        # base weights intentionally absent from the checkpoint
        assert all("lora_" not in m for m in missing)

        for name, expected in post_train.items():
            got = dict(fresh.named_parameters())[name].detach().float().cpu()
            assert torch.allclose(got, expected, atol=0, rtol=0), (
                f"{name} after load != post-train (corrupt save / orphan Parameter)"
            )
