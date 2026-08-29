"""Load fp32 LoRA safetensors and compute Delta metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def _load_lora_mod():
    """File-path import of z_image_diffsynth.lora (avoids parent DiT package init)."""
    name = "_zids_exp_lora"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "lora.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_lora_state(path: Path | str) -> dict[str, torch.Tensor]:
    """Load safetensors and normalize DiffSynth keys to toolkit convention."""
    path = Path(path)
    raw = load_file(str(path))
    lora = _load_lora_mod()
    return lora.convert_lora_weights_before_load(raw)


def find_latest_lora(save_root: Path | str) -> Path | None:
    """Latest ``*.safetensors`` under save_root (by mtime)."""
    save_root = Path(save_root)
    if not save_root.is_dir():
        return None
    files = sorted(
        (p for p in save_root.glob("*.safetensors") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def _is_weight_key(key: str) -> bool:
    k = key.lower()
    return (
        "lora_down" in k
        or "lora_up" in k
        or "lora_a" in k
        or "lora_b" in k
    ) and key.endswith(".weight")


def _role(key: str) -> str | None:
    k = key.lower()
    if "lora_down" in k or "lora_a" in k:
        return "down"
    if "lora_up" in k or "lora_b" in k:
        return "up"
    return None


def delta_tensors(
    warm: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-key float32 Delta = after - warm (intersecting weight keys)."""
    keys = sorted(set(warm) & set(after))
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        if not _is_weight_key(key):
            continue
        out[key] = after[key].float() - warm[key].float()
    return out


def subtract_deltas(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-key a − b over shared keys."""
    return {k: a[k].float() - b[k].float() for k in sorted(set(a) & set(b))}


def update_rms(delta: dict[str, torch.Tensor]) -> float:
    """RMS over all elements of all Delta tensors."""
    if not delta:
        return 0.0
    flats = [t.reshape(-1).float() for t in delta.values()]
    cat = torch.cat(flats)
    return float(torch.sqrt(torch.mean(cat * cat)).item())


def cosine_delta(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
) -> float:
    """Cosine similarity of flattened Deltas over shared keys."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    fa = torch.cat([a[k].reshape(-1).float() for k in keys])
    fb = torch.cat([b[k].reshape(-1).float() for k in keys])
    denom = float(fa.norm().item() * fb.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(fa, fb).item() / denom)


def split_down_up(
    delta: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    down: dict[str, torch.Tensor] = {}
    up: dict[str, torch.Tensor] = {}
    for key, tens in delta.items():
        role = _role(key)
        if role == "down":
            down[key] = tens
        elif role == "up":
            up[key] = tens
    return down, up


def per_key_rms_ratios(
    base: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
) -> list[float]:
    """Per-shared-key ||other|| / ||base|| (element RMS)."""
    ratios: list[float] = []
    for key in sorted(set(base) & set(other)):
        b = base[key].float().reshape(-1)
        o = other[key].float().reshape(-1)
        rb = float(torch.sqrt(torch.mean(b * b)).item())
        if rb < 1e-12:
            continue
        ro = float(torch.sqrt(torch.mean(o * o)).item())
        ratios.append(ro / rb)
    return ratios


def summarize_delta_pair(
    warm: dict[str, torch.Tensor],
    base_after: dict[str, torch.Tensor],
    other_after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    d_base = delta_tensors(warm, base_after)
    d_other = delta_tensors(warm, other_after)
    down_b, up_b = split_down_up(d_base)
    down_o, up_o = split_down_up(d_other)
    ratios = per_key_rms_ratios(d_base, d_other)
    median_ratio = float(torch.tensor(ratios).median().item()) if ratios else 0.0
    return {
        "update_rms_base": update_rms(d_base),
        "update_rms_other": update_rms(d_other),
        "ratio": (
            update_rms(d_other) / update_rms(d_base)
            if update_rms(d_base) > 1e-12
            else 0.0
        ),
        "cosine": cosine_delta(d_base, d_other),
        "median_per_key_ratio": median_ratio,
        "down_ratio": (
            update_rms(down_o) / update_rms(down_b)
            if update_rms(down_b) > 1e-12
            else 0.0
        ),
        "up_ratio": (
            update_rms(up_o) / update_rms(up_b) if update_rms(up_b) > 1e-12 else 0.0
        ),
        "n_keys": len(d_base),
    }


def block_index_for_key(key: str) -> int | None:
    lora = _load_lora_mod()
    parsed = lora.parse_lora_block(key)
    if parsed is None:
        return None
    return parsed[1]
