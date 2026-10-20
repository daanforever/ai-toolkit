# Stable-LoRA: shrink LoRA A before the optimizer step until rms(A) <= rms(B).

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

import torch

from toolkit.optimizers.optimizer_utils import copy_stochastic

STABILITY_CHECK_INTERVAL = 10
_FOREACH_CHUNK = 8


@dataclass
class StableLoraPair:
    A: torch.Tensor
    B: torch.Tensor
    stable: bool = False


@dataclass
class StableLoraSetup:
    pairs: List[StableLoraPair]
    skipped_full_rank: int = 0
    skipped_frozen: int = 0
    skipped_unrecognized: int = 0
    skip_reason: Optional[str] = None


def validate_shrink_percent(value) -> float:
    """Return the percent, or raise ValueError when it is outside [0, 100)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"stable_lora_shrink must be a number in [0, 100), got {value!r}"
        )
    percent = float(value)
    if not math.isfinite(percent) or percent < 0.0 or percent >= 100.0:
        raise ValueError(
            f"stable_lora_shrink must be a number in [0, 100), got {value!r}"
        )
    return percent


def shrink_percent_from_configs(*configs) -> float:
    """Read stable_lora_shrink from the first config that sets it. Default is 1."""
    for config in configs:
        kwargs = getattr(config, "network_kwargs", None) if config is not None else None
        if isinstance(kwargs, dict) and "stable_lora_shrink" in kwargs:
            return validate_shrink_percent(kwargs["stable_lora_shrink"])
    return 1.0


def should_check(step: int, interval: int = STABILITY_CHECK_INTERVAL) -> bool:
    """True on step 1 and every ``interval`` steps after that. ``step`` is 1-based."""
    if step < 1:
        raise ValueError(f"stable LoRA step must be >= 1, got {step}")
    return step == 1 or step % interval == 0


def _as_weight(module_or_param):
    if isinstance(module_or_param, torch.nn.Parameter):
        return module_or_param
    weight = getattr(module_or_param, "weight", None)
    if isinstance(weight, torch.nn.Parameter):
        return weight
    return None


def _append_toolkit_pair(module, setup: StableLoraSetup) -> bool:
    """Consume a toolkit LoRA module. Return True when the module is that form."""
    if not (hasattr(module, "lora_down") and hasattr(module, "lora_up")):
        return False
    if getattr(module, "full_rank", False) or _as_weight(module.lora_up) is None:
        setup.skipped_full_rank += 1
        return True
    down = _as_weight(module.lora_down)
    up = _as_weight(module.lora_up)
    if down is None or up is None:
        setup.skipped_unrecognized += 1
        return True
    if down.numel() == 0 or up.numel() == 0:
        setup.skipped_unrecognized += 1
        return True
    if not down.requires_grad:
        setup.skipped_frozen += 1
        return True
    setup.pairs.append(StableLoraPair(A=down, B=up))
    return True


def _append_peft_pairs(module, setup: StableLoraSetup) -> bool:
    layer = getattr(module, "layer", None)
    if layer is None or not (hasattr(layer, "lora_A") and hasattr(layer, "lora_B")):
        return False
    lora_a = layer.lora_A
    lora_b = layer.lora_B
    found = False
    for name in list(lora_a.keys()):
        if name not in lora_b:
            setup.skipped_unrecognized += 1
            found = True
            continue
        down = _as_weight(lora_a[name])
        up = _as_weight(lora_b[name])
        if down is None or up is None:
            setup.skipped_unrecognized += 1
            found = True
            continue
        found = True
        if down.numel() == 0 or up.numel() == 0:
            setup.skipped_unrecognized += 1
            continue
        if not down.requires_grad:
            setup.skipped_frozen += 1
            continue
        setup.pairs.append(StableLoraPair(A=down, B=up))
    return found


def setup_stable_lora(network) -> StableLoraSetup:
    """Build A/B pairs from ``network.unet_loras``. ``peft_dora`` is skipped."""
    network_type = str(getattr(network, "network_type", "") or "").lower()
    if network_type == "peft_dora":
        return StableLoraSetup(pairs=[], skip_reason="peft_dora")

    setup = StableLoraSetup(pairs=[])
    for module in getattr(network, "unet_loras", None) or []:
        if _append_toolkit_pair(module, setup):
            continue
        if _append_peft_pairs(module, setup):
            continue
        setup.skipped_unrecognized += 1
    if not setup.pairs:
        setup.skip_reason = "no_pairs"
    return setup


def _percent_text(percent: float) -> str:
    text = f"{percent:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_setup_log(percent: float, setup: StableLoraSetup) -> str:
    label = _percent_text(percent)
    if setup.skip_reason == "peft_dora":
        return f"[zimage_diffsynth] stable_lora_shrink={label} skipped: peft_dora"
    if setup.skip_reason == "no_pairs":
        return (
            f"[zimage_diffsynth] stable_lora_shrink={label} skipped: no LoRA A/B pairs"
        )
    parts = [
        f"[zimage_diffsynth] stable_lora_shrink={label} pairs={len(setup.pairs)} "
        f"check_every={STABILITY_CHECK_INTERVAL}"
    ]
    if setup.skipped_full_rank:
        parts.append(f"skipped_full_rank={setup.skipped_full_rank}")
    if setup.skipped_frozen:
        parts.append(f"skipped_frozen={setup.skipped_frozen}")
    if setup.skipped_unrecognized:
        parts.append(f"skipped_unrecognized={setup.skipped_unrecognized}")
    return " ".join(parts)


def format_stable_log(stable: int, total: int) -> str:
    return f"[zimage_diffsynth] stable_lora stable={stable}/{total}"


def _rms(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """fp32 RMS on the tensors' device. The caller performs the only host sync."""
    parts: List[torch.Tensor] = []
    for start in range(0, len(tensors), _FOREACH_CHUNK):
        chunk = list(tensors[start : start + _FOREACH_CHUNK])
        floats = [tensor.detach().float().reshape(-1) for tensor in chunk]
        norms = torch._foreach_norm(floats)
        parts.append(torch.stack(list(norms)))
        del floats
    stacked = torch.cat(parts) if len(parts) > 1 else parts[0]
    counts = torch.tensor(
        [tensor.numel() for tensor in tensors],
        device=stacked.device,
        dtype=torch.float32,
    )
    return stacked / torch.sqrt(counts)


def mark_stable(pairs: Sequence[StableLoraPair]) -> int:
    """Set ``stable`` where rms(A) <= rms(B). Returns how many flags flipped."""
    active = [pair for pair in pairs if not pair.stable]
    if not active:
        return 0
    by_device = {}
    for pair in active:
        by_device.setdefault((pair.A.device, pair.B.device), []).append(pair)
    newly = 0
    for group in by_device.values():
        rms_a = _rms([pair.A for pair in group])
        rms_b = _rms([pair.B for pair in group]).to(device=rms_a.device)
        flags = (rms_a <= rms_b).tolist()
        for pair, flag in zip(group, flags):
            if flag and not pair.stable:
                pair.stable = True
                newly += 1
    return newly


def shrink_matrix(matrix: torch.Tensor, factor: float) -> None:
    """Multiply ``matrix`` by ``factor`` in fp32. Non-fp32 uses stochastic rounding."""
    if matrix.dtype == torch.float32:
        with torch.no_grad():
            matrix.mul_(factor)
        return
    if matrix.device.type == "cpu":
        raise RuntimeError(
            "stable_lora_shrink for non-fp32 weights requires CUDA "
            f"(dtype={matrix.dtype}, device={matrix.device})"
        )
    scaled = matrix.detach().float() * factor
    copy_stochastic(matrix, scaled)


def apply_shrink(
    pairs: Sequence[StableLoraPair],
    shrink_percent,
    *,
    check: bool,
) -> int:
    """Shrink non-stable A. Returns how many pairs became stable on this call.

    ``shrink_percent`` of 0 leaves every tensor and flag unchanged.
    """
    percent = validate_shrink_percent(shrink_percent)
    if percent == 0.0 or not pairs:
        return 0
    newly = mark_stable(pairs) if check else 0
    factor = 1.0 - percent / 100.0
    for pair in pairs:
        if pair.stable:
            continue
        shrink_matrix(pair.A, factor)
    return newly
