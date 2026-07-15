"""Per-block torch.compile for DiffSynth Z-Image DiT ModuleLists."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from toolkit.util.debug import is_debug_enabled
from toolkit.util.device import devices_equal, safe_module_to_device

try:
    from diffusers.utils.torch_utils import is_compiled_module
except ImportError:  # pragma: no cover
    def is_compiled_module(module):  # type: ignore[misc]
        return hasattr(module, "_orig_mod")


_DIT_BLOCK_ATTRS = ("layers", "noise_refiner", "context_refiner")


def discover_dit_block_names(dit: nn.Module) -> List[str]:
    """Return ModuleList attribute names that hold DiT blocks."""
    names = []
    for name in _DIT_BLOCK_ATTRS:
        if isinstance(getattr(dit, name, None), nn.ModuleList):
            names.append(name)
    return names


def unwrap_compiled_module_lists(
    dit: nn.Module,
    block_names: Optional[Sequence[str]] = None,
) -> int:
    """Replace OptimizedModule entries with their `_orig_mod`. Returns unwrap count."""
    if block_names is None:
        block_names = discover_dit_block_names(dit)
    count = 0
    for name in block_names:
        module_list = getattr(dit, name, None)
        if not isinstance(module_list, nn.ModuleList):
            continue
        for i, block in enumerate(module_list):
            if is_compiled_module(block) or hasattr(block, "_orig_mod"):
                module_list[i] = block._orig_mod
                count += 1
    return count


def _device_is_cpu(device: Union[torch.device, str, int, None]) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cpu"
    if isinstance(device, str):
        return device == "cpu" or device.startswith("cpu:")
    if isinstance(device, int):
        return False
    return False


def _module_params_on_cpu(module: nn.Module) -> bool:
    try:
        p = next(module.parameters(), None)
    except Exception:
        return False
    return p is None or p.device.type == "cpu"


def _resolve_target_device(
    module: nn.Module, *args, **kwargs
) -> Optional[torch.device]:
    """Target device from nn.Module.to(*args, **kwargs). None = dtype-only / no move."""
    if "device" in kwargs:
        return torch.device(kwargs["device"])
    if args:
        first = args[0]
        if isinstance(first, torch.device):
            return first
        if isinstance(first, str):
            return torch.device(first)
        if isinstance(first, torch.Tensor):
            return first.device
        if isinstance(first, int):
            return torch.device("cuda", first)
        if isinstance(first, torch.dtype):
            return None
    if "dtype" in kwargs or "memory_format" in kwargs:
        return None
    return None


def _module_already_on_device(module: nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for p in module.parameters():
        if not devices_equal(p.device, target):
            return False
    for b in module.buffers():
        if not devices_equal(b.device, target):
            return False
    return True


def _resolve_result_is_cpu(module: nn.Module, *args, **kwargs) -> bool:
    """Whether module will be / is on CPU after applying to(*args, **kwargs)."""
    target = _resolve_target_device(module, *args, **kwargs)
    if target is not None:
        return _device_is_cpu(target)
    return _module_params_on_cpu(module)


def move_dit_with_compiled_blocks(
    dit: nn.Module,
    block_names: Optional[Sequence[str]] = None,
    *args,
    log_fn: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> nn.Module:
    """Move DiT without Module.to() when compiled+quantized.

    - Same-device (ignore dtype): early return; keep compiled blocks.
    - Real device change: unwrap → Parameter/buffer replace move → recompile on GPU.

    Avoids ``Couldn't swap ... weight`` / weakref failures from torch.compile + quanto.
    After a CPU offload, blocks stay unwrapped; the next non-CPU move recompiles.
    """
    log = log_fn or (lambda _msg: None)
    if block_names is None:
        block_names = discover_dit_block_names(dit)

    target = _resolve_target_device(dit, *args, **kwargs)
    if target is None or _module_already_on_device(dit, target):
        return dit

    unwrapped = unwrap_compiled_module_lists(dit, block_names)
    if unwrapped:
        setattr(dit, "_zimage_blocks_need_recompile", True)
        if is_debug_enabled():
            log(f"[compile] unwrapped {unwrapped} block(s) for device move")

    # Device only — do not cast QBytesTensor scale dtype via torch_dtype.
    safe_module_to_device(dit, torch.device(target), dtype=None)

    need_recompile = getattr(dit, "_zimage_blocks_need_recompile", False)
    if need_recompile and not _device_is_cpu(target):
        compile_dit_module_lists(dit, block_names, log_fn=log)
        setattr(dit, "_zimage_blocks_need_recompile", False)
    return dit


def compile_dit_module_lists(
    dit: nn.Module,
    block_names: Sequence[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """Compile each block in the named ModuleLists with torch.compile(dynamic=True).

    Returns counts: ok / failed / skipped.
    Soft-fails per block so a single inductor error does not abort training setup.
    """
    log = log_fn or (lambda _msg: None)
    stats = {"ok": 0, "failed": 0, "skipped": 0}
    compiled_names: List[str] = []

    for name in block_names:
        module_list = getattr(dit, name, None)
        if not isinstance(module_list, nn.ModuleList):
            log(f"[compile] skip '{name}': not a ModuleList ({type(module_list).__name__})")
            stats["skipped"] += 1
            continue

        for i, block in enumerate(module_list):
            if is_compiled_module(block) or hasattr(block, "_orig_mod"):
                stats["skipped"] += 1
                continue
            try:
                module_list[i] = torch.compile(block, dynamic=True)
                stats["ok"] += 1
                compiled_names.append(f"{name}[{i}]")
            except Exception as e:
                stats["failed"] += 1
                log(f"[compile] failed {name}[{i}]: {e}")

    log(
        f"[compile] Dit blocks done: ok={stats['ok']} failed={stats['failed']} "
        f"skipped={stats['skipped']}"
    )
    if is_debug_enabled() and compiled_names:
        # Compact ranges: layers[0-29] instead of layers[0], layers[1], ...
        by_list: Dict[str, List[int]] = defaultdict(list)
        for full in compiled_names:
            list_name, idx_s = full.rsplit("[", 1)
            by_list[list_name].append(int(idx_s.rstrip("]")))
        parts = []
        for list_name, idxs in by_list.items():
            idxs.sort()
            start = prev = idxs[0]
            ranges = []
            for i in idxs[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                ranges.append(f"{start}-{prev}" if start != prev else str(start))
                start = prev = i
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            parts.append(f"{list_name}[{','.join(ranges)}]")
        log(f"[compile] compiled blocks: {', '.join(parts)}")
    return stats
