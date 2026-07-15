"""Per-block torch.compile for DiffSynth Z-Image DiT ModuleLists."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from toolkit.util.debug import is_debug_enabled

try:
    from diffusers.utils.torch_utils import is_compiled_module
except ImportError:  # pragma: no cover
    def is_compiled_module(module):  # type: ignore[misc]
        return hasattr(module, "_orig_mod")


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
