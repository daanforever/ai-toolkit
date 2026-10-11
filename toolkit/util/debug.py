"""
Debug utilities. memory_debug context manager measures GPU/RAM around a block;
enabled via set_debug_config(config) with config.debug.
"""
import contextlib
import gc
import sys
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

# --- RAM snapshot (Windows only; Unix/macOS not implemented yet) ---
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

_debug_config = None


def set_debug_config(config) -> None:
    """Register the config object used to decide if memory debug is enabled (config.debug)."""
    global _debug_config
    _debug_config = config


def is_debug_enabled() -> bool:
    """Return True if debug logging is enabled (config.debug). Used for optional debug messages."""
    if _debug_config is None:
        return False
    return bool(getattr(_debug_config, "debug", False))


def _is_enabled_for_cuda() -> bool:
    if _debug_config is None:
        return False
    if not getattr(_debug_config, "debug", False):
        return False
    return torch.cuda.is_available()


def _cuda_snapshot_mb():
    """Return (allocated_mb, reserved_mb, max_allocated_mb, max_reserved_mb)."""
    return (
        torch.cuda.memory_allocated() / 2**20,
        torch.cuda.memory_reserved() / 2**20,
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
    )


def _format_cuda_diff(label: str, before: tuple, after: tuple) -> list:
    alloc_before, reserved_before, max_alloc_before, max_reserved_before = before
    alloc_after, reserved_after, max_alloc_after, max_reserved_after = after
    
    cache_before = reserved_before - alloc_before
    cache_after = reserved_after - alloc_after
    
    return [
        f"\n[DEBUG {label}] CUDA alloc: {alloc_after / 1024:.1f} GB | reserved: {reserved_after / 1024:.1f} GB | cache: {cache_after / 1024:.1f} GB",
        f"[DEBUG {label}] CUDA peaks: {max_alloc_after / 1024:.1f} GB | reserved: {max_reserved_after / 1024:.1f} GB",
        f"[DEBUG {label}] CUDA diff:  {(alloc_after - alloc_before) / 1024:.1f} GB | reserved: {(reserved_after - reserved_before) / 1024:.1f} GB | cache: {(cache_after - cache_before) / 1024:.1f} GB",
    ]


def _ram_snapshot_mb() -> float | None:
    """
    Return current process RSS (Working Set) in MB, or None if unavailable.
    Implemented only on Windows via GetProcessMemoryInfo (psapi).
    Unix/macOS: not implemented yet, returns None.
    """
    if sys.platform != "win32":
        # Unix/macOS support not implemented yet
        return None
    try:
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetCurrentProcess.argtypes = []
        GetCurrentProcess.restype = wintypes.HANDLE
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = GetCurrentProcess()
        pmc = _ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.WorkingSetSize / (2**20)
    except (AttributeError, OSError):
        return None


def _format_ram_diff(label: str, before_mb: float, after_mb: float) -> str:
    delta = before_mb - after_mb
    delta_str = f"(freed {delta:.1f} MB)" if delta >= 0 else f"(+{-delta:.1f} MB)"
    return f"[DEBUG {label}] RAM RSS: {before_mb:.1f} MB -> {after_mb:.1f} MB {delta_str}"


_GIB = 1024**3
# Payload attribute names of quantized tensor subclasses without __tensor_flatten__ (quanto).
_QUANTO_PAYLOAD_ATTRS = ("_data", "_scale")


def _cuda_storages(t: torch.Tensor) -> Iterator[Tuple[int, int]]:
    """Yield (data_ptr, nbytes) of every CUDA storage behind ``t``.

    Tensor subclasses (torchao ``__tensor_flatten__``, quanto ``_data``/``_scale``)
    are unwrapped to their payload tensors.
    """
    if hasattr(t, "__tensor_flatten__"):
        names, _ = t.__tensor_flatten__()
        inner = [getattr(t, n) for n in names]
    elif isinstance(getattr(t, "_data", None), torch.Tensor):
        inner = [getattr(t, a) for a in _QUANTO_PAYLOAD_ATTRS]
    else:
        inner = None
    if inner is not None:
        for x in inner:
            if isinstance(x, torch.Tensor):
                yield from _cuda_storages(x)
        return
    if t.is_cuda:
        st = t.untyped_storage()
        yield st.data_ptr(), st.nbytes()


def optimizer_state_tensors(optimizer) -> List[torch.Tensor]:
    """All tensors held in ``optimizer.state`` (nested dicts/lists/tuples)."""
    out: List[torch.Tensor] = []

    def walk(o) -> None:
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(optimizer.state)
    return out


class CudaMemoryLayout:
    """Attribute allocated CUDA memory to named owners and log step-over-step deltas.

    ``owners_fn`` returns ordered ``(label, tensors)``; the first owner of a
    storage claims it (so list LoRA params before the base model that may
    contain them). Remaining live tensors reachable from Python are grouped by
    (dtype, shape, requires_grad); ``untracked`` is allocated memory not
    reachable from Python (autograd saved tensors, C++-held buffers).
    Deltas are relative to the previous ``log(update_prev=True)`` call.
    """

    def __init__(self, owners_fn: Callable[[], List[Tuple[str, List[torch.Tensor]]]]):
        self._owners_fn = owners_fn
        self._prev: Optional[dict] = None

    def _census(self) -> dict:
        claimed: set = set()
        cats: Dict[str, List[int]] = {}
        for label, tensors in self._owners_fn():
            count = nbytes = 0
            for t in tensors:
                for ptr, nb in _cuda_storages(t):
                    if ptr not in claimed:
                        claimed.add(ptr)
                        count += 1
                        nbytes += nb
            cats[label] = [count, nbytes]
        others: Dict[tuple, List[int]] = {}
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            for ptr, nb in _cuda_storages(obj):
                if ptr in claimed:
                    continue
                claimed.add(ptr)
                group = others.setdefault(
                    (str(obj.dtype), tuple(obj.shape), bool(obj.requires_grad)), [0, 0]
                )
                group[0] += 1
                group[1] += nb
        tracked = sum(b for _, b in cats.values()) + sum(g[1] for g in others.values())
        return {
            "cats": cats,
            "others": others,
            "untracked": int(torch.cuda.memory_allocated()) - tracked,
        }

    def log(self, print_fn: Callable[[str], None], label: str, *, update_prev: bool = True) -> None:
        census = self._census()
        prev = self._prev
        print_fn(f"[DEBUG {label}] CUDA layout" + (" (Δ vs previous layout):" if prev else ":"))

        def row(name: str, count: int, nbytes: int, prev_bytes: Optional[int]) -> None:
            delta = "" if prev_bytes is None else f"  Δ{(nbytes - prev_bytes) / _GIB:+.3f}"
            print_fn(f"    {name:<26} n={count:<5} {nbytes / _GIB:8.3f} GiB{delta}")

        for name, (count, nbytes) in census["cats"].items():
            row(name, count, nbytes, prev["cats"][name][1] if prev else None)
        others = census["others"]
        prev_others = prev["others"] if prev else {}
        row(
            "other_live_tensors",
            sum(g[0] for g in others.values()),
            sum(g[1] for g in others.values()),
            sum(g[1] for g in prev_others.values()) if prev else None,
        )
        row("untracked(autograd/C++)", 0, census["untracked"], prev["untracked"] if prev else None)
        cache_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        print_fn(f"    {'cache_free(reserved-alloc)':<26} {'':8} {cache_free / _GIB:8.3f} GiB")

        def group_line(key: tuple, count: int, nbytes: int, delta: Optional[int]) -> str:
            dtype, shape, requires_grad = key
            extra = "" if delta is None else f" Δ{delta / _GIB:+.3f} GiB"
            return (
                f"      {dtype} {list(shape)} grad={requires_grad} "
                f"n={count} {nbytes / _GIB:.3f} GiB{extra}"
            )

        print_fn("    other_live_tensors top by size:")
        for key, (count, nbytes) in sorted(others.items(), key=lambda kv: -kv[1][1])[:6]:
            delta = nbytes - prev_others.get(key, [0, 0])[1] if prev else None
            print_fn(group_line(key, count, nbytes, delta))
        if prev:
            grown = sorted(
                ((k, g, g[1] - prev_others.get(k, [0, 0])[1]) for k, g in others.items()),
                key=lambda x: -x[2],
            )
            print_fn("    other_live_tensors top growth vs previous layout:")
            for key, (count, nbytes), delta in [x for x in grown if x[2] > 0][:6]:
                print_fn(group_line(key, count, nbytes, delta))
        if update_prev:
            self._prev = census


@contextlib.contextmanager
def memory_debug(
    print_fn: Callable[[str], None],
    label: str,
    kind: str = "all",
    verbose: bool = False,
    layout: Optional[CudaMemoryLayout] = None,
):
    """
    Context manager: measure memory around the block and log if debug is enabled.
    enabled is read from the config set via set_debug_config(); no need to pass it.
    kind="cuda": CUDA allocated/max only.
    kind="ram": process RSS (Windows only; on other platforms logs "not supported").
    kind="all": both CUDA (if available) and RAM.
    verbose: if True, also print torch.cuda.memory_summary() after measurements.
    layout: if given (and CUDA is measured), also log the per-owner CUDA layout.
    """
    if kind == "cuda":
        if not _is_enabled_for_cuda():
            yield
            return
        before_cuda = _cuda_snapshot_mb()
        before_ram = None
    elif kind == "ram":
        if not is_debug_enabled():
            yield
            return
        before_cuda = None
        before_ram = _ram_snapshot_mb()
    elif kind == "all":
        if not is_debug_enabled():
            yield
            return
        before_cuda = _cuda_snapshot_mb() if torch.cuda.is_available() else None
        before_ram = _ram_snapshot_mb()
    else:
        yield
        return

    try:
        yield
    finally:
        if before_cuda is not None:
            # torch.cuda.synchronize()
            after_cuda = _cuda_snapshot_mb()
            for line in _format_cuda_diff(label, before_cuda, after_cuda):
                print_fn(line)
            if layout is not None:
                layout.log(print_fn, label)
            if verbose:
                print_fn(f"[DEBUG {label}] Memory summary:")
                print_fn(torch.cuda.memory_summary(abbreviated=True))
        if kind in ("ram", "all"):
            after_ram = _ram_snapshot_mb()
            if after_ram is not None and before_ram is not None:
                print_fn(_format_ram_diff(label, before_ram, after_ram))
            else:
                print_fn(f"[DEBUG {label}] RAM: not supported on this platform yet")


def cuda_memory_debug(print_fn: Callable[[str], None], label: str, verbose: bool = False):
    """Alias for memory_debug(print_fn, label, kind="cuda")."""
    return memory_debug(print_fn, label, kind="cuda", verbose=verbose)
