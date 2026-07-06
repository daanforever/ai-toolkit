"""
Debug utilities. memory_debug context manager measures GPU/RAM around a block;
enabled via set_debug_config(config) with config.debug.
"""
import contextlib
import sys
from typing import Callable

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
        f"[DEBUG {label}] CUDA  diff: {alloc_after - alloc_before / 1024:.1f} GB | reserved: {reserved_after - reserved_before / 1024:.1f} GB | cache: {cache_after - cache_before / 1024:.1f} GB",
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


@contextlib.contextmanager
def memory_debug(
    print_fn: Callable[[str], None],
    label: str,
    kind: str = "all",
    verbose: bool = False,
):
    """
    Context manager: measure memory around the block and log if debug is enabled.
    enabled is read from the config set via set_debug_config(); no need to pass it.
    kind="cuda": CUDA allocated/max only.
    kind="ram": process RSS (Windows only; on other platforms logs "not supported").
    kind="all": both CUDA (if available) and RAM.
    verbose: if True, also print torch.cuda.memory_summary() after measurements.
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
