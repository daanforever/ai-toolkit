"""Block index parsing for kohya-style LoRA module names.

Row ids are ``layers_0``, ``noise_refiner_1``, or ``other`` when the name has no
block index. Longer block names are listed first so ``blocks`` does not take
``double_blocks``.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Longer names first so alternation does not steal a suffix (``blocks`` vs ``double_blocks``).
_BLOCK_NAMES = (
    "single_transformer_blocks",
    "transformer_blocks",
    "double_blocks",
    "single_blocks",
    "context_refiner",
    "noise_refiner",
    "layers",
    "blocks",
    "layer",
    "block",
)
_BLOCK_ALT = "|".join(_BLOCK_NAMES)
_BLOCK_US_RE = re.compile(rf"(?:^|[._])(?P<name>{_BLOCK_ALT})_(?P<idx>\d+)(?=_|$)")
_BLOCK_DOT_RE = re.compile(rf"(?:^|[._])(?P<name>{_BLOCK_ALT})\.(?P<idx>\d+)(?=\.|$)")
_ROW_SPLIT_RE = re.compile(r"(.+)_(\d+)$")


def parse_lora_row_col(module: str) -> Tuple[str, str]:
    """Return ``(row_id, column)`` for a LoRA module prefix."""
    found = None
    for regex in (_BLOCK_US_RE, _BLOCK_DOT_RE):
        match = regex.search(module)
        if match is None:
            continue
        if found is None or match.start() < found.start():
            found = match
    if found is None:
        column = module.replace(".", "_").strip("_") or "weight"
        return "other", column
    row = f"{found.group('name')}_{found.group('idx')}"
    rest = module[found.end() :].lstrip("._")
    column = rest.replace(".", "_").strip("_") or "weight"
    return row, column


def row_sort_key(row: str) -> Tuple[int, str, int]:
    """Sort indexed rows by block name then index. ``other`` is last."""
    match = _ROW_SPLIT_RE.fullmatch(row)
    if match is None:
        return (1, row, 0)
    return (0, match.group(1), int(match.group(2)))


def block_index(row: str) -> Optional[int]:
    """Numeric block index for a row id, or ``None`` for ``other``."""
    match = _ROW_SPLIT_RE.fullmatch(row)
    if match is None:
        return None
    return int(match.group(2))
