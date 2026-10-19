"""Per-module LoRA weight heatmap from a kohya-style ``.safetensors`` file.

Supported tensors, per module prefix:

- ``{prefix}.lora_down.weight`` shape ``(rank, in)``
- ``{prefix}.lora_up.weight`` shape ``(out, rank)``
- ``{prefix}.alpha`` optional scalar

``dW = scale * up @ down`` with ``scale = alpha / rank``. A missing or zero
alpha uses ``scale = 1``, matching kohya (alpha 0 means alpha = rank).

Metrics, computed in fp32 from ``rank x rank`` Gram matrices so ``dW`` is never
materialized:

- ``frobenius``: ``||dW||_F``
- ``balance``: ``||up||_F / ||down||_F``
- ``stable_rank``: ``||dW||_F^2 / ||dW||_2^2``
- ``effective_rank``: ``exp(entropy(singular values))``

A zero down or up matrix makes the ratio and both ranks undefined; those cells
are ``null`` and ``frobenius`` is ``0``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import SafetensorError, safe_open

DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"

METRIC_FROBENIUS = "frobenius"
METRIC_BALANCE = "balance"
METRIC_STABLE_RANK = "stable_rank"
METRIC_EFFECTIVE_RANK = "effective_rank"
METRIC_NAMES = (
    METRIC_FROBENIUS,
    METRIC_BALANCE,
    METRIC_STABLE_RANK,
    METRIC_EFFECTIVE_RANK,
)

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

_NORM_EPS = 1e-12
_FRO2_ROUNDTRIP = 1e-5

Cell = Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]


class UnsupportedLoRAFormat(ValueError):
    """Checkpoint is not a kohya-style ``lora_down`` / ``lora_up`` LoRA."""


def _kind(key: str) -> str:
    if key.endswith(DOWN_SUFFIX):
        return "down"
    if key.endswith(UP_SUFFIX):
        return "up"
    if key.endswith(ALPHA_SUFFIX):
        return "alpha"
    if ".lora_A" in key or key.endswith(".lora_A") or ".lora_B" in key or key.endswith(".lora_B"):
        return "peft"
    tail = key.rsplit(".", 1)[-1]
    if tail.startswith(("lokr_", "loha_")) or ".lokr_" in key or ".loha_" in key or "hada_" in key:
        return "lycoris"
    return "other"


def _reject_unsupported(keys: Sequence[str]) -> None:
    peft_key = None
    lycoris_key = None
    has_down = False
    for key in keys:
        kind = _kind(key)
        if kind == "down":
            has_down = True
        elif kind == "peft" and peft_key is None:
            peft_key = key
        elif kind == "lycoris" and lycoris_key is None:
            lycoris_key = key
    if lycoris_key is not None:
        raise UnsupportedLoRAFormat(
            f"LyCORIS/LoKr weights are not supported (found {lycoris_key}). Expected lora_down/lora_up."
        )
    if peft_key is not None:
        raise UnsupportedLoRAFormat(
            f"PEFT lora_A/lora_B weights are not supported (found {peft_key}). Expected lora_down/lora_up."
        )
    if not has_down:
        raise UnsupportedLoRAFormat(
            "No lora_down.weight tensors found. Expected a kohya-style LoRA safetensors file."
        )


def _parse_row_col(module: str) -> Tuple[str, str]:
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


def _row_sort_key(row: str) -> Tuple[int, str, int]:
    match = _ROW_SPLIT_RE.fullmatch(row)
    if match is None:
        return (1, row, 0)
    return (0, match.group(1), int(match.group(2)))


def _scale(alpha: Optional[torch.Tensor], rank: int) -> float:
    if alpha is None:
        return 1.0
    if alpha.ndim != 0 and alpha.numel() != 1:
        raise ValueError(f"alpha must be a scalar, got shape {tuple(alpha.shape)}")
    value = float(alpha.detach().to(dtype=torch.float32, device="cpu").reshape(()).item())
    if not math.isfinite(value):
        raise ValueError(f"alpha is not finite: {value}")
    if value == 0.0:
        return 1.0
    return value / float(rank)


def _singular_values(down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Singular values of ``up @ down`` without building the ``out x in`` matrix.

    When both factors are at least ``rank``-wide, a reduced QR reduces the product
    to a ``rank x rank`` core. Smaller factors fall back to the product itself,
    which is then smaller than ``rank`` on one side.
    """
    rank, in_features = down.shape
    out_features, up_rank = up.shape
    if up_rank != rank:
        raise ValueError(f"rank mismatch: down {tuple(down.shape)} up {tuple(up.shape)}")
    if in_features >= rank and out_features >= rank:
        _q_down, r_down = torch.linalg.qr(down.transpose(0, 1), mode="reduced")
        _q_up, r_up = torch.linalg.qr(up, mode="reduced")
        return torch.linalg.svdvals(r_up @ r_down.transpose(0, 1))
    return torch.linalg.svdvals(up @ down)


def _ranks(sigmas: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    finite = torch.isfinite(sigmas)
    if not bool(torch.all(finite)):
        raise ValueError("singular values are not finite")
    clipped = sigmas.clamp(min=0)
    sigma_max = clipped.max()
    if float(sigma_max) <= _NORM_EPS:
        return None, None
    stable = float(torch.sum(clipped * clipped) / (sigma_max * sigma_max))
    total = torch.sum(clipped)
    probs = clipped / total
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    effective = float(torch.exp(entropy))
    return stable, effective


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value}")
    return value


def _module_metrics(
    down: torch.Tensor,
    up: torch.Tensor,
    alpha: Optional[torch.Tensor],
) -> Cell:
    if down.ndim != 2 or up.ndim != 2:
        raise UnsupportedLoRAFormat(
            f"non-matrix LoRA weights down={tuple(down.shape)} up={tuple(up.shape)}. Conv LoRA is not supported."
        )
    matrix_down = down.detach().to(dtype=torch.float32, device="cpu")
    matrix_up = up.detach().to(dtype=torch.float32, device="cpu")
    if matrix_down.shape[0] != matrix_up.shape[1]:
        raise ValueError(
            f"rank mismatch: down {tuple(matrix_down.shape)} up {tuple(matrix_up.shape)}"
        )
    rank = int(matrix_down.shape[0])
    if rank < 1:
        raise ValueError(f"invalid LoRA rank {rank}")
    scale = _scale(alpha, rank)
    gram_down = matrix_down @ matrix_down.transpose(0, 1)
    gram_up = matrix_up.transpose(0, 1) @ matrix_up
    fro2 = float(torch.sum(gram_up * gram_down.transpose(0, 1)))
    reference = float(torch.trace(gram_down) + torch.trace(gram_up)) + 1.0
    if fro2 < 0:
        if abs(fro2) <= _FRO2_ROUNDTRIP * reference:
            fro2 = 0.0
        else:
            raise ValueError(f"negative Frobenius energy {fro2}")
    frobenius = _finite(scale * math.sqrt(fro2), METRIC_FROBENIUS)
    norm_down = float(torch.sqrt(torch.clamp(torch.trace(gram_down), min=0)))
    norm_up = float(torch.sqrt(torch.clamp(torch.trace(gram_up), min=0)))
    if norm_down <= _NORM_EPS or norm_up <= _NORM_EPS:
        return frobenius, None, None, None
    balance = _finite(norm_up / norm_down, METRIC_BALANCE)
    stable, effective = _ranks(_singular_values(matrix_down, matrix_up))
    if stable is not None:
        stable = _finite(stable, METRIC_STABLE_RANK)
    if effective is not None:
        effective = _finite(effective, METRIC_EFFECTIVE_RANK)
    return frobenius, balance, stable, effective


def _training_step(metadata: Optional[dict]) -> Tuple[Optional[int], Optional[int]]:
    if not metadata:
        return None, None
    raw = metadata.get("training_info")
    if raw is None or raw == "":
        return None, None
    info = json.loads(raw)
    if not isinstance(info, dict):
        raise ValueError("training_info metadata is not an object")
    step = info.get("step")
    epoch = info.get("epoch")
    return (None if step is None else int(step), None if epoch is None else int(epoch))


def _suffix_module(key: str, suffix: str) -> str:
    if not key.endswith(suffix):
        raise ValueError(f"key {key} does not end with {suffix}")
    module = key[: -len(suffix)]
    if module == "":
        raise ValueError(f"empty module prefix in {key}")
    return module


def compute_lora_heatmap(path: str) -> dict:
    """Read ``path`` and return rows, columns, metrics, and file metadata.

    The process stays on CPU. ``rows`` are block ids (``layers_0``, ``noise_refiner_1``,
    or ``other`` when the key has no block index). ``cols`` are the module names
    under that block. Missing pairs are JSON ``null``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        _reject_unsupported(keys)
        metadata = handle.metadata()
        for key in keys:
            kind = _kind(key)
            if kind not in ("down", "up", "alpha"):
                continue
            suffix = {"down": DOWN_SUFFIX, "up": UP_SUFFIX, "alpha": ALPHA_SUFFIX}[kind]
            module = _suffix_module(key, suffix)
            bucket = grouped.setdefault(module, {})
            if kind in bucket:
                raise ValueError(f"duplicate {kind} tensor for {module}")
            bucket[kind] = handle.get_tensor(key)
        step, epoch = _training_step(metadata)

    cells: Dict[Tuple[str, str], Cell] = {}
    for module, parts in grouped.items():
        if "down" not in parts or "up" not in parts:
            raise ValueError(f"LoRA module {module} is missing lora_down or lora_up")
        metrics = _module_metrics(parts["down"], parts["up"], parts.get("alpha"))
        row, col = _parse_row_col(module)
        if (row, col) in cells:
            raise ValueError(f"duplicate heatmap cell {row} / {col} from {module}")
        cells[(row, col)] = metrics

    if not cells:
        raise UnsupportedLoRAFormat("No LoRA modules could be scored.")

    rows = sorted({row for row, _col in cells}, key=_row_sort_key)
    cols = sorted({col for _row, col in cells})
    row_index = {row: index for index, row in enumerate(rows)}
    col_index = {col: index for index, col in enumerate(cols)}
    blank: List[List[Optional[float]]] = [
        [None for _col in cols] for _row in rows
    ]
    matrices = {name: [list(row) for row in blank] for name in METRIC_NAMES}
    for (row, col), metrics in cells.items():
        i = row_index[row]
        j = col_index[col]
        for name, value in zip(METRIC_NAMES, metrics):
            matrices[name][i][j] = value

    return {
        "file": {
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "mtime_ms": int(os.path.getmtime(path) * 1000),
            "step": step,
            "epoch": epoch,
        },
        "rows": rows,
        "cols": cols,
        "metrics": matrices,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python toolkit/lora_heatmap.py <file.safetensors>", file=sys.stderr)
        return 2
    try:
        payload = compute_lora_heatmap(args[0])
    except (UnsupportedLoRAFormat, ValueError, FileNotFoundError, OSError, SafetensorError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
