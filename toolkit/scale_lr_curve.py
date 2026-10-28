"""Gaussian Scale-LR-by-index curve from a kohya-style LoRA file.

Layer indices come from LoRA key names. ``N = max_index + 1``. Weights are
``Adafactor._compute_gaussian_lr_weights`` so the chart matches the optimizer.
A non-empty mask sets ``w = 1`` on indices whose row id matches no substring,
the same rule as ``Adafactor._scale_lr_applies`` (empty mask scales every index).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors import SafetensorError, safe_open

from toolkit.lora_block_index import block_index, parse_lora_row_col

DOWN_SUFFIX = ".lora_down.weight"


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
    try:
        step_i = None if step is None else int(step)
        epoch_i = None if epoch is None else int(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training_info step/epoch is not an integer: {exc}") from exc
    return step_i, epoch_i


def _indexed_rows(path: str) -> Tuple[Dict[int, Set[str]], dict]:
    """Map block index to row ids, plus file metadata for the chart caption."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    rows_by_index: Dict[int, Set[str]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        has_down = False
        for key in keys:
            if not key.endswith(DOWN_SUFFIX):
                continue
            has_down = True
            module = key[: -len(DOWN_SUFFIX)]
            if module == "":
                raise ValueError(f"empty module prefix in {key}")
            row, _column = parse_lora_row_col(module)
            index = block_index(row)
            if index is None:
                continue
            rows_by_index.setdefault(index, set()).add(row)
        step, epoch = _training_step(metadata)

    if not has_down:
        raise ValueError(
            "No lora_down.weight tensors found. Expected a kohya-style LoRA safetensors file."
        )
    file_info = {
        "filename": os.path.basename(path),
        "path": os.path.abspath(path),
        "mtime_ms": int(os.path.getmtime(path) * 1000),
        "step": step,
        "epoch": epoch,
    }
    return rows_by_index, file_info


def _apply_mask(
    weights: List[float],
    rows_by_index: Dict[int, Set[str]],
    mask: Optional[Tuple[str, ...]],
) -> List[float]:
    """Force ``w = 1`` where no row id at that index contains a mask substring."""
    if not mask:
        return weights
    out: List[float] = []
    for index, weight in enumerate(weights):
        rows = rows_by_index.get(index, ())
        if any(substr in row for row in rows for substr in mask):
            out.append(weight)
        else:
            out.append(1.0)
    return out


def compute_scale_lr_curve(
    path: str,
    mean: float,
    std: float,
    mask: Optional[Sequence[str]] = None,
) -> dict:
    """Return ``indices``, ``weights``, ``n``, and file metadata for ``path``."""
    from toolkit.optimizers.adafactor import Adafactor

    mean_f, std_f = Adafactor._validate_scale_lr_mean_std(mean, std, required=True)
    mask_key = Adafactor._validate_scale_lr_mask(mask)
    if mean_f is None or std_f is None:
        raise ValueError(
            "scale_lr_by_index=True requires finite scale_lr_mean and scale_lr_std > 0"
        )

    rows_by_index, file_info = _indexed_rows(path)
    if not rows_by_index:
        raise ValueError("No layer indices found in LoRA keys.")
    max_index = max(rows_by_index)
    if max_index <= 0:
        raise ValueError(
            f"scale_lr_by_index requires max_index > 0, got max_index={max_index}"
        )

    n = max_index + 1
    pdf = Adafactor._compute_gaussian_lr_weights(n, float(mean_f), float(std_f))
    weights = _apply_mask(
        [float(pdf[i].item()) for i in range(n)],
        rows_by_index,
        mask_key,
    )
    return {
        "indices": list(range(n)),
        "weights": weights,
        "n": n,
        "file": file_info,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python toolkit/scale_lr_curve.py")
    parser.add_argument("file")
    parser.add_argument("--mean", required=True, type=float)
    parser.add_argument("--std", required=True, type=float)
    parser.add_argument("--mask", action="append", default=None)
    try:
        parsed = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    try:
        payload = compute_scale_lr_curve(parsed.file, parsed.mean, parsed.std, parsed.mask)
    except (ValueError, FileNotFoundError, OSError, SafetensorError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
