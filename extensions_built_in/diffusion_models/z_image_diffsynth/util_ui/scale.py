# One-shot LoRA scale for a util-ui command. Does not update Stable-LoRA flags.

import json
from pathlib import Path

from extensions_built_in.diffusion_models.z_image_diffsynth.stable_lora import (
    setup_stable_lora,
    shrink_matrix,
)

GROW_ACTION = "grow_lora"
SHRINK_ACTION = "shrink_lora"
ACTION_FACTORS = {
    GROW_ACTION: 1.1,
    SHRINK_ACTION: 0.9,
}
ACTIONS_PATH = Path(__file__).with_name("actions.json")

_SKIP_LOGS = {
    "peft_dora": "[zimage_diffsynth] lora_scale skipped: peft_dora",
    "no_pairs": "[zimage_diffsynth] lora_scale skipped: no LoRA A/B pairs",
}


def load_actions() -> list:
    """Return the button list the Next route reads from ``actions.json``."""
    loaded = json.loads(ACTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(
            f"util-ui actions.json must be a list, got {type(loaded).__name__}"
        )
    return loaded


def _factor_text(factor: float) -> str:
    text = f"{factor:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def scale_log(pair_count: int, factor: float) -> str:
    return (
        f"[zimage_diffsynth] lora_scale factor={_factor_text(factor)} "
        f"pairs={pair_count}"
    )


def scale_lora(network, factor: float = ACTION_FACTORS[SHRINK_ACTION]) -> str:
    """Multiply every current pair's A and B by ``factor``. Return the log line.

    ``pair.stable`` on the pairs this call builds is left at its default.
    Caller-owned pair objects are not updated. ``stable_lora.pt`` and optimizer
    state are not read or written.
    """
    setup = setup_stable_lora(network)
    if setup.skip_reason is not None:
        try:
            return _SKIP_LOGS[setup.skip_reason]
        except KeyError as exc:
            raise ValueError(
                f"unknown util-ui skip reason: {setup.skip_reason}"
            ) from exc
    for pair in setup.pairs:
        shrink_matrix(pair.A, factor)
        shrink_matrix(pair.B, factor)
    return scale_log(len(setup.pairs), factor)
