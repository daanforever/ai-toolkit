# Drain RuntimeCommand rows for the current job. One row is one scale.

import sqlite3

from toolkit.print import print_acc

from extensions_built_in.diffusion_models.z_image_diffsynth.util_ui.scale import (
    ACTION_FACTORS,
    scale_lora,
)


def _is_main_process(trainer) -> bool:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and not accelerator.is_main_process:
        return False
    return True


def _fetch_commands(trainer):
    """Return ``(id, action)`` rows, or None when the table does not exist."""
    try:
        with trainer._db_connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, action FROM RuntimeCommand WHERE jobId = ? ORDER BY id",
                (trainer.job_id,),
            )
            return cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise


def _delete_command(trainer, row_id: int) -> None:
    with trainer._db_connect() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RuntimeCommand WHERE id = ?", (row_id,))


def drain_runtime_commands(trainer) -> None:
    """Apply pending commands, then delete each applied id.

    An unknown action raises ``ValueError`` and that row stays. The connection
    is not held across ``scale_lora``.
    """
    if not getattr(trainer, "is_ui_trainer", False):
        return
    if not _is_main_process(trainer):
        return
    rows = _fetch_commands(trainer)
    if not rows:
        return
    for row_id, action in rows:
        try:
            factor = ACTION_FACTORS[action]
        except KeyError as exc:
            raise ValueError(f"unknown util-ui action: {action}") from exc
        print_acc(scale_lora(trainer.network, factor))
        _delete_command(trainer, row_id)
