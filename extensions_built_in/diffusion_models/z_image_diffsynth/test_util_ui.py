"""util-ui LoRA scale and RuntimeCommand drain."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.stable_lora import (
    setup_stable_lora,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
)
from extensions_built_in.diffusion_models.z_image_diffsynth import trainer as trainer_module
from extensions_built_in.diffusion_models.z_image_diffsynth.util_ui.queue import (
    drain_runtime_commands,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.util_ui.scale import (
    ACTION_FACTORS,
    ACTIONS_PATH,
    scale_lora,
)
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class _LoRA(torch.nn.Module):
    def __init__(self, full_rank=False, frozen=False):
        super().__init__()
        self.full_rank = full_rank
        self.lora_down = torch.nn.Linear(6, 2, bias=False)
        if full_rank:
            self.lora_up = torch.nn.Identity()
        else:
            self.lora_up = torch.nn.Linear(2, 4, bias=False)
        if frozen:
            self.lora_down.weight.requires_grad_(False)


def _network(modules, network_type="lora"):
    return SimpleNamespace(network_type=network_type, unet_loras=list(modules))


def _filled(a_value=2.0, b_value=4.0):
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(a_value)
        module.lora_up.weight.fill_(b_value)
    module.lora_down.weight.grad = torch.ones_like(module.lora_down.weight)
    module.lora_up.weight.grad = torch.full_like(module.lora_up.weight, 3.0)
    return module


def test_actions_json_is_grow_then_shrink():
    assert json.loads(ACTIONS_PATH.read_text(encoding="utf-8")) == [
        {
            "id": "grow_lora",
            "icon": "ArrowUp",
            "title": "Multiply LoRA A and B by 1.1",
        },
        {
            "id": "shrink_lora",
            "icon": "ArrowDown",
            "title": "Multiply LoRA A and B by 0.9",
        },
    ]


def test_fp32_scales_both_matrices_and_leaves_stable_flag_and_optimizer_state():
    module = _filled(0.1, 5.0)
    held = setup_stable_lora(_network([module]))
    held.pairs[0].stable = True
    optimizer_state = {"step": 3, "exp_avg": torch.tensor([1.25])}
    exp_avg = optimizer_state["exp_avg"].clone()

    line = scale_lora(_network([module]))

    assert held.pairs[0].stable is True
    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 0.1) * 0.9)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 5.0) * 0.9)
    assert torch.equal(module.lora_down.weight.grad, torch.ones_like(module.lora_down.weight))
    assert torch.equal(module.lora_up.weight.grad, torch.full_like(module.lora_up.weight, 3.0))
    assert optimizer_state["step"] == 3
    assert torch.equal(optimizer_state["exp_avg"], exp_avg)
    assert line == "[zimage_diffsynth] lora_scale factor=0.9 pairs=1"


def test_fp32_grow_scales_both_matrices_and_leaves_stable_flag():
    module = _filled(2.0, 4.0)
    held = setup_stable_lora(_network([module]))
    held.pairs[0].stable = True
    optimizer_state = {"step": 3, "exp_avg": torch.tensor([1.25])}
    exp_avg = optimizer_state["exp_avg"].clone()

    line = scale_lora(_network([module]), ACTION_FACTORS["grow_lora"])

    assert held.pairs[0].stable is True
    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 2.0) * 1.1)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 4.0) * 1.1)
    assert optimizer_state["step"] == 3
    assert torch.equal(optimizer_state["exp_avg"], exp_avg)
    assert line == "[zimage_diffsynth] lora_scale factor=1.1 pairs=1"


def test_frozen_and_full_rank_stay_put_while_a_live_pair_scales():
    live = _filled(2.0, 4.0)
    frozen = _filled(8.0, 8.0)
    frozen.lora_down.weight.requires_grad_(False)
    full = _LoRA(full_rank=True)
    with torch.no_grad():
        full.lora_down.weight.fill_(7.0)
    frozen_down = frozen.lora_down.weight.detach().clone()
    frozen_up = frozen.lora_up.weight.detach().clone()
    full_down = full.lora_down.weight.detach().clone()

    line = scale_lora(_network([live, frozen, full, torch.nn.Linear(2, 2)]))

    assert line == "[zimage_diffsynth] lora_scale factor=0.9 pairs=1"
    assert torch.equal(live.lora_down.weight, torch.full_like(live.lora_down.weight, 2.0) * 0.9)
    assert torch.equal(live.lora_up.weight, torch.full_like(live.lora_up.weight, 4.0) * 0.9)
    assert torch.equal(frozen.lora_down.weight, frozen_down)
    assert torch.equal(frozen.lora_up.weight, frozen_up)
    assert torch.equal(full.lora_down.weight, full_down)


def test_peft_dora_writes_nothing():
    module = _filled()
    before_down = module.lora_down.weight.detach().clone()
    before_up = module.lora_up.weight.detach().clone()

    line = scale_lora(_network([module], network_type="peft_dora"))

    assert line == "[zimage_diffsynth] lora_scale skipped: peft_dora"
    assert torch.equal(module.lora_down.weight, before_down)
    assert torch.equal(module.lora_up.weight, before_up)


def test_no_pairs_writes_nothing():
    module = _LoRA(full_rank=True)
    with torch.no_grad():
        module.lora_down.weight.fill_(6.0)
    before = module.lora_down.weight.detach().clone()

    line = scale_lora(_network([module]))

    assert line == "[zimage_diffsynth] lora_scale skipped: no LoRA A/B pairs"
    assert torch.equal(module.lora_down.weight, before)


def test_cpu_bf16_refuses_and_leaves_both_matrices():
    module = _LoRA()
    module.lora_down.weight = torch.nn.Parameter(module.lora_down.weight.detach().to(torch.bfloat16))
    module.lora_up.weight = torch.nn.Parameter(module.lora_up.weight.detach().to(torch.bfloat16))
    before_down = module.lora_down.weight.detach().clone()
    before_up = module.lora_up.weight.detach().clone()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        scale_lora(_network([module]))

    assert torch.equal(module.lora_down.weight, before_down)
    assert torch.equal(module.lora_up.weight, before_up)


def _create_db(path, actions):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE RuntimeCommand ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "jobId TEXT NOT NULL, "
        "action TEXT NOT NULL, "
        "createdAt TEXT)"
    )
    conn.executemany(
        "INSERT INTO RuntimeCommand (jobId, action) VALUES (?, ?)",
        [("job-1", action) for action in actions],
    )
    conn.commit()
    conn.close()
    return path


def _actions(path):
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT action FROM RuntimeCommand ORDER BY id").fetchall()
    conn.close()
    return [row[0] for row in rows]


def _ui_trainer(db_path, network, *, ui=True, main=True):
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    trainer.sqlite_db_path = db_path
    trainer.job_id = "job-1"
    trainer.network = network
    trainer.is_ui_trainer = ui
    trainer.accelerator = SimpleNamespace(is_main_process=main)
    trainer._db_connect = DiffusionTrainer._db_connect.__get__(trainer, ZImageDiffSynthTrainer)
    return trainer


def _capture_print(monkeypatch):
    lines = []

    def _record(*args, **_kwargs):
        lines.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.z_image_diffsynth.util_ui.queue.print_acc",
        _record,
    )
    return lines


def test_two_commands_scale_by_0_81_and_clear_the_queue(tmp_path, monkeypatch):
    module = _filled(10.0, 2.0)
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora", "shrink_lora"]))
    lines = _capture_print(monkeypatch)

    drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 10.0) * 0.9 * 0.9)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 2.0) * 0.9 * 0.9)
    assert _actions(db) == []
    assert lines == [
        "[zimage_diffsynth] lora_scale factor=0.9 pairs=1",
        "[zimage_diffsynth] lora_scale factor=0.9 pairs=1",
    ]


def test_grow_then_shrink_leaves_0_99(tmp_path, monkeypatch):
    module = _filled(10.0, 2.0)
    db = str(_create_db(tmp_path / "queue.db", ["grow_lora", "shrink_lora"]))
    lines = _capture_print(monkeypatch)

    drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 10.0) * 1.1 * 0.9)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 2.0) * 1.1 * 0.9)
    assert _actions(db) == []
    assert lines == [
        "[zimage_diffsynth] lora_scale factor=1.1 pairs=1",
        "[zimage_diffsynth] lora_scale factor=0.9 pairs=1",
    ]


def test_two_grows_leave_1_21(tmp_path):
    module = _filled(10.0, 2.0)
    db = str(_create_db(tmp_path / "queue.db", ["grow_lora", "grow_lora"]))

    drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 10.0) * 1.1 * 1.1)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 2.0) * 1.1 * 1.1)
    assert _actions(db) == []


def test_unknown_action_raises_and_keeps_that_row(tmp_path):
    module = _filled(10.0, 2.0)
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora", "other"]))

    with pytest.raises(ValueError, match="unknown util-ui action: other"):
        drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 10.0) * 0.9)
    assert _actions(db) == ["other"]


def test_empty_queue_logs_nothing(tmp_path, monkeypatch):
    module = _filled()
    before_down = module.lora_down.weight.detach().clone()
    before_up = module.lora_up.weight.detach().clone()
    db = str(_create_db(tmp_path / "queue.db", []))
    lines = _capture_print(monkeypatch)

    drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert lines == []
    assert torch.equal(module.lora_down.weight, before_down)
    assert torch.equal(module.lora_up.weight, before_up)
    assert _actions(db) == []


def test_missing_table_is_a_no_op(tmp_path):
    module = _filled()
    before = module.lora_down.weight.detach().clone()
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()

    drain_runtime_commands(_ui_trainer(db, _network([module])))

    assert torch.equal(module.lora_down.weight, before)


def test_non_ui_trainer_does_not_drain(tmp_path):
    module = _filled()
    before = module.lora_down.weight.detach().clone()
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora"]))

    drain_runtime_commands(_ui_trainer(db, _network([module]), ui=False))

    assert torch.equal(module.lora_down.weight, before)
    assert _actions(db) == ["shrink_lora"]


def test_non_main_process_does_not_drain(tmp_path):
    module = _filled()
    before = module.lora_down.weight.detach().clone()
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora"]))

    drain_runtime_commands(_ui_trainer(db, _network([module]), main=False))

    assert torch.equal(module.lora_down.weight, before)
    assert _actions(db) == ["shrink_lora"]


def test_peft_dora_command_is_consumed(tmp_path, monkeypatch):
    module = _filled()
    before_down = module.lora_down.weight.detach().clone()
    before_up = module.lora_up.weight.detach().clone()
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora"]))
    lines = _capture_print(monkeypatch)
    network = _network([module], network_type="peft_dora")

    drain_runtime_commands(_ui_trainer(db, network))

    assert lines == ["[zimage_diffsynth] lora_scale skipped: peft_dora"]
    assert torch.equal(module.lora_down.weight, before_down)
    assert torch.equal(module.lora_up.weight, before_up)
    assert _actions(db) == []


def test_end_step_hook_drains_after_super(tmp_path, monkeypatch):
    order = []

    def _super(self):
        order.append("super")

    monkeypatch.setattr(DiffusionTrainer, "end_step_hook", _super)
    real_drain = trainer_module.drain_runtime_commands

    def _drain(trainer):
        order.append("drain")
        real_drain(trainer)

    monkeypatch.setattr(trainer_module, "drain_runtime_commands", _drain)
    module = _filled(2.0, 3.0)
    db = str(_create_db(tmp_path / "queue.db", ["shrink_lora"]))
    trainer = _ui_trainer(db, _network([module]))

    ZImageDiffSynthTrainer.end_step_hook(trainer)

    assert order == ["super", "drain"]
    assert torch.equal(module.lora_down.weight, torch.full_like(module.lora_down.weight, 2.0) * 0.9)
    assert torch.equal(module.lora_up.weight, torch.full_like(module.lora_up.weight, 3.0) * 0.9)
    assert _actions(db) == []


def test_pre_step_hook_does_not_drain(monkeypatch):
    calls = []
    monkeypatch.setattr(trainer_module, "drain_runtime_commands", lambda trainer: calls.append(trainer))
    monkeypatch.setattr(SDTrainer, "hook_before_optimizer_step", lambda self: calls.append("super"))
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    trainer.network_config = SimpleNamespace(network_kwargs={"stable_lora_shrink": 0})
    trainer.network = _network([])

    ZImageDiffSynthTrainer.hook_before_optimizer_step(trainer)

    assert calls == ["super"]
