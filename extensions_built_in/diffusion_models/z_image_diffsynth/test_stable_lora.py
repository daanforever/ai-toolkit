"""Stable-LoRA shrink: pairs, precision, stop condition, resume flags, and the pre-step hook."""

import os
import pickle
import tempfile
from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.stable_lora import (
    STABILITY_CHECK_INTERVAL,
    STABLE_LORA_STATE_FILENAME,
    StableLoraPair,
    StableStateRestore,
    apply_shrink,
    format_restore_ignored_log,
    format_restored_log,
    format_setup_log,
    restore_stable_state,
    save_stable_state,
    setup_stable_lora,
    should_check,
    shrink_percent_from_configs,
    validate_shrink_percent,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
    ZImageDiffSynthTrainer,
)
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.timer import Timer


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


class _PeftAdapter(torch.nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(6, 2, bias=False)}
        )
        self.layer.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 4, bias=False)}
        )
        if frozen:
            self.layer.lora_A["default"].weight.requires_grad_(False)


def _pair(a_value, b_value, dtype=torch.float32, device="cpu"):
    return StableLoraPair(
        A=torch.full((4, 4), a_value, dtype=dtype, device=device),
        B=torch.full((4, 4), b_value, dtype=dtype, device=device),
    )


def _network(modules, network_type="lora"):
    return SimpleNamespace(network_type=network_type, unet_loras=list(modules))


def test_setup_toolkit_and_peft_pairs_skips_full_rank_and_frozen():
    lora = _LoRA()
    full = _LoRA(full_rank=True)
    frozen = _LoRA(frozen=True)
    adapter = _PeftAdapter()
    unknown = torch.nn.Linear(2, 2)

    setup = setup_stable_lora(_network([lora, full, frozen, adapter, unknown]))

    assert setup.skip_reason is None
    assert len(setup.pairs) == 2
    assert setup.pairs[0].A is lora.lora_down.weight
    assert setup.pairs[0].B is lora.lora_up.weight
    assert setup.pairs[1].A is adapter.layer.lora_A["default"].weight
    assert setup.pairs[1].B is adapter.layer.lora_B["default"].weight
    assert setup.skipped_full_rank == 1
    assert setup.skipped_frozen == 1
    assert setup.skipped_unrecognized == 1
    assert "skipped_full_rank=1" in format_setup_log(1, setup)
    assert "pairs=2" in format_setup_log(1, setup)


def test_setup_peft_dora_is_skipped():
    lora = _LoRA()
    before = lora.lora_down.weight.detach().clone()
    setup = setup_stable_lora(_network([lora], network_type="peft_dora"))

    assert setup.skip_reason == "peft_dora"
    assert setup.pairs == []
    assert format_setup_log(1, setup).endswith("skipped: peft_dora")
    assert torch.equal(lora.lora_down.weight, before)


def test_setup_no_pairs_log():
    setup = setup_stable_lora(_network([_LoRA(full_rank=True)]))
    assert setup.skip_reason == "no_pairs"
    assert format_setup_log(1, setup).endswith("skipped: no LoRA A/B pairs")


def test_fp32_shrink_is_exact_and_leaves_b_and_grad():
    weight = torch.nn.Parameter(torch.full((4, 4), 2.0))
    weight.grad = torch.ones_like(weight)
    pair = StableLoraPair(A=weight, B=torch.zeros(4, 4))
    before_b = pair.B.clone()

    apply_shrink([pair], 1, check=True)

    assert torch.equal(weight, torch.full((4, 4), 2.0) * 0.99)
    assert torch.equal(pair.B, before_b)
    assert torch.equal(weight.grad, torch.ones_like(weight))
    assert pair.stable is False


def test_stop_condition_blocks_further_shrink():
    pair = _pair(1.0, 5.0)
    before = pair.A.clone()

    assert apply_shrink([pair], 1, check=True) == 1
    assert pair.stable is True
    assert torch.equal(pair.A, before)

    apply_shrink([pair], 1, check=True)
    assert torch.equal(pair.A, before)


def test_zero_percent_does_not_shrink_or_mark_stable():
    pair = _pair(4.0, 0.0)
    before = pair.A.clone()

    assert apply_shrink([pair], 0, check=True) == 0
    assert pair.stable is False
    assert torch.equal(pair.A, before)


def test_chunked_norm_matches_single_step_factor():
    pairs = [_pair(2.0, 0.0) for _ in range(STABILITY_CHECK_INTERVAL + 1)]
    apply_shrink(pairs, 1, check=True)
    expected = torch.full((4, 4), 2.0) * 0.99
    for pair in pairs:
        assert torch.equal(pair.A, expected)
        assert pair.stable is False


def test_cpu_bf16_refuses_stochastic_write():
    pair = _pair(1.0, 0.0, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        apply_shrink([pair], 1, check=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="copy_stochastic requires CUDA")
def test_bf16_shrink_changes_the_stored_weight():
    pair = _pair(1.0, 0.0, dtype=torch.bfloat16, device="cuda")
    before = pair.A.clone()
    before_b = pair.B.clone()

    apply_shrink([pair], 1, check=True)

    assert not torch.equal(pair.A, before)
    assert torch.equal(pair.B, before_b)
    assert pair.stable is False


@pytest.mark.parametrize(
    "value",
    [-0.1, 100, 100.0, True, False, "1", None, float("nan"), float("inf")],
)
def test_reject_invalid_percent(value):
    with pytest.raises(ValueError, match=r"stable_lora_shrink must be a number in \[0, 100\)"):
        validate_shrink_percent(value)


def test_absent_key_defaults_to_tenth_percent():
    assert shrink_percent_from_configs(SimpleNamespace(network_kwargs={})) == 0.1
    assert shrink_percent_from_configs(
        SimpleNamespace(network_kwargs={"stable_lora_shrink": 0}),
        SimpleNamespace(network_kwargs={"stable_lora_shrink": 2}),
    ) == 0.0


def test_check_cadence():
    assert should_check(1) is True
    assert should_check(2) is False
    assert should_check(STABILITY_CHECK_INTERVAL) is True
    assert should_check(STABILITY_CHECK_INTERVAL + 1) is False
    with pytest.raises(ValueError):
        should_check(0)


def _trainer(network, shrink, save_root=None):
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    kwargs = {} if shrink is None else {"stable_lora_shrink": shrink}
    trainer.network_config = SimpleNamespace(network_kwargs=kwargs)
    trainer.network = network
    if save_root is None:
        save_root = os.path.join(
            tempfile.gettempdir(), "ai-toolkit-stable-lora-absent"
        )
    trainer.save_root = save_root
    return trainer


def _capture_print_acc(monkeypatch):
    lines = []

    def _record(*args, **_kwargs):
        lines.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(
        "extensions_built_in.diffusion_models.z_image_diffsynth.trainer.print_acc",
        _record,
    )
    return lines


def _state_path(root):
    return os.path.join(root, STABLE_LORA_STATE_FILENAME)


def _stable_module():
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(0.1)
        module.lora_up.weight.fill_(5.0)
    return module


def _unstable_module():
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(2.0)
        module.lora_up.weight.fill_(0.0)
    return module


def test_trainer_hook_shrinks_until_stable_and_rebuilds_on_new_network():
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(2.0)
        module.lora_up.weight.fill_(0.0)
    up_before = module.lora_up.weight.detach().clone()
    trainer = _trainer(_network([module]), shrink=None)

    trainer.hook_before_optimizer_step()
    trainer.hook_before_optimizer_step()

    factor = 1.0 - shrink_percent_from_configs() / 100.0
    expected = torch.full((2, 6), 2.0).mul_(factor).mul_(factor)
    assert torch.equal(module.lora_down.weight, expected)
    assert torch.equal(module.lora_up.weight, up_before)
    assert trainer._stable_lora_step == 2

    stable_module = _LoRA()
    with torch.no_grad():
        stable_module.lora_down.weight.fill_(0.1)
        stable_module.lora_up.weight.fill_(5.0)
    down_before = stable_module.lora_down.weight.detach().clone()
    trainer.network = _network([stable_module])
    trainer.hook_before_optimizer_step()

    assert torch.equal(stable_module.lora_down.weight, down_before)
    assert trainer._stable_lora_setup.pairs[0].stable is True
    assert trainer._stable_lora_step == 1


def test_trainer_hook_zero_and_peft_dora_do_not_write():
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(2.0)
    before = module.lora_down.weight.detach().clone()

    disabled = _trainer(_network([module]), shrink=0)
    disabled.hook_before_optimizer_step()
    assert torch.equal(module.lora_down.weight, before)
    assert not hasattr(disabled, "_stable_lora_setup")

    dora = _trainer(_network([module], network_type="peft_dora"), shrink=1)
    dora.hook_before_optimizer_step()
    dora.hook_before_optimizer_step()
    assert torch.equal(module.lora_down.weight, before)
    assert dora._stable_lora_setup.skip_reason == "peft_dora"


def test_trainer_hook_rejects_out_of_range_percent():
    trainer = _trainer(_network([_LoRA()]), shrink=100)
    with pytest.raises(ValueError, match="stable_lora_shrink"):
        trainer.hook_before_optimizer_step()


def test_save_restore_round_trip(tmp_path):
    path = _state_path(tmp_path)
    saved = [_pair(1.0, 1.0) for _ in range(3)]
    saved[0].stable = True
    saved[2].stable = True

    save_stable_state(path, saved)

    assert os.path.isfile(path)
    assert list(tmp_path.glob(".stable_lora.*.tmp")) == []
    payload = torch.load(path, weights_only=True)
    assert payload["version"] == 1
    assert payload["stable"].dtype == torch.bool
    assert payload["stable"].tolist() == [True, False, True]

    fresh = [_pair(9.0, 9.0) for _ in range(3)]
    fresh[1].stable = True
    result = restore_stable_state(path, fresh)

    assert result == StableStateRestore(applied=True, stable_count=2, saved_length=3)
    assert [pair.stable for pair in fresh] == [True, False, True]


def test_length_mismatch_restores_nothing(tmp_path):
    path = _state_path(tmp_path)
    save_stable_state(path, [_pair(1.0, 1.0), _pair(1.0, 1.0)])
    current = [_pair(1.0, 1.0)]
    current[0].stable = True

    result = restore_stable_state(path, current)

    assert result == StableStateRestore(applied=False, stable_count=0, saved_length=2)
    assert current[0].stable is True


def test_corrupt_state_raises(tmp_path):
    path = _state_path(tmp_path)
    with open(path, "wb") as handle:
        handle.write(b"not-a-checkpoint")
    pair = _pair(1.0, 1.0)
    pair.stable = True

    with pytest.raises(pickle.UnpicklingError):
        restore_stable_state(path, [pair])

    assert pair.stable is True


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"version": 2, "stable": torch.tensor([True])}, "version must be 1"),
        ({"version": 1, "stable": torch.zeros(1)}, "1-D bool tensor"),
        ({"version": 1}, "1-D bool tensor"),
        ([True, False], "must be a dict"),
    ],
)
def test_invalid_state_payload_raises(tmp_path, payload, match):
    path = _state_path(tmp_path)
    torch.save(payload, path)
    pair = _pair(1.0, 1.0)
    pair.stable = True

    with pytest.raises(ValueError, match=match):
        restore_stable_state(path, [pair])

    assert pair.stable is True


def test_resume_restores_stable_and_still_shrinks_the_rest(tmp_path, monkeypatch):
    logs = _capture_print_acc(monkeypatch)
    stable_module = _stable_module()
    unstable_module = _unstable_module()
    first = _trainer(
        _network([stable_module, unstable_module]),
        shrink=10,
        save_root=tmp_path,
    )

    first.hook_before_optimizer_step()
    assert [pair.stable for pair in first._stable_lora_setup.pairs] == [True, False]
    first.post_save_hook(None)

    assert os.path.isfile(_state_path(tmp_path))
    assert list(tmp_path.glob(".stable_lora.*.tmp")) == []
    with torch.no_grad():
        stable_module.lora_down.weight.fill_(8.0)
    stable_before = stable_module.lora_down.weight.detach().clone()
    unstable_before = unstable_module.lora_down.weight.detach().clone()
    logs.clear()
    second = _trainer(
        _network([stable_module, unstable_module]),
        shrink=10,
        save_root=tmp_path,
    )
    second.hook_before_optimizer_step()

    assert format_restored_log(1, 2) in logs
    assert second._stable_lora_setup.pairs[0].stable is True
    assert second._stable_lora_setup.pairs[1].stable is False
    assert torch.equal(stable_module.lora_down.weight, stable_before)
    assert torch.equal(unstable_module.lora_down.weight, unstable_before * 0.9)


def test_changed_weights_do_not_restore(tmp_path, monkeypatch):
    logs = _capture_print_acc(monkeypatch)
    module = _stable_module()
    first = _trainer(_network([module]), shrink=10, save_root=tmp_path)
    first.hook_before_optimizer_step()
    first.post_save_hook(None)
    with torch.no_grad():
        module.lora_down.weight.fill_(8.0)
    before = module.lora_down.weight.detach().clone()
    network = _network([module])
    network.did_change_weights = True
    logs.clear()
    second = _trainer(network, shrink=10, save_root=tmp_path)

    second.hook_before_optimizer_step()

    assert format_restore_ignored_log("weights changed") in logs
    assert second._stable_lora_setup.pairs[0].stable is False
    assert torch.equal(module.lora_down.weight, before * 0.9)


def test_resume_ignores_pair_count_mismatch(tmp_path, monkeypatch):
    logs = _capture_print_acc(monkeypatch)
    first = _trainer(
        _network([_stable_module(), _stable_module()]),
        shrink=10,
        save_root=tmp_path,
    )
    first.hook_before_optimizer_step()
    first.post_save_hook(None)
    module = _unstable_module()
    before = module.lora_down.weight.detach().clone()
    logs.clear()
    second = _trainer(_network([module]), shrink=10, save_root=tmp_path)

    second.hook_before_optimizer_step()

    assert format_restore_ignored_log("pair count 2 != 1") in logs
    assert second._stable_lora_setup.pairs[0].stable is False
    assert torch.equal(module.lora_down.weight, before * 0.9)


def test_rebuild_does_not_read_state_again(tmp_path, monkeypatch):
    logs = _capture_print_acc(monkeypatch)
    trainer = _trainer(_network([_stable_module()]), shrink=10, save_root=tmp_path)
    trainer.hook_before_optimizer_step()
    trainer.post_save_hook(None)
    other = _unstable_module()
    before = other.lora_down.weight.detach().clone()
    logs.clear()
    trainer.network = _network([other])

    trainer.hook_before_optimizer_step()

    assert format_restored_log(1, 1) not in logs
    assert all("restore ignored" not in line for line in logs)
    assert trainer._stable_lora_setup.pairs[0].stable is False
    assert trainer._stable_lora_step == 1
    assert torch.equal(other.lora_down.weight, before * 0.9)


def test_disabled_runs_do_not_write_state(tmp_path):
    module = _LoRA()
    with torch.no_grad():
        module.lora_down.weight.fill_(2.0)
    before = module.lora_down.weight.detach().clone()

    disabled = _trainer(_network([module]), shrink=0, save_root=tmp_path)
    disabled.hook_before_optimizer_step()
    disabled.post_save_hook(None)

    dora = _trainer(
        _network([module], network_type="peft_dora"), shrink=1, save_root=tmp_path
    )
    dora.hook_before_optimizer_step()
    dora.post_save_hook(None)

    empty = _trainer(
        _network([_LoRA(full_rank=True)]), shrink=1, save_root=tmp_path
    )
    empty.hook_before_optimizer_step()
    empty.post_save_hook(None)

    assert torch.equal(module.lora_down.weight, before)
    assert not os.path.exists(_state_path(tmp_path))


class _Probe(SDTrainer):
    def __init__(self):
        self.events = []
        self.timer = Timer()
        self.optimizer = self._Optimizer(self.events)
        self.lr_scheduler = self._Scheduler(self.events)
        self.accelerator = SimpleNamespace(
            clip_grad_norm_=lambda *_args, **_kwargs: self.events.append("clip")
        )
        self._step_metrics = self._Metrics(self.events)
        self.train_config = SimpleNamespace(optimizer="adamw", max_grad_norm=1.0)
        self.params = [torch.nn.Parameter(torch.zeros(2))]
        self.sd = SimpleNamespace(is_multistage=False)
        self.is_grad_accumulation_step = False
        self.steps_this_boundary = 0
        self.adapter = None
        self.embedding = None
        self.ema = None

    def hook_before_optimizer_step(self):
        self.events.append("hook")

    def train_single_accumulation(self, batch, microbatch_scale=1.0):
        return torch.tensor(1.0)

    def end_of_training_loop(self):
        self.events.append("end")

    class _Optimizer:
        def __init__(self, events):
            self.events = events

        def zero_grad(self, set_to_none=False):
            self.events.append("zero_grad")

        def step(self):
            self.events.append("step")

    class _Scheduler:
        def __init__(self, events):
            self.events = events

        def step(self):
            self.events.append("scheduler")

    class _Metrics:
        def __init__(self, events):
            self.events = events

        def before_step(self, _optimizer):
            self.events.append("before_step")

        def after_step(self, _optimizer):
            self.events.append("after_step")


def test_hook_runs_once_between_clip_and_optimizer_step():
    probe = _Probe()
    batch = SimpleNamespace()

    probe.hook_train_loop(batch)

    assert probe.events.index("clip") < probe.events.index("hook")
    assert probe.events.index("hook") < probe.events.index("before_step")
    assert probe.events.index("before_step") < probe.events.index("step")
    assert probe.events.count("hook") == 1
    assert probe.events.count("step") == 1

    probe.is_grad_accumulation_step = True
    probe.events.clear()
    probe.hook_train_loop(batch)
    assert "hook" not in probe.events
    assert "step" not in probe.events


def test_base_hook_is_a_no_op():
    bare = SDTrainer.__new__(SDTrainer)
    assert SDTrainer.hook_before_optimizer_step(bare) is None
