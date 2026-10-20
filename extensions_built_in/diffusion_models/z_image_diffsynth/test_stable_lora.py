"""Stable-LoRA shrink: pairs, precision, stop condition, and the pre-step hook."""

from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth.stable_lora import (
    STABILITY_CHECK_INTERVAL,
    StableLoraPair,
    apply_shrink,
    format_setup_log,
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


def _trainer(network, shrink):
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    kwargs = {} if shrink is None else {"stable_lora_shrink": shrink}
    trainer.network_config = SimpleNamespace(network_kwargs=kwargs)
    trainer.network = network
    return trainer


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
