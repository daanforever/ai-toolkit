"""Runtime linear_alpha for zimage_diffsynth LoRA networks."""

import sqlite3
import threading
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
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.lora_special import LoRASpecialNetwork


class DummyTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


class UNet2DConditionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv_out = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

        class LoRACompatibleLinear(torch.nn.Linear):
            pass

        class LoRACompatibleConv(torch.nn.Conv2d):
            pass

        self.block = torch.nn.Module()
        self.block.linear = LoRACompatibleLinear(4, 4)
        self.block.conv = LoRACompatibleConv(4, 4, kernel_size=1)


def _network(alpha: float, dim: int) -> LoRASpecialNetwork:
    return LoRASpecialNetwork(
        text_encoder=DummyTextEncoder(),
        unet=UNet2DConditionModel(),
        train_text_encoder=False,
        train_unet=True,
        lora_dim=dim,
        alpha=alpha,
        multiplier=1.0,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )


def _pair(alpha: float = 4.0, dim: int = 4):
    main = _network(alpha, dim)
    sampling = _network(alpha, dim)
    sampling.share_parameters_with(main)
    with torch.no_grad():
        for module in main.unet_loras:
            module.lora_down.weight.normal_()
            module.lora_up.weight.normal_()
    return main, sampling


def _create_db(path, value):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE RuntimeParams (jobId TEXT PRIMARY KEY, runtime_linear_alpha REAL)"
    )
    conn.execute(
        "INSERT INTO RuntimeParams (jobId, runtime_linear_alpha) VALUES (?, ?)",
        ("job-1", value),
    )
    conn.commit()
    conn.close()
    return str(path)


def _stored_alpha(path):
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT runtime_linear_alpha FROM RuntimeParams WHERE jobId = ?",
        ("job-1",),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return row[0]


def _trainer(db, network, *, ui=True, main=True, sampling=None, saved=None, sd_network=None, wait=None):
    trainer = ZImageDiffSynthTrainer.__new__(ZImageDiffSynthTrainer)
    trainer.sqlite_db_path = db
    trainer.job_id = "job-1"
    trainer.is_ui_trainer = ui
    trainer.network = network
    trainer.sd = SimpleNamespace(
        network=network if sd_network is None else sd_network,
        _sampling_network=sampling,
        _saved_train_network=saved,
    )
    accelerator = SimpleNamespace(is_main_process=main)
    if wait is not None:
        accelerator.wait_for_everyone = wait
    trainer.accelerator = accelerator
    trainer._db_connect = DiffusionTrainer._db_connect.__get__(trainer, ZImageDiffSynthTrainer)
    return trainer


def _weight_snapshot(network):
    return [
        (
            module.lora_down.weight.detach().clone(),
            module.lora_up.weight.detach().clone(),
        )
        for module in network.unet_loras
    ]


def _assert_weights(network, snapshot):
    assert len(network.unet_loras) == len(snapshot)
    for module, (down, up) in zip(network.unet_loras, snapshot):
        assert torch.equal(module.lora_down.weight, down)
        assert torch.equal(module.lora_up.weight, up)


def _inputs(module):
    if isinstance(module.lora_down, torch.nn.Linear):
        return torch.randn(2, module.lora_down.in_features)
    return torch.randn(2, module.lora_down.in_channels, 4, 4)


def _forward(module, inputs):
    with torch.no_grad():
        return module._call_forward(inputs)


def _silence_log(monkeypatch):
    monkeypatch.setattr(trainer_module, "print_acc", lambda *args, **kwargs: None)


def test_apply_rescales_shared_loras_and_clears_column(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    lines = []
    monkeypatch.setattr(
        trainer_module,
        "print_acc",
        lambda *args, **kwargs: lines.append(" ".join(str(arg) for arg in args)),
    )
    main, sampling = _pair()
    assert main.unet_loras[0].alpha is sampling.unet_loras[0].alpha
    assert main.unet_loras[0].scale == 1.0
    main_weights = _weight_snapshot(main)
    sampling_weights = _weight_snapshot(sampling)
    forwards = [(module, _inputs(module)) for module in main.unet_loras]
    before = [_forward(module, inputs) for module, inputs in forwards]
    assert all(tensor.abs().sum().item() > 0 for tensor in before)
    setup = setup_stable_lora(main)
    assert setup.pairs
    for pair in setup.pairs:
        pair.stable = True
    for module in main.unet_loras:
        module._deferred_lora_init_pending = True

    db = _create_db(tmp_path / "alpha.db", 8.0)
    trainer = _trainer(db, main, sampling=sampling)
    trainer.apply_runtime_linear_alpha()

    for network in (main, sampling):
        assert network.alpha == 8.0
        for module in network.unet_loras:
            assert float(module.alpha.item()) == 8.0
            assert module.scale == 8.0 / module.lora_dim
    assert main.unet_loras[0].alpha is sampling.unet_loras[0].alpha
    _assert_weights(main, main_weights)
    _assert_weights(sampling, sampling_weights)
    for (module, inputs), previous in zip(forwards, before):
        updated = _forward(module, inputs)
        assert torch.allclose(updated, previous * 2.0)
    for pair in setup.pairs:
        assert pair.stable is True
    assert all(module._deferred_lora_init_pending for module in main.unet_loras)
    assert _stored_alpha(db) is None
    assert any("runtime linear_alpha from UI/DB: 8.0" in line for line in lines)

    trainer.apply_runtime_linear_alpha()
    assert main.unet_loras[0].scale == 2.0
    _assert_weights(main, main_weights)


def test_turbo_swap_updates_saved_train_network(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    for module in sampling.unet_loras:
        module.scale = 0.25
    db = _create_db(tmp_path / "turbo.db", 8.0)
    trainer = _trainer(
        db,
        sampling,
        sampling=sampling,
        saved=main,
        sd_network=sampling,
    )
    trainer.apply_runtime_linear_alpha()
    for network in (main, sampling):
        assert network.alpha == 8.0
        for module in network.unet_loras:
            assert module.scale == 8.0 / module.lora_dim
            assert float(module.alpha.item()) == 8.0


def test_wrapped_network_is_unwrapped_once(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    seen = []
    original = ZImageDiffSynthTrainer._linear_alpha_networks

    def tracked(self):
        found = original(self)
        seen.extend(found)
        return found

    monkeypatch.setattr(ZImageDiffSynthTrainer, "_linear_alpha_networks", tracked)
    db = _create_db(tmp_path / "wrap.db", 8.0)
    trainer = _trainer(
        db,
        main,
        sampling=sampling,
        sd_network=SimpleNamespace(module=main),
        saved=SimpleNamespace(module=main),
    )
    trainer.apply_runtime_linear_alpha()
    assert seen == [main, sampling]


def test_invalid_values_raise_and_keep_the_column(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    weights = _weight_snapshot(main)

    db = _create_db(tmp_path / "zero.db", 0.0)
    trainer = _trainer(db, main, sampling=sampling)
    with pytest.raises(ValueError, match="finite number > 0"):
        trainer.apply_runtime_linear_alpha()
    assert _stored_alpha(db) == 0.0
    assert main.unet_loras[0].scale == 1.0
    _assert_weights(main, weights)

    db = _create_db(tmp_path / "neg.db", -2.0)
    trainer = _trainer(db, main, sampling=sampling)
    with pytest.raises(ValueError, match="finite number > 0"):
        trainer.apply_runtime_linear_alpha()
    assert _stored_alpha(db) == -2.0

    trainer = _trainer(db, main, sampling=sampling)
    trainer._read_runtime_linear_alpha = lambda: float("nan")
    with pytest.raises(ValueError, match="finite number > 0"):
        trainer.apply_runtime_linear_alpha()
    _assert_weights(main, weights)


def test_non_lora_raises_without_clearing_or_writing(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    main.network_type = "lycoris"
    sampling.network_type = "lycoris"
    weights = _weight_snapshot(main)
    db = _create_db(tmp_path / "lycoris.db", 8.0)
    trainer = _trainer(db, main, sampling=sampling)
    with pytest.raises(ValueError, match="network.type lora"):
        trainer.apply_runtime_linear_alpha()
    assert _stored_alpha(db) == 8.0
    assert main.alpha == 4.0
    assert float(main.unet_loras[0].alpha.item()) == 4.0
    _assert_weights(main, weights)


def test_missing_row_table_and_network_are_safe(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    trainer = _trainer(str(empty), main, sampling=sampling)
    trainer.apply_runtime_linear_alpha()
    assert main.unet_loras[0].scale == 1.0

    db = tmp_path / "norow.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE RuntimeParams (jobId TEXT PRIMARY KEY, runtime_linear_alpha REAL)"
    )
    conn.commit()
    conn.close()
    trainer = _trainer(str(db), main, sampling=sampling)
    trainer.apply_runtime_linear_alpha()
    assert main.unet_loras[0].scale == 1.0

    db = _create_db(tmp_path / "deferred.db", 8.0)
    trainer = _trainer(db, None)
    with pytest.raises(ValueError, match="no network"):
        trainer.apply_runtime_linear_alpha()
    assert _stored_alpha(db) == 8.0

    trainer = _trainer(db, main, sampling=sampling, ui=False)
    trainer.apply_runtime_linear_alpha()
    assert _stored_alpha(db) == 8.0
    assert main.unet_loras[0].scale == 1.0


def test_non_main_rank_applies_and_leaves_the_column(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    main, sampling = _pair()
    db = _create_db(tmp_path / "rank.db", 8.0)
    trainer = _trainer(db, main, sampling=sampling, main=False)
    trainer.apply_runtime_linear_alpha()
    assert main.unet_loras[0].scale == 2.0
    assert sampling.unet_loras[0].scale == 2.0
    assert _stored_alpha(db) == 8.0


def test_both_ranks_read_before_clear(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    rank0_main, rank0_sampling = _pair()
    rank1_main, rank1_sampling = _pair()
    db = _create_db(tmp_path / "ranks.db", 8.0)
    barrier = threading.Barrier(2)
    seen = []
    errors = []

    def wait():
        seen.append(_stored_alpha(db))
        barrier.wait(timeout=10)

    def run(rank, main, sampling):
        try:
            trainer = _trainer(
                db,
                main,
                sampling=sampling,
                main=rank == 0,
                wait=wait,
            )
            trainer.apply_runtime_linear_alpha()
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(0, rank0_main, rank0_sampling)),
        threading.Thread(target=run, args=(1, rank1_main, rank1_sampling)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()
    assert errors == []
    assert seen == [8.0, 8.0]
    assert _stored_alpha(db) is None
    for network in (rank0_main, rank0_sampling, rank1_main, rank1_sampling):
        assert network.alpha == 8.0
        for module in network.unet_loras:
            assert module.scale == 8.0 / module.lora_dim
            assert float(module.alpha.item()) == 8.0


def test_end_step_hook_applies_before_drain(tmp_path, monkeypatch):
    _silence_log(monkeypatch)
    order = []
    monkeypatch.setattr(DiffusionTrainer, "end_step_hook", lambda self: order.append("super"))
    monkeypatch.setattr(
        trainer_module,
        "drain_runtime_commands",
        lambda trainer: order.append("drain"),
    )
    main, sampling = _pair()
    db = _create_db(tmp_path / "hook.db", 8.0)
    trainer = _trainer(db, main, sampling=sampling)
    ZImageDiffSynthTrainer.end_step_hook(trainer)
    assert order == ["super", "drain"]
    assert main.unet_loras[0].scale == 2.0
    assert _stored_alpha(db) is None


def test_load_state_dict_overwrites_alpha_not_scale():
    main, sampling = _pair(alpha=4.0, dim=4)
    state = {key: value.detach().clone() for key, value in main.state_dict().items()}
    for key in state:
        if key.endswith(".alpha"):
            state[key] = torch.tensor(16.0)
    main.load_state_dict(state, strict=False)
    assert float(main.unet_loras[0].alpha.item()) == 16.0
    assert float(sampling.unet_loras[0].alpha.item()) == 16.0
    assert main.unet_loras[0].scale == 1.0
    assert sampling.unet_loras[0].scale == 1.0


def test_load_weights_reapplies_config_alpha(monkeypatch):
    main, sampling = _pair(alpha=4.0, dim=4)
    for module in sampling.unet_loras:
        module.scale = 99.0
    state = {key: value.detach().clone() for key, value in main.state_dict().items()}
    alpha_keys = [key for key in state if key.endswith(".alpha")]
    assert alpha_keys
    for key in alpha_keys:
        state[key] = torch.tensor(16.0)
    weights = _weight_snapshot(main)

    def super_load(self, path):
        assert path == "ckpt.safetensors"
        self.network.load_state_dict(state, strict=False)

    monkeypatch.setattr(DiffusionTrainer, "load_weights", super_load)
    trainer = _trainer("unused", main, sampling=sampling, ui=False)
    trainer.network_config = SimpleNamespace(linear_alpha=4.0)
    ZImageDiffSynthTrainer.load_weights(trainer, "ckpt.safetensors")

    assert main.unet_loras[0].alpha is sampling.unet_loras[0].alpha
    for network in (main, sampling):
        assert network.alpha == 4.0
        for module in network.unet_loras:
            assert float(module.alpha.item()) == 4.0
            assert module.scale == 1.0
    _assert_weights(main, weights)


def test_load_weights_leaves_non_lora_and_zero_alpha(monkeypatch):
    main, sampling = _pair(alpha=4.0, dim=4)
    state = {key: value.detach().clone() for key, value in main.state_dict().items()}
    for key in state:
        if key.endswith(".alpha"):
            state[key] = torch.tensor(16.0)

    def super_load(self, path):
        self.network.load_state_dict(state, strict=False)

    monkeypatch.setattr(DiffusionTrainer, "load_weights", super_load)
    main.network_type = "peft"
    trainer = _trainer("unused", main, sampling=sampling, ui=False)
    trainer.network_config = SimpleNamespace(linear_alpha=4.0)
    ZImageDiffSynthTrainer.load_weights(trainer, "ckpt.safetensors")
    assert float(main.unet_loras[0].alpha.item()) == 16.0
    assert main.unet_loras[0].scale == 1.0
    assert sampling.unet_loras[0].scale == 1.0

    main.network_type = "lora"
    for module in main.unet_loras:
        module.alpha.fill_(4.0)
    trainer.network_config = SimpleNamespace(linear_alpha=0)
    ZImageDiffSynthTrainer.load_weights(trainer, "ckpt.safetensors")
    assert float(main.unet_loras[0].alpha.item()) == 16.0
    assert main.unet_loras[0].scale == 1.0
