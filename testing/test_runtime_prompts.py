"""Unit tests for runtime sample prompt apply / recache."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.unloader import FakeTextEncoder, unload_text_encoder
from toolkit.util.device import devices_equal

_HAS_CUDA = torch.cuda.is_available()
_CUDA = torch.device("cuda") if _HAS_CUDA else None
_CPU = torch.device("cpu")


def _conn_returning(value):
    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return value

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    return _Conn()


class _TinyLinear(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


class _TinyLoRANetwork(nn.Module):
    def __init__(self, n: int = 4, rank: int = 2):
        super().__init__()
        self.lora_A = nn.Linear(n, rank, bias=False)
        self.lora_B = nn.Linear(rank, n, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.unet_loras = [self.lora_A, self.lora_B]

    def force_to(self, device, dtype):
        self.to(device)
        for lora in self.unet_loras:
            lora.to(device, dtype)


def _any_on(module: nn.Module, device: torch.device) -> bool:
    for p in module.parameters():
        if devices_equal(p.device, device):
            return True
    for b in module.buffers():
        if b is not None and devices_equal(b.device, device):
            return True
    return False


def _make_trainer(
    *,
    prompts_db,
    sample_prompts,
    caching=False,
    unload_te=False,
):
    """Build a DiffusionTrainer-like object without running __init__."""
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "job-test"
    t.device_torch = "cpu"
    t._last_applied_runtime_prompts = None
    t.is_caching_text_embeddings = caching
    t.train_config = SimpleNamespace(unload_text_encoder=unload_te)
    samples = [SimpleNamespace(prompt=p) for p in sample_prompts]
    t.sample_config = SimpleNamespace(samples=samples)
    t.sd = SimpleNamespace(
        unet=MagicMock(),
        text_encoder_to=MagicMock(),
        _sampling_transformer=MagicMock(),
    )
    t.cache_sample_prompts = MagicMock()
    t._recache_sample_prompts_runtime = MagicMock()

    def _get():
        return prompts_db

    t.get_runtime_prompts = _get
    return t


def _make_recache_trainer(
    *,
    caching: bool,
    unload_te: bool = False,
    train_device: Optional[torch.device] = None,
    train_on_turbo: bool = False,
    start_with_fake_te: bool = True,
) -> Any:
    """Tiny real-module fixture for behavioral runtime recache sequence tests."""
    device = train_device if train_device is not None else _CPU
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "job-recache"
    t.device_torch = device
    t.is_caching_text_embeddings = caching
    t.train_config = SimpleNamespace(unload_text_encoder=unload_te)
    t.sample_config = SimpleNamespace(samples=[SimpleNamespace(prompt="old")])
    t._last_applied_runtime_prompts = None
    t.cache_sample_prompts = MagicMock()

    te = _TinyLinear(2).to(device if not start_with_fake_te else _CPU)
    unet = _TinyLinear(2).to(device)
    network = _TinyLoRANetwork(2).to(device)
    sampling = _TinyLinear(2).to(_CPU if not train_on_turbo else device)

    def text_encoder_to(dev, *a, **k):
        live = t.sd.text_encoder
        if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
            live.to(dev)
        real = getattr(t.sd, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            real.to(dev)

    turbo_calls: List[bool] = []

    def apply_turbo_teacher_mode(enabled: bool):
        turbo_calls.append(bool(enabled))
        if enabled:
            unet.to(_CPU)
            network.force_to(_CPU, torch.float32)
            sampling.to(device)
        else:
            unet.to(device)
            network.force_to(device, torch.float32)
            sampling.to(_CPU)

    t.sd = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=unet,
        network=network,
        _sampling_transformer=sampling,
        text_encoder_to=text_encoder_to,
        _place_training_dit=lambda dev: unet.to(dev) or True,
        _move_main_network=lambda dev: (
            unet.to(dev),
            network.force_to(dev, torch.float32),
        ),
        _move_sampling_transformer=lambda dev: sampling.to(dev),
        _train_on_turbo=train_on_turbo,
        apply_turbo_teacher_mode=apply_turbo_teacher_mode,
        _turbo_calls=turbo_calls,
    )
    if start_with_fake_te and caching:
        # Mid-train caching state: Fake live, real stashed on CPU, backbone on train device.
        te.to(_CPU)
        unload_text_encoder(t.sd)
        unet.to(device)
        network.force_to(device, torch.float32)
        if train_on_turbo:
            apply_turbo_teacher_mode(True)
        else:
            sampling.to(_CPU)
    elif not caching:
        # Unload-only: real TE stays attached (typically CPU between steps).
        te.to(_CPU)
        unet.to(device)
        network.force_to(device, torch.float32)
        sampling.to(_CPU)

    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    return t


def _te_on_cuda(sd: Any) -> bool:
    if not _HAS_CUDA or _CUDA is None:
        return False
    live = getattr(sd, "text_encoder", None)
    if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
        if _any_on(live, _CUDA):
            return True
    real = getattr(sd, "_real_text_encoder", None)
    if isinstance(real, nn.Module):
        return _any_on(real, _CUDA)
    if isinstance(real, list):
        return any(isinstance(x, nn.Module) and _any_on(x, _CUDA) for x in real)
    return False


def _backbone_on_cuda(sd: Any) -> bool:
    if not _HAS_CUDA or _CUDA is None:
        return False
    owners = (
        getattr(sd, "unet", None),
        getattr(sd, "_sampling_transformer", None),
        getattr(sd, "network", None),
    )
    return any(isinstance(m, nn.Module) and _any_on(m, _CUDA) for m in owners)


def _assert_no_te_backbone_coresidency(sd: Any) -> None:
    assert not (_te_on_cuda(sd) and _backbone_on_cuda(sd)), (
        "must not leave real TE and transformer/network co-resident on CUDA"
    )


def test_get_runtime_prompts_parses_json():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "j1"
    t._db_connect = lambda: _conn_returning((json.dumps(["a", "b"]),))
    assert t.get_runtime_prompts() == ["a", "b"]


def test_get_runtime_prompts_invalid_json():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "j1"
    t._db_connect = lambda: _conn_returning(("not-json",))
    assert t.get_runtime_prompts() is None


def test_apply_same_strings_skips_recache():
    t = _make_trainer(
        prompts_db=["hello", "world"],
        sample_prompts=["hello", "world"],
        caching=True,
    )
    t.apply_runtime_prompts()
    assert t._last_applied_runtime_prompts == ("hello", "world")
    t._recache_sample_prompts_runtime.assert_not_called()
    t.cache_sample_prompts.assert_not_called()


def test_apply_changed_strings_updates_by_index():
    t = _make_trainer(
        prompts_db=["new1", "new2", "extra-ignored"],
        sample_prompts=["old1", "old2"],
        caching=False,
        unload_te=False,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "new1"
    assert t.sample_config.samples[1].prompt == "new2"
    assert len(t.sample_config.samples) == 2


def test_apply_shorter_list_leaves_remaining_unchanged():
    t = _make_trainer(
        prompts_db=["only-first"],
        sample_prompts=["a", "b"],
        caching=False,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "only-first"
    assert t.sample_config.samples[1].prompt == "b"


def test_apply_with_cache_calls_recache():
    t = _make_trainer(
        prompts_db=["x"],
        sample_prompts=["y"],
        caching=True,
    )
    t.apply_runtime_prompts()
    assert t.sample_config.samples[0].prompt == "x"
    t._recache_sample_prompts_runtime.assert_called_once()


def test_apply_without_cache_no_te_reload():
    t = _make_trainer(
        prompts_db=["x"],
        sample_prompts=["y"],
        caching=False,
        unload_te=False,
    )
    t.apply_runtime_prompts()
    t._recache_sample_prompts_runtime.assert_not_called()
    t.sd.text_encoder_to.assert_not_called()


def test_apply_unload_only_triggers_recache():
    t = _make_trainer(
        prompts_db=["x"],
        sample_prompts=["y"],
        caching=False,
        unload_te=True,
    )
    t.apply_runtime_prompts()
    t._recache_sample_prompts_runtime.assert_called_once()


def test_recache_sequence_cache_path(monkeypatch):
    """Caching: reload → enter → cache → unload → exit (common pair, not obsolete helpers)."""
    order: List[str] = []
    t = _make_recache_trainer(caching=True, train_device=_CPU)
    import toolkit.unloader as unloader

    real_reload = unloader.reload_text_encoder
    real_enter = unloader.enter_text_cache_residency
    real_unload = unloader.unload_text_encoder
    real_exit = unloader.exit_text_cache_residency

    monkeypatch.setattr(
        unloader,
        "reload_text_encoder",
        lambda m: (order.append("reload"), real_reload(m))[1],
    )
    monkeypatch.setattr(
        unloader,
        "enter_text_cache_residency",
        lambda m, d=None: (order.append("enter"), real_enter(m, d))[1],
    )
    monkeypatch.setattr(
        unloader,
        "unload_text_encoder",
        lambda m: (order.append("unload"), real_unload(m))[1],
    )
    monkeypatch.setattr(
        unloader,
        "exit_text_cache_residency",
        lambda m, d=None: (order.append("exit"), real_exit(m, d))[1],
    )

    def cache_spy():
        order.append("cache")
        assert bool(getattr(t.sd, "_text_cache_residency_active", False))
        assert isinstance(t.sd.text_encoder, nn.Module)
        assert not isinstance(t.sd.text_encoder, FakeTextEncoder)

    t.cache_sample_prompts = cache_spy
    t._recache_sample_prompts_runtime()

    assert order == ["reload", "enter", "cache", "unload", "exit"]
    assert isinstance(t.sd.text_encoder, FakeTextEncoder)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))
    assert devices_equal(next(t.sd.unet.parameters()).device, _CPU)
    assert devices_equal(next(t.sd._sampling_transformer.parameters()).device, _CPU)


def test_recache_unload_only_path(monkeypatch):
    """Unload-only: enter → cache → TE→CPU (no Fake) → exit."""
    order: List[str] = []
    t = _make_recache_trainer(caching=False, unload_te=True, train_device=_CPU)
    import toolkit.unloader as unloader

    real_enter = unloader.enter_text_cache_residency
    real_exit = unloader.exit_text_cache_residency
    reload_mock = MagicMock(side_effect=AssertionError("reload must not run"))
    unload_mock = MagicMock(side_effect=AssertionError("Fake unload must not run"))

    monkeypatch.setattr(unloader, "reload_text_encoder", reload_mock)
    monkeypatch.setattr(unloader, "unload_text_encoder", unload_mock)
    monkeypatch.setattr(
        unloader,
        "enter_text_cache_residency",
        lambda m, d=None: (order.append("enter"), real_enter(m, d))[1],
    )
    monkeypatch.setattr(
        unloader,
        "exit_text_cache_residency",
        lambda m, d=None: (order.append("exit"), real_exit(m, d))[1],
    )

    def cache_spy():
        order.append("cache")
        assert not isinstance(t.sd.text_encoder, FakeTextEncoder)

    t.cache_sample_prompts = cache_spy
    t._recache_sample_prompts_runtime()

    assert order == ["enter", "cache", "exit"]
    reload_mock.assert_not_called()
    unload_mock.assert_not_called()
    assert not isinstance(t.sd.text_encoder, FakeTextEncoder)
    assert devices_equal(next(t.sd.text_encoder.parameters()).device, _CPU)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


def test_recache_normal_restore_layout_cpu():
    t = _make_recache_trainer(caching=True, train_device=_CPU, train_on_turbo=False)
    t._recache_sample_prompts_runtime()
    assert devices_equal(next(t.sd.unet.parameters()).device, _CPU)
    assert devices_equal(next(t.sd.network.parameters()).device, _CPU)
    assert devices_equal(next(t.sd._sampling_transformer.parameters()).device, _CPU)
    assert isinstance(t.sd.text_encoder, FakeTextEncoder)


def test_recache_turbo_restore_layout_cpu():
    t = _make_recache_trainer(caching=True, train_device=_CPU, train_on_turbo=True)
    t._recache_sample_prompts_runtime()
    assert t.sd._turbo_calls and t.sd._turbo_calls[-1] is True
    assert devices_equal(next(t.sd.unet.parameters()).device, _CPU)
    assert devices_equal(next(t.sd._sampling_transformer.parameters()).device, _CPU)


@pytest.mark.parametrize(
    "fail_at",
    ["reload", "enter", "cache", "unload", "exit"],
)
def test_recache_failure_matrix_propagates_and_aborts(monkeypatch, fail_at):
    """Failure at reload/enter/cache/unload/exit: original error, abort cleanup, no coresidency."""
    t = _make_recache_trainer(
        caching=True,
        train_device=_CUDA if _HAS_CUDA else _CPU,
        start_with_fake_te=True,
    )
    import toolkit.unloader as unloader

    real_reload = unloader.reload_text_encoder
    real_enter = unloader.enter_text_cache_residency
    real_unload = unloader.unload_text_encoder
    real_exit = unloader.exit_text_cache_residency
    real_abort = unloader.abort_text_cache_residency
    abort_calls = []

    def maybe_boom(label, fn, *a, **k):
        if fail_at == label:
            raise RuntimeError(f"{label} boom")
        return fn(*a, **k)

    monkeypatch.setattr(
        unloader,
        "reload_text_encoder",
        lambda m: maybe_boom("reload", real_reload, m),
    )
    monkeypatch.setattr(
        unloader,
        "enter_text_cache_residency",
        lambda m, d=None: maybe_boom("enter", real_enter, m, d),
    )
    monkeypatch.setattr(
        unloader,
        "unload_text_encoder",
        lambda m: maybe_boom("unload", real_unload, m),
    )
    monkeypatch.setattr(
        unloader,
        "exit_text_cache_residency",
        lambda m, d=None: maybe_boom("exit", real_exit, m, d),
    )
    monkeypatch.setattr(
        unloader,
        "abort_text_cache_residency",
        lambda m: (abort_calls.append("abort"), real_abort(m))[1],
    )

    if fail_at == "cache":
        t.cache_sample_prompts = MagicMock(side_effect=RuntimeError("cache boom"))

    with pytest.raises(RuntimeError, match=rf"{fail_at} boom") as ei:
        t._recache_sample_prompts_runtime()

    assert abort_calls == ["abort"]
    # Original error must surface (cleanup may be chained as __cause__).
    assert f"{fail_at} boom" in str(ei.value)
    _assert_no_te_backbone_coresidency(t.sd)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


def test_recache_cleanup_failure_chains_original(monkeypatch):
    t = _make_recache_trainer(caching=True, train_device=_CPU)
    import toolkit.unloader as unloader

    t.cache_sample_prompts = MagicMock(side_effect=RuntimeError("cache boom"))

    def boom_abort(_m):
        raise RuntimeError("abort boom")

    monkeypatch.setattr(unloader, "abort_text_cache_residency", boom_abort)

    with pytest.raises(RuntimeError, match="cache boom") as ei:
        t._recache_sample_prompts_runtime()

    assert ei.value.__cause__ is not None
    assert "abort boom" in str(ei.value.__cause__)


def test_reset_last_applied_clears_prompts():
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t._last_applied_runtime_lr = 1.0
    t._last_applied_runtime_gaussian_mean = None
    t._last_applied_runtime_gaussian_std = None
    t._last_applied_runtime_gaussian_mean_2 = None
    t._last_applied_runtime_gaussian_std_2 = None
    t._last_applied_runtime_weight_decay = None
    t._last_applied_runtime_weight_decay_increment = None
    t._last_applied_runtime_weight_decay_mode = None
    t._last_applied_runtime_beta1 = None
    t._last_applied_runtime_beta2 = None
    t._last_applied_runtime_content_or_style = None
    t._last_applied_runtime_timestep_type = None
    t._last_applied_runtime_timestep_weighting = None
    t._last_applied_runtime_network_weights = None
    t._last_applied_runtime_prompts = ("a",)
    t._last_applied_runtime_batch_size = None
    t._last_applied_runtime_gradient_accumulation = None
    t._last_applied_runtime_save_every = None
    t._last_applied_runtime_sample_every = None
    t._last_applied_runtime_warmup_steps = None
    t._last_applied_runtime_warmup_boost = None
    t._last_applied_runtime_min_snr_gamma = None
    t._last_applied_runtime_debug = None
    t._last_applied_runtime_fc_key = None
    t._last_applied_runtime_turbo_prior_steps = None
    t._last_applied_runtime_turbo_t_jitter = None
    t._last_applied_runtime_turbo_teacher_weight = None
    DiffusionTrainer._reset_last_applied_runtime(t)
    assert t._last_applied_runtime_prompts is None


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for residency check")
def test_recache_cuda_normal_restore_no_coresidency():
    assert _CUDA is not None
    t = _make_recache_trainer(caching=True, train_device=_CUDA, train_on_turbo=False)
    t._recache_sample_prompts_runtime()
    assert isinstance(t.sd.text_encoder, FakeTextEncoder)
    assert _any_on(t.sd.unet, _CUDA)
    assert _any_on(t.sd.network, _CUDA)
    assert not _any_on(t.sd._sampling_transformer, _CUDA)
    _assert_no_te_backbone_coresidency(t.sd)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for turbo residency check")
def test_recache_cuda_turbo_restore_no_coresidency():
    assert _CUDA is not None
    t = _make_recache_trainer(caching=True, train_device=_CUDA, train_on_turbo=True)
    t._recache_sample_prompts_runtime()
    assert t.sd._turbo_calls and t.sd._turbo_calls[-1] is True
    assert not _any_on(t.sd.unet, _CUDA)
    assert _any_on(t.sd._sampling_transformer, _CUDA)
    _assert_no_te_backbone_coresidency(t.sd)
