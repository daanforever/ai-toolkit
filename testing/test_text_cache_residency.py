"""Regression contract for text-cache residency enter/exit lifecycle.

Lifecycle:
  enter: real TE -> CPU; all persistent non-TE owners -> CPU; flush; real TE -> target
  encode with TE-only CUDA residency
  unload TE (FakeTextEncoder / CPU stash)
  exit: restore normal or turbo train layout; refuse if real TE still on CUDA
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from toolkit.unloader import (
    FakeTextEncoder,
    _TEXT_CACHE_AUX_RESTORE,
    _move_owner_to_device,
    abort_text_cache_residency,
    enter_text_cache_residency,
    exit_text_cache_residency,
    unload_text_encoder,
)
from toolkit.util.device import devices_equal, quantized_payload_device

_HAS_CUDA = torch.cuda.is_available()
_CUDA = torch.device("cuda") if _HAS_CUDA else None
_CPU = torch.device("cpu")

# Documented lifecycle errors for second-exit / exit-gate contracts.
_LIFECYCLE_ERROR_TYPES = (RuntimeError, ValueError)
_LIFECYCLE_ERROR_MATCH = r"(?i)lifecycle|residency|text encoder|exit"


# ---------------------------------------------------------------------------
# Tiny owners / seams
# ---------------------------------------------------------------------------


class _TinyLinear(nn.Module):
    """Minimal persistent owner: one Parameter + one buffer."""

    def __init__(self, n: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, n))
        self.register_buffer("bias_buf", torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


class _TinyLoRANetwork(nn.Module):
    """Classic LoRA-like network with registered children + ``force_to``."""

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

    def parameters_for_optimizer(self) -> List[nn.Parameter]:
        return [self.lora_A.weight, self.lora_B.weight]


class _ListOnlyLoRANetwork(nn.Module):
    """LoRASpecial-like: LoRA modules live only in plain lists, not as children.

    ``safe_module_to_device(self)`` cannot see them; ``force_to(device, dtype)`` must.
    """

    def __init__(self, n: int = 4, rank: int = 2):
        super().__init__()
        # Intentionally NOT registered via self.lora_A = ...
        self.unet_loras = [
            nn.Linear(n, rank, bias=False),
            nn.Linear(rank, n, bias=False),
        ]
        nn.init.zeros_(self.unet_loras[1].weight)
        self.text_encoder_loras: List[nn.Module] = []

    def force_to(self, device, dtype):
        if dtype is None:
            raise TypeError("force_to requires dtype (LoRASpecial contract)")
        for lora in list(self.unet_loras) + list(self.text_encoder_loras):
            lora.to(device, dtype)

    def parameters_for_optimizer(self) -> List[nn.Parameter]:
        return [p for m in self.unet_loras for p in m.parameters()]


class _TinyPeftNetwork(nn.Module):
    """PEFT-like wrapper: frozen base + trainable adapter; ``force_to`` moves both."""

    def __init__(self, n: int = 4, rank: int = 2):
        super().__init__()
        self.base = nn.Linear(n, n, bias=False)
        self.base.weight.requires_grad_(False)
        self.adapter = nn.Linear(n, rank, bias=False)
        self.adapter_out = nn.Linear(rank, n, bias=False)
        nn.init.zeros_(self.adapter_out.weight)

    def force_to(self, device, dtype):
        self.to(device)
        if dtype is not None:
            for p in self.adapter.parameters():
                p.data = p.data.to(dtype=dtype)
            for p in self.adapter_out.parameters():
                p.data = p.data.to(dtype=dtype)

    def parameters_for_optimizer(self) -> List[nn.Parameter]:
        return list(self.adapter.parameters()) + list(self.adapter_out.parameters())


_NetworkT = Union[_TinyLoRANetwork, _TinyPeftNetwork, _ListOnlyLoRANetwork]


def _attach_fake_quant_payload(param: nn.Parameter, device: torch.device) -> None:
    """Simulate torchao/quanto payloads that Module.to() does not relocate."""
    param.qdata = torch.randn_like(param.data, device=device)
    param.scale = torch.tensor(1.0, device=device)


def _first_param(module: nn.Module) -> nn.Parameter:
    return next(module.parameters())


def _module_param_devices(module: nn.Module) -> List[torch.device]:
    return [p.device for p in module.parameters()]


def _module_buffer_devices(module: nn.Module) -> List[torch.device]:
    return [b.device for b in module.buffers() if b is not None]


def _all_on(module: nn.Module, device: torch.device) -> bool:
    return all(devices_equal(d, device) for d in _module_param_devices(module)) and all(
        devices_equal(d, device) for d in _module_buffer_devices(module)
    )


def _any_on(module: nn.Module, device: torch.device) -> bool:
    return any(devices_equal(d, device) for d in _module_param_devices(module)) or any(
        devices_equal(d, device) for d in _module_buffer_devices(module)
    )


def _payload_devices(module: nn.Module) -> List[torch.device]:
    out: List[torch.device] = []
    for p in module.parameters():
        d = quantized_payload_device(p)
        if d is not None:
            out.append(d)
    return out


def _owner_map(model: Any) -> Dict[str, Any]:
    """Named non-TE residency owners (dedupe by object id)."""
    names = (
        "unet",
        "vae",
        "network",
        "_sampling_transformer",
        "_sampling_network",
        "adapter",
        "refiner_unet",
    )
    seen: set[int] = set()
    out: Dict[str, Any] = {}
    for name in names:
        obj = getattr(model, name, None)
        if obj is None or not isinstance(obj, nn.Module):
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        out[name] = obj
    return out


def _te_modules(model: Any) -> List[nn.Module]:
    te = getattr(model, "text_encoder", None)
    if te is None:
        return []
    if isinstance(te, list):
        return [x for x in te if isinstance(x, nn.Module) and not isinstance(x, FakeTextEncoder)]
    if isinstance(te, FakeTextEncoder):
        return []
    return [te] if isinstance(te, nn.Module) else []


def _cuda_owner_names(model: Any) -> List[str]:
    """Owners with any persistent param/buffer (or fake quant payload) on CUDA."""
    if not _HAS_CUDA or _CUDA is None:
        return []
    names: List[str] = []
    for name, mod in _owner_map(model).items():
        if _any_on(mod, _CUDA):
            names.append(name)
            continue
        if any(devices_equal(d, _CUDA) for d in _payload_devices(mod)):
            names.append(name)
    for i, te in enumerate(_te_modules(model)):
        if _any_on(te, _CUDA) or any(devices_equal(d, _CUDA) for d in _payload_devices(te)):
            names.append(f"text_encoder[{i}]" if i else "text_encoder")
    return names


def _device_is_cpu(dev_str: str) -> bool:
    return torch.device(dev_str).type == "cpu"


def _share_network_params(main: _NetworkT, sampling: _NetworkT) -> None:
    if isinstance(main, _TinyPeftNetwork) and isinstance(sampling, _TinyPeftNetwork):
        sampling.adapter.weight = main.adapter.weight
        return
    if isinstance(main, _TinyLoRANetwork) and isinstance(sampling, _TinyLoRANetwork):
        sampling.lora_A.weight = main.lora_A.weight
        sampling.lora_B.weight = main.lora_B.weight
        return
    if isinstance(main, _ListOnlyLoRANetwork) and isinstance(sampling, _ListOnlyLoRANetwork):
        sampling.unet_loras[0].weight = main.unet_loras[0].weight
        sampling.unet_loras[1].weight = main.unet_loras[1].weight
        return
    raise TypeError("main/sampling networks must be the same Tiny network type")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_model(
    *,
    device: torch.device,
    as_list_te: bool = False,
    with_pipeline: bool = False,
    use_peft: bool = False,
    with_quant_payload_on_unet: bool = False,
    train_on_turbo: bool = False,
) -> SimpleNamespace:
    te = _TinyLinear(4).to(device)
    te2 = _TinyLinear(4).to(device) if as_list_te else None
    unet = _TinyLinear(4).to(device)
    vae = _TinyLinear(4).to(device)
    adapter = _TinyLinear(4).to(device)
    refiner = _TinyLinear(4).to(device)
    sampling = _TinyLinear(4).to(device)

    if use_peft:
        network: _NetworkT = _TinyPeftNetwork(4).to(device)
        sampling_network: _NetworkT = _TinyPeftNetwork(4).to(device)
    else:
        network = _TinyLoRANetwork(4).to(device)
        sampling_network = _TinyLoRANetwork(4).to(device)
    _share_network_params(network, sampling_network)

    if with_quant_payload_on_unet:
        _attach_fake_quant_payload(_first_param(unet), device)

    pipe = None
    if with_pipeline:
        if as_list_te:
            pipe = SimpleNamespace(text_encoder=te, text_encoder_2=te2)
        else:
            pipe = SimpleNamespace(text_encoder=te)

    te_attr: Any = [te, te2] if as_list_te else te

    move_log: List[Tuple[str, str]] = []

    def text_encoder_to(dev, *args, **kwargs):
        move_log.append(("text_encoder_to", str(torch.device(dev))))
        if isinstance(te_attr, list):
            for enc in te_attr:
                if enc is not None and not isinstance(enc, FakeTextEncoder):
                    enc.to(dev)
        elif not isinstance(te_attr, FakeTextEncoder):
            te_attr.to(dev)

    def place_training_dit(dev):
        move_log.append(("_place_training_dit", str(torch.device(dev))))
        unet.to(dev)
        return True

    def move_main_network(dev):
        move_log.append(("_move_main_network", str(torch.device(dev))))
        unet.to(dev)
        network.force_to(dev, torch.float32)

    def move_sampling_transformer(dev):
        move_log.append(("_move_sampling_transformer", str(torch.device(dev))))
        sampling.to(dev)

    def apply_turbo_teacher_mode(enabled: bool):
        move_log.append(("apply_turbo_teacher_mode", str(bool(enabled))))
        if enabled:
            unet.to("cpu")
            network.force_to("cpu", torch.float32)
            sampling.to(device)
            sampling_network.force_to(device, torch.float32)
        else:
            sampling.to("cpu")
            sampling_network.force_to("cpu", torch.float32)
            unet.to(device)
            network.force_to(device, torch.float32)

    opt_params = network.parameters_for_optimizer()
    optimizer = torch.optim.SGD(opt_params, lr=1e-3)
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

    model = SimpleNamespace(
        text_encoder=te_attr,
        pipeline=pipe,
        device_torch=device,
        torch_dtype=torch.float32,
        unet=unet,
        vae=vae,
        network=network,
        _sampling_transformer=sampling,
        _sampling_network=sampling_network,
        adapter=adapter,
        refiner_unet=refiner,
        text_encoder_to=text_encoder_to,
        _place_training_dit=place_training_dit,
        _move_main_network=move_main_network,
        _move_sampling_transformer=move_sampling_transformer,
        apply_turbo_teacher_mode=apply_turbo_teacher_mode,
        _train_on_turbo=train_on_turbo,
        _move_log=move_log,
        _optimizer=optimizer,
        _opt_ids=opt_ids,
        _shared_main_param=network.parameters_for_optimizer()[0],
        _shared_sampling_param=sampling_network.parameters_for_optimizer()[0],
    )
    return model


def _assert_te_only_on_cuda(model: Any) -> None:
    cuda_owners = _cuda_owner_names(model)
    non_te = [n for n in cuda_owners if not n.startswith("text_encoder")]
    assert not non_te, (
        "during text-cache encode, only real TE may hold CUDA tensors; "
        f"leaked owners={non_te!r} all_cuda={cuda_owners!r}"
    )
    te_on = [n for n in cuda_owners if n.startswith("text_encoder")]
    assert te_on, "real TE must be on CUDA during encode"


def _assert_no_real_te_on_cuda(model: Any) -> None:
    if _CUDA is None:
        return
    for te in _te_modules(model):
        assert not _any_on(te, _CUDA), "real TE must not remain on CUDA after unload"
    real = getattr(model, "_real_text_encoder", None)
    if real is None:
        return
    modules = real if isinstance(real, list) else [real]
    for te in modules:
        if isinstance(te, nn.Module) and not isinstance(te, FakeTextEncoder):
            assert not _any_on(te, _CUDA)


def _assert_normal_restore_layout(model: Any, train_device: torch.device) -> None:
    assert _all_on(model.unet, train_device), "normal exit: main transformer on train device"
    assert _all_on(model._sampling_transformer, _CPU), "normal exit: sampling transformer on CPU"
    assert devices_equal(model._shared_main_param.device, train_device), (
        "normal exit: shared network params on train device"
    )
    if _HAS_CUDA and _CUDA is not None and devices_equal(train_device, _CUDA):
        assert not (
            _any_on(model.unet, _CUDA) and _any_on(model._sampling_transformer, _CUDA)
        ), "normal exit must not co-reside main and sampling transformers on CUDA"


def _assert_turbo_restore_layout(model: Any, train_device: torch.device) -> None:
    assert _all_on(model._sampling_transformer, train_device), (
        "turbo exit: sampling transformer on train device"
    )
    assert _all_on(model.unet, _CPU), "turbo exit: main transformer on CPU"
    assert devices_equal(model._shared_main_param.device, train_device), (
        "turbo exit: shared network params on train device"
    )
    if _HAS_CUDA and _CUDA is not None and devices_equal(train_device, _CUDA):
        assert not (
            _any_on(model.unet, _CUDA) and _any_on(model._sampling_transformer, _CUDA)
        ), "turbo exit must not co-reside main and sampling transformers on CUDA"


# ---------------------------------------------------------------------------
# Enter / exit contract regressions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for TE-only residency invariant")
@pytest.mark.parametrize("use_peft", [False, True], ids=["lora", "peft"])
def test_regression_enter_offloads_lora_peft_sampling(use_peft):
    """Enter must park network and sampling off CUDA (LoRA and PEFT)."""
    assert _CUDA is not None
    model = _build_model(device=_CUDA, use_peft=use_peft)
    enter_text_cache_residency(model)
    leaked = [
        n
        for n in _cuda_owner_names(model)
        if n in ("network", "_sampling_transformer", "_sampling_network")
    ]
    assert leaked == [], f"enter left network/sampling on CUDA: {leaked!r}"
    assert not _any_on(model.network, _CUDA)
    assert not _any_on(model._sampling_transformer, _CUDA)
    assert not _any_on(model._sampling_network, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for TE-only residency invariant")
@pytest.mark.parametrize("use_peft", [False, True])
@pytest.mark.parametrize("as_list_te,with_pipeline", [(False, False), (True, True)])
def test_regression_enter_leaves_only_real_te_on_cuda(use_peft, as_list_te, with_pipeline):
    """After enter, persistent CUDA tensors belong only to real TE."""
    assert _CUDA is not None
    model = _build_model(
        device=_CUDA,
        use_peft=use_peft,
        as_list_te=as_list_te,
        with_pipeline=with_pipeline,
    )
    enter_text_cache_residency(model)
    _assert_te_only_on_cuda(model)


def test_regression_enter_order_covers_all_non_te_owners_cpu(monkeypatch):
    """Full order: TE→CPU; every non-TE owner→CPU; flush; TE→target."""
    events: List[Tuple[str, str]] = []
    device = _CUDA if _HAS_CUDA else _CPU
    required_owners = (
        "unet",
        "vae",
        "adapter",
        "refiner_unet",
        "_sampling_transformer",
        "network",
        "_sampling_network",
    )

    te = _TinyLinear(2).to(device)
    unet = _TinyLinear(2).to(device)
    vae = _TinyLinear(2).to(device)
    adapter = _TinyLinear(2).to(device)
    refiner = _TinyLinear(2).to(device)
    sampling = _TinyLinear(2).to(device)
    network = _TinyLoRANetwork(2).to(device)
    sampling_network = _TinyLoRANetwork(2).to(device)

    import toolkit.unloader as unloader_mod

    real_flush = unloader_mod.flush
    flush_seq = {"n": 0}

    def counting_flush(*a, **k):
        flush_seq["n"] += 1
        events.append(("flush", str(flush_seq["n"])))
        return real_flush(*a, **k)

    monkeypatch.setattr(unloader_mod, "flush", counting_flush)

    real_move = unloader_mod._move_owner_to_device

    def tracking_move(name, module, device_arg):
        events.append((name, str(torch.device(device_arg))))
        return real_move(name, module, device_arg)

    monkeypatch.setattr(unloader_mod, "_move_owner_to_device", tracking_move)

    real_net_move = unloader_mod._move_network_owner_to_device

    def tracking_net_move(name, network_mod, device_arg, model=None):
        events.append((name, str(torch.device(device_arg))))
        return real_net_move(name, network_mod, device_arg, model=model)

    monkeypatch.setattr(unloader_mod, "_move_network_owner_to_device", tracking_net_move)

    def text_encoder_to(dev, *a, **k):
        events.append(("text_encoder", str(torch.device(dev))))
        te.to(dev)

    def place(dev):
        events.append(("unet", str(torch.device(dev))))
        unet.to(dev)
        return True

    def move_sampling(dev):
        events.append(("_sampling_transformer", str(torch.device(dev))))
        sampling.to(dev)

    model = SimpleNamespace(
        text_encoder=te,
        text_encoder_to=text_encoder_to,
        _place_training_dit=place,
        _move_sampling_transformer=move_sampling,
        unet=unet,
        vae=vae,
        adapter=adapter,
        refiner_unet=refiner,
        network=network,
        _sampling_transformer=sampling,
        _sampling_network=sampling_network,
        device_torch=device,
        torch_dtype=torch.float32,
        pipeline=None,
    )

    enter_text_cache_residency(model)

    def _first_index(predicate: Callable[[Tuple[str, str]], bool]) -> Optional[int]:
        for i, ev in enumerate(events):
            if predicate(ev):
                return i
        return None

    te_cpu_idx = _first_index(lambda ev: ev[0] == "text_encoder" and _device_is_cpu(ev[1]))
    assert te_cpu_idx is not None, f"enter must start with TE→CPU; events={events!r}"

    owner_cpu_idxs: Dict[str, int] = {}
    missing: List[str] = []
    for owner in required_owners:
        idx = _first_index(
            lambda ev, name=owner: ev[0] == name and _device_is_cpu(ev[1])
        )
        if idx is None:
            missing.append(owner)
            continue
        assert idx > te_cpu_idx, (
            f"{owner}→CPU must occur after first TE→CPU "
            f"(te_cpu={te_cpu_idx}, {owner}={idx}); events={events!r}"
        )
        owner_cpu_idxs[owner] = idx

    assert not missing, (
        "enter must move every non-TE owner to CPU; "
        f"missing={missing} events={events!r}"
    )

    last_owner_cpu = max(owner_cpu_idxs.values())
    flush_idx = None
    for i in range(last_owner_cpu + 1, len(events)):
        if events[i][0] == "flush":
            flush_idx = i
            break
    assert flush_idx is not None, (
        "required flush must occur after all non-TE CPU moves; "
        f"last_owner_cpu={last_owner_cpu} events={events!r}"
    )

    te_target_idx = None
    for i in range(flush_idx + 1, len(events)):
        lab, dev = events[i]
        if lab == "text_encoder" and not _device_is_cpu(dev):
            te_target_idx = i
            break
    assert te_target_idx is not None, (
        "TE→target must occur after the required flush; "
        f"flush_idx={flush_idx} events={events!r}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for quantized payload leak")
def test_regression_enter_moves_quantized_payloads_off_cuda():
    """Enter relocates fake quant payloads, not only param.device via Module.to()."""
    assert _CUDA is not None
    model = _build_model(device=_CUDA, with_quant_payload_on_unet=True)
    payload_before = quantized_payload_device(_first_param(model.unet))
    assert payload_before is not None and devices_equal(payload_before, _CUDA)

    enter_text_cache_residency(model)

    payload_after = quantized_payload_device(_first_param(model.unet))
    assert payload_after is not None
    assert devices_equal(payload_after, _CPU), (
        "enter must relocate quantized payloads to CPU; "
        f"got payload={payload_after} param={_first_param(model.unet).device}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_regression_normal_restore_layout_after_unload():
    assert _CUDA is not None
    model = _build_model(device=_CUDA, train_on_turbo=False)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    _assert_no_real_te_on_cuda(model)
    exit_text_cache_residency(model)
    _assert_normal_restore_layout(model, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_regression_turbo_restore_layout_after_unload():
    assert _CUDA is not None
    model = _build_model(device=_CUDA, train_on_turbo=True)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    _assert_turbo_restore_layout(model, _CUDA)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_regression_exit_forbidden_while_real_te_on_cuda():
    """Exit must refuse remounting train components while real TE remains on CUDA."""
    assert _CUDA is not None
    model = _build_model(device=_CUDA)
    enter_text_cache_residency(model)
    assert _any_on(_te_modules(model)[0], _CUDA)

    with pytest.raises(_LIFECYCLE_ERROR_TYPES, match=_LIFECYCLE_ERROR_MATCH):
        exit_text_cache_residency(model)


def test_regression_repeated_enter_idempotent_no_extra_owner_moves():
    """Second enter must be a documented no-op (no extra owner moves)."""
    device = _CUDA if _HAS_CUDA else _CPU
    model = _build_model(device=device)
    enter_text_cache_residency(model)
    log_after_first = list(model._move_log)

    enter_text_cache_residency(model)
    log_after_second = list(model._move_log)

    extra = log_after_second[len(log_after_first) :]
    assert extra == [], f"repeated enter must not perform extra moves; got {extra!r}"


def test_regression_second_exit_is_noop_or_lifecycle_error():
    """Second exit after a completed restore is no-op or explicit lifecycle error."""
    device = _CUDA if _HAS_CUDA else _CPU
    model = _build_model(device=device, train_on_turbo=False)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    log_after_first = list(model._move_log)

    try:
        exit_text_cache_residency(model)
    except _LIFECYCLE_ERROR_TYPES as exc:
        msg = str(exc).lower()
        assert (
            "lifecycle" in msg or "residency" in msg or "exit" in msg or "text encoder" in msg
        ), f"unexpected lifecycle error message: {exc!r}"
        return

    extra = model._move_log[len(log_after_first) :]
    assert extra == [], f"second exit must no-op; got extra moves {extra!r}"


def test_move_owner_failure_includes_component_and_devices(monkeypatch):
    """Component move failures must name the owner and source/target devices."""
    if not _HAS_CUDA or _CUDA is None:
        pytest.skip("CUDA required to force a real device transition")

    mod = _TinyLinear(2).to(_CUDA)

    def boom(module, device, dtype=None):
        raise RuntimeError("simulated move failure")

    monkeypatch.setattr("toolkit.unloader.safe_module_to_device", boom)
    with pytest.raises(RuntimeError) as ei:
        _move_owner_to_device("vae", mod, "cpu")
    msg = str(ei.value)
    assert "vae" in msg
    assert "cpu" in msg.lower()
    assert "cuda" in msg.lower()


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_enter_moves_list_only_lora_via_force_to():
    """LoRA modules held only in plain lists require force_to(device, dtype)."""
    assert _CUDA is not None
    network = _ListOnlyLoRANetwork(4)
    for lora in network.unet_loras:
        lora.to(_CUDA)
    assert any(p.device.type == "cuda" for p in network.parameters_for_optimizer())
    # Parent has no registered children — safe_module_to_device alone cannot see list loras.
    assert list(network.parameters()) == []

    force_calls: List[Tuple[str, Any]] = []
    orig = network.force_to

    def spy(device, dtype):
        force_calls.append((str(torch.device(device)), dtype))
        return orig(device, dtype)

    setattr(network, "force_to", spy)

    te = _TinyLinear(4).to(_CUDA)
    unet = _TinyLinear(4).to(_CUDA)

    def text_encoder_to(dev, *a, **k):
        te.to(dev)

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        vae=None,
        adapter=None,
        refiner_unet=None,
        network=network,
        _sampling_transformer=None,
        _sampling_network=None,
        text_encoder_to=text_encoder_to,
        _train_on_turbo=False,
    )
    enter_text_cache_residency(model)
    assert force_calls, "enter must call network.force_to for list-only LoRA"
    assert _device_is_cpu(force_calls[0][0])
    assert force_calls[0][1] is not None
    assert all(p.device.type == "cpu" for p in network.parameters_for_optimizer())


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_exit_refuses_when_stashed_real_te_on_cuda():
    """Exit must see stashed real TE on CUDA even when live slot is Fake."""
    assert _CUDA is not None
    model = _build_model(device=_CUDA)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    assert isinstance(model.text_encoder, FakeTextEncoder)
    real = model._real_text_encoder
    assert isinstance(real, nn.Module)
    # Simulate broken unload that left stash on CUDA.
    real.to(_CUDA)
    assert _any_on(real, _CUDA)

    with pytest.raises(_LIFECYCLE_ERROR_TYPES, match=_LIFECYCLE_ERROR_MATCH):
        exit_text_cache_residency(model)
    assert getattr(model, "_text_cache_residency_active", False) is True
    assert not _any_on(model.unet, _CUDA), "exit must not remount backbone when stash TE is CUDA"


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
def test_enter_partial_te_move_fails_without_activating_lifecycle():
    """Multi-TE partial text_encoder_to on remount must fail; no active flag; no backbone remount."""
    assert _CUDA is not None
    te1 = _TinyLinear(4).to(_CUDA)
    te2 = _TinyLinear(4).to(_CUDA)
    unet = _TinyLinear(4).to(_CUDA)
    phase = {"n": 0}

    def partial_text_encoder_to(dev, *a, **k):
        phase["n"] += 1
        target = torch.device(dev)
        if phase["n"] == 1:
            # First call (TE→CPU): move both so offload can proceed.
            te1.to(target)
            te2.to(target)
            return
        # Second call (TE→target): only first encoder — partial/no-op for te2.
        te1.to(target)

    model = SimpleNamespace(
        text_encoder=[te1, te2],
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        vae=None,
        adapter=None,
        refiner_unet=None,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        text_encoder_to=partial_text_encoder_to,
        _train_on_turbo=False,
    )
    with pytest.raises(RuntimeError, match=r"text_encoder|still not on"):
        enter_text_cache_residency(model)
    assert getattr(model, "_text_cache_residency_active", False) is not True
    assert phase["n"] >= 2, "both TE→CPU and TE→target phases must run before fail"
    assert not _any_on(unet, _CUDA), "backbone must remain off CUDA after failed TE remount"
    assert not (_any_on(te1, _CUDA) and _any_on(unet, _CUDA))
    assert not (_any_on(te2, _CUDA) and _any_on(unet, _CUDA))


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required to force a network device transition")
def test_network_force_to_failure_not_suppressed(monkeypatch):
    """force_to failures must propagate with component + source/target."""
    assert _CUDA is not None
    network = _ListOnlyLoRANetwork(2)
    for lora in network.unet_loras:
        lora.to(_CUDA)

    def boom(device_arg, dtype):
        raise RuntimeError("force_to exploded")

    setattr(network, "force_to", boom)
    from toolkit.unloader import _move_network_owner_to_device

    with pytest.raises(RuntimeError) as ei:
        _move_network_owner_to_device(
            "network",
            network,
            "cpu",
            model=SimpleNamespace(torch_dtype=torch.float32),
        )
    msg = str(ei.value)
    assert "network" in msg
    assert "cpu" in msg.lower()
    assert "cuda" in msg.lower()
    assert "force_to exploded" in msg


# ---------------------------------------------------------------------------
# Identity seams
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("use_peft", [False, True])
def test_parameter_optimizer_and_shared_identity_survive_enter_exit_cycle(use_peft):
    """Parameter / optimizer / shared main↔sampling identity must survive enter/unload/exit."""
    assert _CUDA is not None
    model = _build_model(device=_CUDA, use_peft=use_peft)
    shared_main = model._shared_main_param
    shared_sampling = model._shared_sampling_param
    assert shared_main is shared_sampling
    opt_ids_before = set(model._opt_ids)
    param_ids_before = {id(p) for p in model.network.parameters_for_optimizer()}

    enter_text_cache_residency(model)
    unload_text_encoder(model)
    exit_text_cache_residency(model)

    assert model._shared_main_param is shared_main
    assert model._shared_sampling_param is shared_main
    assert {id(p) for p in model.network.parameters_for_optimizer()} == param_ids_before
    assert {id(p) for g in model._optimizer.param_groups for p in g["params"]} == opt_ids_before


# ---------------------------------------------------------------------------
# Runtime recache exception cleanup (DiffusionTrainer — common enter/exit/abort)
# ---------------------------------------------------------------------------


def _make_recache_trainer_cuda() -> Any:
    """Mid-train CUDA fixture: Fake TE stash + backbone on CUDA before runtime recache."""
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    assert _CUDA is not None
    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.is_ui_trainer = True
    t.job_id = "job-residency"
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.train_config = SimpleNamespace(unload_text_encoder=False)
    t.sample_config = SimpleNamespace(samples=[SimpleNamespace(prompt="old")])
    t._last_applied_runtime_prompts = None

    te = _TinyLinear(2).to(_CUDA)
    unet = _TinyLinear(2).to(_CUDA)
    network = _TinyLoRANetwork(2).to(_CUDA)
    sampling = _TinyLinear(2).to(_CUDA)

    def text_encoder_to(dev, *a, **k):
        live = t.sd.text_encoder
        if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
            live.to(dev)
        real = getattr(t.sd, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            real.to(dev)

    t.sd = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=_CUDA,
        torch_dtype=torch.float32,
        unet=unet,
        network=network,
        _sampling_transformer=sampling,
        text_encoder_to=text_encoder_to,
        _place_training_dit=lambda dev: unet.to(dev) or True,
        _move_main_network=lambda dev: (unet.to(dev), network.force_to(dev, torch.float32)),
        _move_sampling_transformer=lambda dev: sampling.to(dev),
        _train_on_turbo=False,
    )
    unload_text_encoder(t.sd)
    unet.to(_CUDA)
    network.force_to(_CUDA, torch.float32)
    sampling.to(_CUDA)
    return t


def _assert_recache_no_te_backbone_coresidency(sd: Any) -> None:
    te_cuda = False
    if isinstance(sd.text_encoder, nn.Module) and not isinstance(
        sd.text_encoder, FakeTextEncoder
    ):
        te_cuda = _any_on(sd.text_encoder, _CUDA)
    else:
        real = getattr(sd, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            te_cuda = _any_on(real, _CUDA)
        elif isinstance(real, list):
            te_cuda = any(
                isinstance(x, nn.Module) and _any_on(x, _CUDA) for x in real
            )

    backbone_cuda = (
        _any_on(sd.unet, _CUDA)
        or _any_on(sd._sampling_transformer, _CUDA)
        or _any_on(sd.network, _CUDA)
    )
    assert not (te_cuda and backbone_cuda), (
        "exception cleanup must not leave real TE and transformer/network co-resident on CUDA; "
        f"te_cuda={te_cuda} backbone_cuda={backbone_cuda}"
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for coresidency cleanup check")
def test_regression_recache_exception_cleanup_no_te_transformer_coresidency():
    """Runtime cache failure must abort without TE+backbone CUDA co-residency."""
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    assert _CUDA is not None
    t = _make_recache_trainer_cuda()
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )

    def boom():
        raise RuntimeError("encode/cache failed")

    t.cache_sample_prompts = boom

    with pytest.raises(RuntimeError, match="encode/cache failed"):
        t._recache_sample_prompts_runtime()

    _assert_recache_no_te_backbone_coresidency(t.sd)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for runtime failure matrix")
@pytest.mark.parametrize("fail_at", ["reload", "cache", "unload", "exit"])
def test_runtime_recache_failure_matrix_no_coresidency(monkeypatch, fail_at):
    """Inject failures at reload/cache/unload/exit; abort cleanup, original error propagates."""
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    import toolkit.unloader as unloader

    assert _CUDA is not None
    t = _make_recache_trainer_cuda()
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )

    real_reload = unloader.reload_text_encoder
    real_unload = unloader.unload_text_encoder
    real_exit = unloader.exit_text_cache_residency

    def maybe_boom(label, fn, *a, **k):
        if fail_at == label:
            raise RuntimeError(f"{label} boom")
        return fn(*a, **k)

    monkeypatch.setattr(
        unloader, "reload_text_encoder", lambda m: maybe_boom("reload", real_reload, m)
    )
    monkeypatch.setattr(
        unloader, "unload_text_encoder", lambda m: maybe_boom("unload", real_unload, m)
    )
    monkeypatch.setattr(
        unloader,
        "exit_text_cache_residency",
        lambda m, d=None: maybe_boom("exit", real_exit, m, d),
    )
    if fail_at == "cache":
        t.cache_sample_prompts = MagicMock(side_effect=RuntimeError("cache boom"))
    else:
        t.cache_sample_prompts = MagicMock()

    with pytest.raises(RuntimeError, match=rf"{fail_at} boom"):
        t._recache_sample_prompts_runtime()

    _assert_recache_no_te_backbone_coresidency(t.sd)
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for internal unload move failure")
def test_runtime_recache_unload_internal_te_move_failure_propagates(monkeypatch):
    """Unload fails via real TE move (not whole-unload monkeypatch); no Fake; abort cleans."""
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
    import toolkit.unloader as unloader

    assert _CUDA is not None
    t = _make_recache_trainer_cuda()
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    t.cache_sample_prompts = MagicMock()

    real_te = t.sd._real_text_encoder
    assert isinstance(real_te, nn.Module)
    real_safe = unloader.safe_module_to_device
    real_unload = unloader.unload_text_encoder
    boom_on_unload = {"armed": False}

    def selective_boom(module, device, dtype=None):
        target = torch.device(device)
        if boom_on_unload["armed"] and module is real_te and target.type == "cpu":
            raise RuntimeError("internal te unload move boom")
        return real_safe(module, device, dtype)

    def unload_wrapper(model):
        boom_on_unload["armed"] = True
        try:
            return real_unload(model)
        finally:
            boom_on_unload["armed"] = False

    monkeypatch.setattr(unloader, "safe_module_to_device", selective_boom)
    monkeypatch.setattr(unloader, "unload_text_encoder", unload_wrapper)

    with pytest.raises(RuntimeError, match="internal te unload move boom") as ei:
        t._recache_sample_prompts_runtime()

    msg = str(ei.value)
    assert "text_encoder" in msg
    assert "internal te unload move boom" in msg
    assert not bool(getattr(t.sd, "_text_cache_residency_active", False))
    _assert_recache_no_te_backbone_coresidency(t.sd)
    # Failed unload must not leave Fake installed over the real TE identity.
    live = t.sd.text_encoder
    if isinstance(live, FakeTextEncoder):
        # Abort may leave Fake only if unload partially installed — must not.
        raise AssertionError("FakeTextEncoder installed despite unload move failure")
    assert live is real_te or getattr(t.sd, "_real_text_encoder", None) is real_te


# ---------------------------------------------------------------------------
# Primary auxiliary snapshot restore (vae / adapter / refiner_unet / image_encoder)
# ---------------------------------------------------------------------------


def _aux_owner_model(
    *,
    train_device: torch.device,
    aux_device: torch.device,
) -> SimpleNamespace:
    te = _TinyLinear(2).to(train_device)
    unet = _TinyLinear(2).to(train_device)
    vae = _TinyLinear(2).to(aux_device)
    adapter = _TinyLinear(2).to(aux_device)
    refiner = _TinyLinear(2).to(aux_device)

    model = SimpleNamespace(
        text_encoder=te,
        pipeline=None,
        device_torch=train_device,
        torch_dtype=torch.float32,
        unet=unet,
        model=unet,
        vae=vae,
        network=None,
        _sampling_transformer=None,
        _sampling_network=None,
        adapter=adapter,
        refiner_unet=refiner,
        image_encoder=None,
        _train_on_turbo=False,
    )

    def text_encoder_to(dev, *a, **k):
        live = model.text_encoder
        if isinstance(live, nn.Module) and not isinstance(live, FakeTextEncoder):
            live.to(dev)
        real = getattr(model, "_real_text_encoder", None)
        if isinstance(real, nn.Module):
            real.to(dev)

    model.text_encoder_to = text_encoder_to
    return model


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for aux snapshot restore")
def test_runtime_recache_restores_cuda_aux_owners_with_identity():
    """Runtime recache: CUDA vae/adapter/refiner park then restore; Parameter ids stable."""
    assert _CUDA is not None
    from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer

    model = _aux_owner_model(train_device=_CUDA, aux_device=_CUDA)
    unload_text_encoder(model)
    model.unet.to(_CUDA)
    for name in ("vae", "adapter", "refiner_unet"):
        getattr(model, name).to(_CUDA)

    ids_before = {
        name: {id(p) for p in getattr(model, name).parameters()}
        for name in ("vae", "adapter", "refiner_unet")
    }

    t = DiffusionTrainer.__new__(DiffusionTrainer)
    t.device_torch = _CUDA
    t.is_caching_text_embeddings = True
    t.train_config = SimpleNamespace(unload_text_encoder=False)
    t.sd = model
    seen: list = []

    def cache_spy():
        seen.append("cache")
        assert _all_on(model.vae, _CPU)
        assert _all_on(model.adapter, _CPU)
        assert _all_on(model.refiner_unet, _CPU)
        assert bool(getattr(model, "_text_cache_residency_active", False))
        assert isinstance(getattr(model, _TEXT_CACHE_AUX_RESTORE, None), dict)

    t.cache_sample_prompts = cache_spy
    t._recache_sample_prompts_runtime = (
        DiffusionTrainer._recache_sample_prompts_runtime.__get__(t, DiffusionTrainer)
    )
    t._recache_sample_prompts_runtime()

    assert seen == ["cache"]
    assert _all_on(model.vae, _CUDA)
    assert _all_on(model.adapter, _CUDA)
    assert _all_on(model.refiner_unet, _CUDA)
    for name, ids in ids_before.items():
        assert {id(p) for p in getattr(model, name).parameters()} == ids
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE, None) is None
    assert not bool(getattr(model, "_text_cache_residency_active", False))


def test_runtime_recache_cpu_aux_owners_no_extra_cuda_moves():
    """CPU pre-enter auxiliaries: exit restores CPU (no CUDA remount)."""
    train = _CUDA if _HAS_CUDA else _CPU
    model = _aux_owner_model(train_device=train, aux_device=_CPU)
    moves: list = []

    orig_vae_to = model.vae.to
    orig_adapter_to = model.adapter.to
    orig_refiner_to = model.refiner_unet.to

    def _spy(name, orig):
        def wrapped(*a, **k):
            moves.append((name, a, k))
            return orig(*a, **k)

        return wrapped

    model.vae.to = _spy("vae", orig_vae_to)
    model.adapter.to = _spy("adapter", orig_adapter_to)
    model.refiner_unet.to = _spy("refiner_unet", orig_refiner_to)

    enter_text_cache_residency(model)
    # Already CPU: offload may no-op (no .to) via safe_module_to_device early-out.
    unload_text_encoder(model)
    exit_text_cache_residency(model)

    assert _all_on(model.vae, _CPU)
    assert _all_on(model.adapter, _CPU)
    assert _all_on(model.refiner_unet, _CPU)
    cuda_moves = [
        m
        for m in moves
        if any(
            (isinstance(x, torch.device) and x.type == "cuda")
            or (isinstance(x, str) and "cuda" in x)
            for x in list(m[1]) + list(m[2].values())
        )
    ]
    assert cuda_moves == [], f"CPU aux must not be remounted to CUDA; got {cuda_moves!r}"
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE, None) is None


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for initial CPU aux contract")
def test_initial_pre_enter_cpu_aux_exit_does_not_remount_before_trainer():
    """Initial caching: pre-enter CPU vae/adapter/refiner stay CPU after common exit."""
    assert _CUDA is not None
    model = _aux_owner_model(train_device=_CUDA, aux_device=_CPU)
    enter_text_cache_residency(model)
    assert _all_on(model.vae, _CPU)
    assert _all_on(model.adapter, _CPU)
    assert _all_on(model.refiner_unet, _CPU)
    unload_text_encoder(model)
    exit_text_cache_residency(model)
    # Common exit restores snapshot (CPU); SDTrainer remount happens later.
    assert _all_on(model.vae, _CPU)
    assert _all_on(model.adapter, _CPU)
    assert _all_on(model.refiner_unet, _CPU)
    assert _all_on(model.unet, _CUDA)
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE, None) is None


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for abort snapshot cleanup")
def test_abort_clears_aux_snapshot_and_keeps_owners_cpu():
    assert _CUDA is not None
    model = _aux_owner_model(train_device=_CUDA, aux_device=_CUDA)
    enter_text_cache_residency(model)
    assert isinstance(getattr(model, _TEXT_CACHE_AUX_RESTORE, None), dict)
    abort_text_cache_residency(model)
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE, None) is None
    assert not bool(getattr(model, "_text_cache_residency_active", False))
    assert _all_on(model.vae, _CPU)
    assert _all_on(model.adapter, _CPU)
    assert _all_on(model.refiner_unet, _CPU)
    assert _all_on(model.unet, _CPU)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for exit-failure snapshot retain")
def test_exit_failure_keeps_aux_snapshot_for_retry(monkeypatch):
    """Failed exit keeps active + snapshot; successful retry restores and clears."""
    assert _CUDA is not None
    import toolkit.unloader as unloader

    model = _aux_owner_model(train_device=_CUDA, aux_device=_CUDA)
    enter_text_cache_residency(model)
    unload_text_encoder(model)
    snap = getattr(model, _TEXT_CACHE_AUX_RESTORE)
    assert isinstance(snap, dict)
    assert "vae" in snap

    real_restore = unloader._restore_aux_owners_from_snapshot
    calls = {"n": 0}

    def boom_then_ok(m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("aux restore boom")
        return real_restore(m)

    monkeypatch.setattr(unloader, "_restore_aux_owners_from_snapshot", boom_then_ok)

    with pytest.raises(RuntimeError, match="aux restore boom"):
        exit_text_cache_residency(model)

    assert bool(getattr(model, "_text_cache_residency_active", False))
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE) is snap

    exit_text_cache_residency(model)
    assert not bool(getattr(model, "_text_cache_residency_active", False))
    assert getattr(model, _TEXT_CACHE_AUX_RESTORE, None) is None
    assert _all_on(model.vae, _CUDA)
    assert _all_on(model.adapter, _CUDA)
    assert _all_on(model.refiner_unet, _CUDA)
