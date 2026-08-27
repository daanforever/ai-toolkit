import torch
from toolkit.basic import flush
from toolkit.util.debug import is_debug_enabled
from toolkit.util.device import devices_equal, quantized_payload_device, safe_module_to_device
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


# Lifecycle flag on the model instance (not save_device_state).
_TEXT_CACHE_RESIDENCY_ACTIVE = "_text_cache_residency_active"
# Pre-enter devices for primary auxiliary owners (set on successful enter).
_TEXT_CACHE_AUX_RESTORE = "_text_cache_aux_restore_snapshot"

# Primary auxiliaries offloaded on enter; restored on exit to exact pre-enter devices.
_TEXT_CACHE_AUX_SNAPSHOT_ATTRS = (
    "vae",
    "adapter",
    "refiner_unet",
    "image_encoder",
)

# Persistent non-TE owners for text-cache residency (dedupe by object id at use site).
# ``unet`` / ``model`` are aliases on BaseModel; listing both is safe with id-dedupe.
# ``image_encoder`` covers Wan21 I2V CLIP vision (not under adapter).
# ``assistant_adapter`` / ``taesd`` are trainer-owned; mirrored onto the model before enter.
# ``assistant_lora`` / ``accuracy_recovery_adapter`` are LoRASpecialNetwork modules.
# ``decorator`` / ``audio_processor`` are model-side Modules created before initial cache.
TEXT_CACHE_PERSISTENT_NON_TE_OWNER_ATTRS: Tuple[str, ...] = (
    "unet",
    "model",
    "vae",
    "network",
    "_sampling_transformer",
    "_sampling_network",
    "adapter",
    "refiner_unet",
    "image_encoder",
    "assistant_lora",
    "accuracy_recovery_adapter",
    "decorator",
    "audio_processor",
    "assistant_adapter",
    "taesd",
)


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        return self


def _is_fake_text_encoder(module: Any) -> bool:
    return isinstance(module, FakeTextEncoder)


def _iter_pipeline_real_text_encoders(pipe: Any) -> List[Tuple[str, torch.nn.Module]]:
    """Real (non-Fake) text encoders attached to a diffusion pipeline."""
    out: List[Tuple[str, torch.nn.Module]] = []
    if pipe is None:
        return out
    te = getattr(pipe, "text_encoder", None)
    if te is not None and not _is_fake_text_encoder(te) and isinstance(te, torch.nn.Module):
        out.append(("text_encoder", te))
    i = 2
    while hasattr(pipe, f"text_encoder_{i}"):
        te_i = getattr(pipe, f"text_encoder_{i}")
        if te_i is not None and not _is_fake_text_encoder(te_i) and isinstance(
            te_i, torch.nn.Module
        ):
            out.append((f"text_encoder_{i}", te_i))
        i += 1
    return out


def _collect_text_encoders_for_cpu_unload(
    model: Any,
) -> List[Tuple[str, torch.nn.Module]]:
    """Live model + pipeline real TEs for unload, deduplicated by object id."""
    seen: set[int] = set()
    out: List[Tuple[str, torch.nn.Module]] = []

    def _add(label: str, enc: Any) -> None:
        if enc is None or _is_fake_text_encoder(enc) or not isinstance(enc, torch.nn.Module):
            return
        oid = id(enc)
        if oid in seen:
            return
        seen.add(oid)
        out.append((label, enc))

    te = getattr(model, "text_encoder", None)
    if isinstance(te, list):
        for i, enc in enumerate(te):
            label = "text_encoder" if i == 0 else f"text_encoder[{i}]"
            _add(label, enc)
    elif te is not None:
        _add("text_encoder", te)

    pipe = getattr(model, "pipeline", None)
    for attr, enc in _iter_pipeline_real_text_encoders(pipe):
        _add(f"pipeline.{attr}", enc)
    return out


def _stash_pipeline_text_encoders(model: "BaseModel", pipe: Any) -> None:
    """Record pipeline TE references (already on CPU) so reload can restore them.

    Does not move modules — callers must CPU-offload first via validated movers.
    """
    if pipe is None or getattr(model, "_real_pipeline_text_encoders", None) is not None:
        return
    stashed = {}
    for attr, te in _iter_pipeline_real_text_encoders(pipe):
        stashed[attr] = te
    if stashed:
        model._real_pipeline_text_encoders = stashed


def unload_text_encoder(model: "BaseModel"):
    """Move all real TEs to CPU, then stash and install FakeTextEncoder.

    Order is mandatory: validated CPU moves for every live/pipeline real TE
    (id-deduped) must succeed before any Fake is installed. Partial move failure
    leaves real TEs attached so callers can abort/cleanup; the original error
    propagates with component/source/target context.
    """
    if model.text_encoder is None:
        flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return

    encoders = _collect_text_encoders_for_cpu_unload(model)
    if encoders:
        sources = {name: _module_source_device(enc) for name, enc in encoders}
        for name, enc in encoders:
            _move_owner_to_device(name, enc, "cpu")
        _assert_modules_on_device(encoders, torch.device("cpu"), sources)

    pipe = model.pipeline
    _stash_pipeline_text_encoders(model, pipe)

    if isinstance(model.text_encoder, list):
        # Stash real TEs once (do not overwrite with fakes on a second unload)
        if getattr(model, "_real_text_encoder", None) is None:
            real_list = [
                enc
                for enc in model.text_encoder
                if enc is not None and not _is_fake_text_encoder(enc)
            ]
            if real_list:
                model._real_text_encoder = real_list

        text_encoder_list = []
        if pipe is not None and hasattr(pipe, "text_encoder"):
            te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            text_encoder_list.append(te)
            pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
        # If pipeline is None (e.g. zimage_diffsynth) we still need at least one fake.
        if not text_encoder_list:
            text_encoder_list.append(
                FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
            )
        model.text_encoder = text_encoder_list
    else:
        if getattr(model, "_real_text_encoder", None) is None and not _is_fake_text_encoder(
            model.text_encoder
        ):
            model._real_text_encoder = model.text_encoder
        if pipe is not None and hasattr(pipe, "text_encoder"):
            pipe.text_encoder = FakeTextEncoder(
                device=model.device_torch, dtype=model.torch_dtype
            )
            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                setattr(
                    pipe,
                    f"text_encoder_{i}",
                    FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype),
                )
                i += 1
        model.text_encoder = FakeTextEncoder(
            device=model.device_torch, dtype=model.torch_dtype
        )

    flush()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reload_text_encoder(model: "BaseModel") -> None:
    """Restore stashed real text encoder(s) onto model and pipeline (still on CPU)."""
    real = getattr(model, "_real_text_encoder", None)
    if real is None:
        return

    pipe = getattr(model, "pipeline", None)
    pipe_stash = getattr(model, "_real_pipeline_text_encoders", None)

    if isinstance(real, list):
        model.text_encoder = list(real)
        if pipe is not None and pipe_stash:
            for attr, te in pipe_stash.items():
                setattr(pipe, attr, te)
        elif pipe is not None and hasattr(pipe, "text_encoder") and real:
            pipe.text_encoder = real[0]
            for i, te in enumerate(real[1:], start=2):
                attr = f"text_encoder_{i}"
                if hasattr(pipe, attr):
                    setattr(pipe, attr, te)
    else:
        model.text_encoder = real
        if pipe is not None and pipe_stash and "text_encoder" in pipe_stash:
            pipe.text_encoder = pipe_stash["text_encoder"]
        elif pipe is not None and hasattr(pipe, "text_encoder"):
            pipe.text_encoder = real


# ---------------------------------------------------------------------------
# Text-cache residency lifecycle (sole source of enter/exit semantics)
# ---------------------------------------------------------------------------


def _module_source_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        for buf in module.buffers():
            if buf is not None:
                return buf.device
    return torch.device("cpu")


def _module_needs_device_move(module: torch.nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for param in module.parameters():
        if not devices_equal(param.device, target):
            return True
        payload = quantized_payload_device(param)
        if payload is not None and not devices_equal(payload, target):
            return True
        for attr in ("qdata", "_data", "scale", "_scale"):
            val = getattr(param, attr, None)
            if isinstance(val, torch.Tensor) and not devices_equal(val.device, target):
                return True
    for buf in module.buffers():
        if buf is not None and not devices_equal(buf.device, target):
            return True
    return False


def _iter_network_tensor_modules(network: torch.nn.Module):
    """Yield network root plus LoRA modules held only in plain lists."""
    yield network
    for attr in ("unet_loras", "text_encoder_loras"):
        loras = getattr(network, attr, None)
        if not loras:
            continue
        for lora in loras:
            if isinstance(lora, torch.nn.Module):
                yield lora


def _network_source_device(network: torch.nn.Module) -> torch.device:
    for mod in _iter_network_tensor_modules(network):
        try:
            return next(mod.parameters()).device
        except StopIteration:
            for buf in mod.buffers():
                if buf is not None:
                    return buf.device
    return torch.device("cpu")


def _network_needs_device_move(network: torch.nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for mod in _iter_network_tensor_modules(network):
        if _module_needs_device_move(mod, target):
            return True
    return False


def _resolve_network_dtype(network: torch.nn.Module, model: Any = None) -> torch.dtype:
    for mod in _iter_network_tensor_modules(network):
        for param in mod.parameters():
            if param.requires_grad:
                return param.dtype
    for mod in _iter_network_tensor_modules(network):
        for param in mod.parameters():
            return param.dtype
    dtype = getattr(model, "torch_dtype", None) if model is not None else None
    return dtype if dtype is not None else torch.float32


def collect_persistent_non_te_owners(model: Any) -> Dict[str, torch.nn.Module]:
    """Named persistent non-TE owners, deduplicated by object id (shared aliases kept once).

    Extensibility: if the model defines ``iter_text_cache_extra_non_te_owners()``
    yielding ``(name, module)`` pairs, those are merged after the hardcoded attrs
    (still id-deduped). Use this for topology stages that are not top-level attrs.
    """
    seen: set[int] = set()
    out: Dict[str, torch.nn.Module] = {}

    def _add(name: str, obj: Any) -> None:
        if obj is None or not isinstance(obj, torch.nn.Module):
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        out[name] = obj

    for name in TEXT_CACHE_PERSISTENT_NON_TE_OWNER_ATTRS:
        _add(name, getattr(model, name, None))

    extra = getattr(model, "iter_text_cache_extra_non_te_owners", None)
    if callable(extra):
        for item in extra():
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    "iter_text_cache_extra_non_te_owners must yield (name, module) pairs"
                )
            _add(str(item[0]), item[1])
    return out


def _move_owner_to_device(name: str, module: torch.nn.Module, device: Union[torch.device, str]) -> None:
    """Move one owner with safe_module_to_device; never suppress errors."""
    target = torch.device(device)
    if not _module_needs_device_move(module, target):
        return
    source = _module_source_device(module)
    try:
        safe_module_to_device(module, target)
    except Exception as exc:
        raise RuntimeError(
            f"text-cache residency: failed moving component {name!r} "
            f"from {source} to {target}: {exc}"
        ) from exc
    if _module_needs_device_move(module, target):
        raise RuntimeError(
            f"text-cache residency: component {name!r} still not on {target} "
            f"after move (source was {source})"
        )


def _move_network_owner_to_device(
    name: str,
    network: torch.nn.Module,
    device: Union[torch.device, str],
    model: Any = None,
) -> None:
    """Move LoRA/PEFT network owners via ``force_to(device, dtype)`` when present.

    LoRASpecial keeps ``unet_loras`` / ``text_encoder_loras`` as plain lists (not
    ``nn.Module`` children); ``force_to`` exists specifically to relocate them.
    After ``force_to``, verify registered + list-held tensors/payloads; use
    ``safe_module_to_device`` only to finish payloads without replacing shared
    trainable Parameter identities.

    Always refresh ``torch_multiplier`` afterward: it is a plain tensor and can
    stay on the pre-move device when ``force_to`` is skipped because shared
    Parameters are already on ``target``.
    """
    target = torch.device(device)
    if not _network_needs_device_move(network, target):
        _refresh_one_network_torch_multiplier(network)
        return
    source = _network_source_device(network)
    dtype = _resolve_network_dtype(network, model)
    try:
        if hasattr(network, "force_to"):
            network.force_to(target, dtype)
        else:
            safe_module_to_device(network, target)
            for mod in _iter_network_tensor_modules(network):
                if mod is network:
                    continue
                if _module_needs_device_move(mod, target):
                    safe_module_to_device(mod, target)
    except Exception as exc:
        raise RuntimeError(
            f"text-cache residency: failed moving component {name!r} "
            f"from {source} to {target}: {exc}"
        ) from exc

    # Finish quantized payloads / any leftovers without swapping Parameter ids.
    try:
        for mod in _iter_network_tensor_modules(network):
            if _module_needs_device_move(mod, target):
                safe_module_to_device(mod, target)
    except Exception as exc:
        raise RuntimeError(
            f"text-cache residency: failed moving component {name!r} "
            f"from {source} to {target}: {exc}"
        ) from exc

    if _network_needs_device_move(network, target):
        raise RuntimeError(
            f"text-cache residency: component {name!r} still not on {target} "
            f"after move (source was {source})"
        )
    _refresh_one_network_torch_multiplier(network)


def _refresh_one_network_torch_multiplier(network: Any) -> None:
    """Rebuild classic/PEFT ``torch_multiplier`` from current weight device if supported."""
    updater = getattr(network, "_update_torch_multiplier", None)
    if callable(updater):
        updater()


def _refresh_network_torch_multipliers(model: Any) -> None:
    """Refresh ``torch_multiplier`` on every text-cache network owner.

    Needed after exit remount when ``_sampling_network`` shares Parameters with
    ``network`` and therefore skips ``force_to`` — its own multiplier tensor
    would otherwise remain on the creation device (CPU).
    """
    seen: set[int] = set()
    for name in _TEXT_CACHE_NETWORK_OWNER_ATTRS:
        network = getattr(model, name, None)
        if network is None or not isinstance(network, torch.nn.Module):
            continue
        oid = id(network)
        if oid in seen:
            continue
        seen.add(oid)
        _refresh_one_network_torch_multiplier(network)


def _iter_live_real_text_encoders(model: Any) -> List[Tuple[str, torch.nn.Module]]:
    """Real (non-Fake) text encoders currently attached to the model."""
    out: List[Tuple[str, torch.nn.Module]] = []
    te = getattr(model, "text_encoder", None)
    if te is None:
        return out
    if isinstance(te, list):
        for i, enc in enumerate(te):
            if enc is None or _is_fake_text_encoder(enc):
                continue
            if isinstance(enc, torch.nn.Module):
                label = "text_encoder" if i == 0 else f"text_encoder[{i}]"
                out.append((label, enc))
    elif not _is_fake_text_encoder(te) and isinstance(te, torch.nn.Module):
        out.append(("text_encoder", te))
    return out


def _iter_all_real_text_encoders(model: Any) -> List[Tuple[str, torch.nn.Module]]:
    """Live + stashed real TEs, deduplicated by object id."""
    seen: set[int] = set()
    out: List[Tuple[str, torch.nn.Module]] = []

    def _add(label: str, enc: Any) -> None:
        if enc is None or _is_fake_text_encoder(enc) or not isinstance(enc, torch.nn.Module):
            return
        oid = id(enc)
        if oid in seen:
            return
        seen.add(oid)
        out.append((label, enc))

    for label, enc in _iter_live_real_text_encoders(model):
        _add(label, enc)

    real = getattr(model, "_real_text_encoder", None)
    if isinstance(real, list):
        for i, enc in enumerate(real):
            _add(f"_real_text_encoder[{i}]", enc)
    elif real is not None:
        _add("_real_text_encoder", real)

    pipe_stash = getattr(model, "_real_pipeline_text_encoders", None)
    if isinstance(pipe_stash, dict):
        for attr, enc in pipe_stash.items():
            _add(f"_real_pipeline_text_encoders.{attr}", enc)

    return out


def _module_has_cuda_residency(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if p.device.type == "cuda":
            return True
        payload = quantized_payload_device(p)
        if payload is not None and payload.type == "cuda":
            return True
        for attr in ("qdata", "_data", "scale", "_scale"):
            val = getattr(p, attr, None)
            if isinstance(val, torch.Tensor) and val.device.type == "cuda":
                return True
    for b in module.buffers():
        if b is not None and b.device.type == "cuda":
            return True
    return False


def _any_real_te_on_cuda(model: Any) -> bool:
    for _name, enc in _iter_all_real_text_encoders(model):
        if _module_has_cuda_residency(enc):
            return True
    return False


def _module_cuda_footprint(module: torch.nn.Module) -> Tuple[int, str, str]:
    """Return (cuda_bytes, param_devices, payload_devices) for a module."""
    cuda_bytes = 0
    param_devs: set[str] = set()
    payload_devs: set[str] = set()
    seen: set[int] = set()

    def _add_tensor(t: Any) -> None:
        nonlocal cuda_bytes
        if not isinstance(t, torch.Tensor):
            return
        try:
            ptr = t.data_ptr()
        except Exception:
            ptr = id(t)
        if ptr in seen:
            return
        seen.add(ptr)
        if t.device.type == "cuda":
            cuda_bytes += t.numel() * t.element_size()
        for attr in ("qdata", "_data", "scale", "_scale"):
            val = getattr(t, attr, None)
            if not isinstance(val, torch.Tensor):
                continue
            try:
                vptr = val.data_ptr()
            except Exception:
                vptr = id(val)
            if vptr in seen:
                continue
            seen.add(vptr)
            if val.device.type == "cuda":
                cuda_bytes += val.numel() * val.element_size()
                payload_devs.add(str(val.device))

    for p in module.parameters():
        param_devs.add(str(p.device))
        _add_tensor(p)
        payload = quantized_payload_device(p)
        if payload is not None:
            payload_devs.add(str(payload))
    for buf in module.buffers():
        if buf is None:
            continue
        param_devs.add(str(buf.device))
        _add_tensor(buf)
    return (
        cuda_bytes,
        ",".join(sorted(param_devs)) or "none",
        ",".join(sorted(payload_devs)) or "none",
    )


def log_text_cache_cuda_residency(print_fn, model: Any, label: str) -> None:
    """Debug-only: per-component CUDA footprint after enter / first encode."""
    if not is_debug_enabled() or not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated()
    lines = [
        f"[DEBUG CUDA residency] {label} "
        f"alloc={alloc / 1024**3:.2f}GB reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
    ]
    accounted = 0
    owners: List[Tuple[str, torch.nn.Module]] = []
    owners.extend(_iter_live_real_text_encoders(model))
    for name, module in collect_persistent_non_te_owners(model).items():
        owners.append((name, module))
    seen_ids: set[int] = set()
    for name, module in owners:
        oid = id(module)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        nbytes, param_devs, payload_devs = _module_cuda_footprint(module)
        accounted += nbytes
        lines.append(
            f"  {name}: params={param_devs} payload={payload_devs} "
            f"cuda={nbytes / 1024**3:.2f}GB"
        )
    unexplained = alloc - accounted
    lines.append(
        f"  accounted={accounted / 1024**3:.2f}GB unexplained={unexplained / 1024**3:.2f}GB"
    )
    print_fn("\n".join(lines))



def _assert_modules_on_device(
    named_modules: List[Tuple[str, torch.nn.Module]],
    device: torch.device,
    sources: Dict[str, torch.device],
) -> None:
    target = torch.device(device)
    for name, module in named_modules:
        if _module_needs_device_move(module, target):
            source = sources.get(name, _module_source_device(module))
            raise RuntimeError(
                f"text-cache residency: component {name!r} still not on {target} "
                f"after move (source was {source})"
            )


def _move_real_text_encoders(model: Any, device: Union[torch.device, str]) -> None:
    """Move live real text encoders to device; errors include component name + devices."""
    target = torch.device(device)
    encoders = _iter_live_real_text_encoders(model)
    if not encoders:
        return

    sources = {name: _module_source_device(enc) for name, enc in encoders}

    # Prefer model.text_encoder_to when present (keeps list/pipeline wiring consistent).
    if hasattr(model, "text_encoder_to"):
        source = sources[encoders[0][0]]
        try:
            model.text_encoder_to(target)
        except Exception as exc:
            raise RuntimeError(
                f"text-cache residency: failed moving component 'text_encoder' "
                f"from {source} to {target}: {exc}"
            ) from exc
    else:
        for name, enc in encoders:
            _move_owner_to_device(name, enc, target)

    _assert_modules_on_device(encoders, target, sources)


# LoRASpecial / PEFT network attrs: list-held modules need ``force_to``.
_TEXT_CACHE_NETWORK_OWNER_ATTRS = frozenset(
    ("network", "_sampling_network", "assistant_lora", "accuracy_recovery_adapter")
)
# Secondary Modules remounted on exit (backbone / VAE / adapter handled elsewhere).
# ``assistant_lora`` stays CPU after exit (generate hooks force_to when active).
# ``audio_processor`` stays CPU (LTX2 mel path is CPU-native).
_TEXT_CACHE_SECONDARY_REMOUNT_ATTRS = (
    "accuracy_recovery_adapter",
    "decorator",
    "assistant_adapter",
    "taesd",
)


def _capture_aux_restore_snapshot(model: Any) -> Dict[str, torch.device]:
    """Pre-enter devices for primary auxiliaries; id-dedupe shared Module aliases."""
    seen: set[int] = set()
    out: Dict[str, torch.device] = {}
    for name in _TEXT_CACHE_AUX_SNAPSHOT_ATTRS:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        oid = id(module)
        if oid in seen:
            continue
        seen.add(oid)
        out[name] = _module_source_device(module)
    return out


def _clear_aux_restore_snapshot(model: Any) -> None:
    if hasattr(model, _TEXT_CACHE_AUX_RESTORE):
        delattr(model, _TEXT_CACHE_AUX_RESTORE)


def _restore_aux_owners_from_snapshot(model: Any) -> None:
    """Return primary auxiliaries to exact pre-enter devices from lifecycle snapshot."""
    snapshot = getattr(model, _TEXT_CACHE_AUX_RESTORE, None)
    if not isinstance(snapshot, dict) or not snapshot:
        return
    seen: set[int] = set()
    for name in _TEXT_CACHE_AUX_SNAPSHOT_ATTRS:
        if name not in snapshot:
            continue
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        oid = id(module)
        if oid in seen:
            continue
        seen.add(oid)
        _move_owner_to_device(name, module, snapshot[name])


def _offload_persistent_non_te_owners(model: Any, device: Union[torch.device, str] = "cpu") -> None:
    target = torch.device(device)
    owners = collect_persistent_non_te_owners(model)
    for name, module in owners.items():
        if name in _TEXT_CACHE_NETWORK_OWNER_ATTRS:
            _move_network_owner_to_device(name, module, target, model=model)
            continue
        # Prefer safe move first so quantized payloads / Parameter identity are
        # handled before model hooks that may use Module.to().
        if name in ("unet", "model") and hasattr(model, "_place_training_dit"):
            source = _module_source_device(module)
            _move_owner_to_device(name, module, target)
            try:
                model._place_training_dit(target)
            except Exception as exc:
                raise RuntimeError(
                    f"text-cache residency: failed moving component {name!r} "
                    f"from {source} to {target}: {exc}"
                ) from exc
            if _module_needs_device_move(module, target):
                _move_owner_to_device(name, module, target)
            continue
        if name == "_sampling_transformer" and hasattr(model, "_move_sampling_transformer"):
            source = _module_source_device(module)
            _move_owner_to_device(name, module, target)
            try:
                model._move_sampling_transformer(target)
            except Exception as exc:
                raise RuntimeError(
                    f"text-cache residency: failed moving component {name!r} "
                    f"from {source} to {target}: {exc}"
                ) from exc
            if _module_needs_device_move(module, target):
                _move_owner_to_device(name, module, target)
            continue
        _move_owner_to_device(name, module, target)


def _remount_secondary_owners_after_text_cache(
    model: Any,
    device: Union[torch.device, str],
) -> None:
    """Remount audited secondary owners after exit (enter parked them on CPU).

    Backbone (unet/network/sampling), VAE, and train adapter are restored by
    ``_restore_*_train_layout`` / trainer remount helpers — not here.
    """
    target = torch.device(device)
    for name in _TEXT_CACHE_SECONDARY_REMOUNT_ATTRS:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        if name in _TEXT_CACHE_NETWORK_OWNER_ATTRS:
            _move_network_owner_to_device(name, module, target, model=model)
        else:
            _move_owner_to_device(name, module, target)


def _prefer_turbo_restore(model: Any) -> bool:
    """Turbo restore when canonical ``_train_on_turbo`` intent is set."""
    return bool(getattr(model, "_train_on_turbo", False))


def _restore_normal_train_layout(model: Any, device: Union[torch.device, str]) -> None:
    target = torch.device(device)
    if hasattr(model, "_move_main_network"):
        # Capture before hook — model may already have owners on mixed devices.
        source = _network_source_device(getattr(model, "network", model.unet)) if isinstance(
            getattr(model, "network", None), torch.nn.Module
        ) else (
            _module_source_device(model.unet)
            if isinstance(getattr(model, "unet", None), torch.nn.Module)
            else torch.device("cpu")
        )
        try:
            model._move_main_network(target)
        except Exception as exc:
            raise RuntimeError(
                f"text-cache residency: failed moving component 'main_network' "
                f"from {source} to {target}: {exc}"
            ) from exc
    else:
        unet = getattr(model, "unet", None)
        if isinstance(unet, torch.nn.Module):
            _move_owner_to_device("unet", unet, target)
        network = getattr(model, "network", None)
        if isinstance(network, torch.nn.Module):
            _move_network_owner_to_device("network", network, target, model=model)

    # Normal layout: sampling transformer on CPU. Do not force sampling_network to
    # CPU — it may share LoRA/PEFT Parameter identity with the active main network.
    sampling = getattr(model, "_sampling_transformer", None)
    if isinstance(sampling, torch.nn.Module):
        if hasattr(model, "_move_sampling_transformer"):
            source = _module_source_device(sampling)
            try:
                model._move_sampling_transformer(torch.device("cpu"))
            except Exception as exc:
                raise RuntimeError(
                    f"text-cache residency: failed moving component '_sampling_transformer' "
                    f"from {source} to cpu: {exc}"
                ) from exc
            if _module_needs_device_move(sampling, torch.device("cpu")):
                _move_owner_to_device("_sampling_transformer", sampling, "cpu")
        else:
            _move_owner_to_device("_sampling_transformer", sampling, "cpu")


def _restore_turbo_train_layout(model: Any) -> None:
    if not hasattr(model, "apply_turbo_teacher_mode"):
        raise RuntimeError(
            "text-cache residency lifecycle: turbo restore requested but model has no "
            "apply_turbo_teacher_mode hook"
        )
    try:
        model.apply_turbo_teacher_mode(True)
    except Exception as exc:
        raise RuntimeError(
            f"text-cache residency: failed turbo restore via apply_turbo_teacher_mode: {exc}"
        ) from exc


def enter_text_cache_residency(
    model: Any,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    """Enter TE-only residency for text-embedding caching.

    Order:
      1) real TE -> CPU
      2) all persistent non-TE owners -> CPU (deduped)
      3) flush
      4) real TE -> target

    Idempotent: a second enter while already active is a no-op (no extra moves).
    Captures primary auxiliary pre-enter devices (vae/adapter/refiner/image_encoder)
    once on successful enter for exact restore on exit. Does not use
    ``save_device_state``. On any move failure the active flag and snapshot are
    not set (no partial lifecycle activate).
    """
    if bool(getattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, False)):
        return

    target = device
    if target is None:
        target = getattr(model, "device_torch", None)
    if target is None:
        target = torch.device("cpu")
    target = torch.device(target)

    # Capture before any moves so devices reflect the pre-enter layout.
    # Repeated enter while active returns above and does not overwrite.
    snapshot = _capture_aux_restore_snapshot(model)

    # TE off first so TE and backbone never co-reside during non-TE offload.
    _move_real_text_encoders(model, "cpu")
    _offload_persistent_non_te_owners(model, "cpu")
    flush()
    _move_real_text_encoders(model, target)

    setattr(model, _TEXT_CACHE_AUX_RESTORE, snapshot)
    setattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, True)


def exit_text_cache_residency(
    model: Any,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    """Restore normal or turbo train layout after TE unload/offload.

    Refuses to remount CUDA train owners while any live or stashed real TE remains
    on CUDA. Second exit while inactive is a documented no-op.
    Turbo when canonical ``_train_on_turbo`` is set and ``apply_turbo_teacher_mode``
    exists; otherwise normal (main/network active, sampling CPU). Primary
    auxiliaries return to exact pre-enter devices from the lifecycle snapshot.
    Success clears the snapshot; failure keeps active + snapshot for retry.
    """
    if not bool(getattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, False)):
        return

    if _any_real_te_on_cuda(model):
        raise RuntimeError(
            "text-cache residency lifecycle: cannot exit while a real text encoder "
            "remains on CUDA; unload or offload the text encoder first"
        )

    target = device
    if target is None:
        target = getattr(model, "device_torch", None)
    if target is None:
        # No train device: still restore auxiliaries from snapshot, then clear.
        _restore_aux_owners_from_snapshot(model)
        flush()
        setattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, False)
        _clear_aux_restore_snapshot(model)
        return

    # On failure leave active/snapshot for retry (TE stays off CUDA).
    if _prefer_turbo_restore(model):
        _restore_turbo_train_layout(model)
    else:
        _restore_normal_train_layout(model, target)
    _restore_aux_owners_from_snapshot(model)
    _remount_secondary_owners_after_text_cache(model, target)
    # Shared sampling/main LoRA may skip force_to; refresh plain torch_multiplier tensors.
    _refresh_network_torch_multipliers(model)
    flush()
    setattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, False)
    _clear_aux_restore_snapshot(model)


def abort_text_cache_residency(model: Any) -> None:
    """Encode/cache failure cleanup: TE and non-TE off CUDA, no train remount.

    Moves **live and stashed** real text encoders (deduped by id) to CPU with a
    postcondition check, then offloads persistent non-TE owners. Clears the
    residency flag and aux restore snapshot only after successful cleanup so a
    failed/no-op TE move cannot leave stashed CUDA TE while Fake is attached.
    Does not restore train layout.
    """
    target = torch.device("cpu")
    encoders = _iter_all_real_text_encoders(model)
    sources = {name: _module_source_device(enc) for name, enc in encoders}
    for name, enc in encoders:
        _move_owner_to_device(name, enc, target)
    _assert_modules_on_device(encoders, target, sources)
    if _any_real_te_on_cuda(model):
        raise RuntimeError(
            "abort_text_cache_residency: real text encoder still on CUDA after offload"
        )
    _offload_persistent_non_te_owners(model, "cpu")
    flush()
    setattr(model, _TEXT_CACHE_RESIDENCY_ACTIVE, False)
    _clear_aux_restore_snapshot(model)
