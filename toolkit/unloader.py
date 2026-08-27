import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING, Any, List, Optional, Union


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


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


def _move_te_to_cpu(text_encoder: Union[torch.nn.Module, List[torch.nn.Module], None]) -> None:
    if text_encoder is None:
        return
    if isinstance(text_encoder, list):
        for encoder in text_encoder:
            if encoder is None or _is_fake_text_encoder(encoder):
                continue
            try:
                encoder.to("cpu")
            except Exception:
                pass
    else:
        if _is_fake_text_encoder(text_encoder):
            return
        try:
            text_encoder.to("cpu")
        except Exception:
            pass


def _stash_pipeline_text_encoders(model: "BaseModel", pipe: Any) -> None:
    """Keep pipeline TE references (on CPU) so reload can restore them."""
    if pipe is None or getattr(model, "_real_pipeline_text_encoders", None) is not None:
        return
    stashed = {}
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        if not _is_fake_text_encoder(pipe.text_encoder):
            try:
                pipe.text_encoder.to("cpu")
            except Exception:
                pass
            stashed["text_encoder"] = pipe.text_encoder
    i = 2
    while hasattr(pipe, f"text_encoder_{i}"):
        te = getattr(pipe, f"text_encoder_{i}")
        if te is not None and not _is_fake_text_encoder(te):
            try:
                te.to("cpu")
            except Exception:
                pass
            stashed[f"text_encoder_{i}"] = te
        i += 1
    if stashed:
        model._real_pipeline_text_encoders = stashed


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.
    #
    # Order: TE.to("cpu") first, then stash CPU modules, then install FakeTextEncoder.
    # Stash is a Python/RAM keep-alive only — it does not keep TE on CUDA.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            # Move all existing encoders to CPU so GPU memory is actually freed
            # (required when pipeline is None, e.g. zimage_diffsynth; also ensures
            # text_encoder_2, text_encoder_3 etc. are off GPU when using pipeline)
            _move_te_to_cpu(model.text_encoder)

            pipe = model.pipeline
            _stash_pipeline_text_encoders(model, pipe)

            # Stash real TEs once (do not overwrite with fakes on a second unload)
            if getattr(model, "_real_text_encoder", None) is None:
                real_list = [
                    enc for enc in model.text_encoder
                    if enc is not None and not _is_fake_text_encoder(enc)
                ]
                if real_list:
                    model._real_text_encoder = real_list

            text_encoder_list = []

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
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
            # If pipeline is None (e.g. zimage_diffsynth) we still need at least one fake so text_encoder[0] doesn't raise.
            if not text_encoder_list:
                text_encoder_list.append(FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype))
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder — move to CPU before replacing
            _move_te_to_cpu(model.text_encoder)
            pipe = model.pipeline
            _stash_pipeline_text_encoders(model, pipe)
            if getattr(model, "_real_text_encoder", None) is None and not _is_fake_text_encoder(model.text_encoder):
                model._real_text_encoder = model.text_encoder
            if pipe is not None and hasattr(pipe, "text_encoder"):
                pipe.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                i = 2
                while hasattr(pipe, f"text_encoder_{i}"):
                    setattr(
                        pipe,
                        f"text_encoder_{i}",
                        FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype),
                    )
                    i += 1
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)

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


def park_main_transformer_for_text_cache(model: "BaseModel") -> None:
    """Move TE then main transformer to CPU so they never co-reside during DiT offload.

    Order matters: quantized DiT CUDA→CPU while TE is still on GPU doubles CUDA
    peak (``max_memory_allocated``). Caller should then apply
    ``set_device_state_preset('cache_text_encoder')`` so TE returns to GPU with
    DiT already parked.
    """
    if hasattr(model, "text_encoder_to") and model.text_encoder is not None:
        try:
            model.text_encoder_to("cpu")
        except Exception:
            _move_te_to_cpu(model.text_encoder)
    else:
        _move_te_to_cpu(getattr(model, "text_encoder", None))
    flush()

    if hasattr(model, "_place_training_dit"):
        model._place_training_dit("cpu")
    elif getattr(model, "unet", None) is not None:
        model.unet.to("cpu")
    flush()


def restore_main_transformer_after_text_cache(
    model: "BaseModel",
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    """Put training transformer back on train device after TE unload.

    When turbo teacher is preferred (``_prefer_turbo_restore_after_te`` or
    ``_train_on_turbo``), exclusive-pin sampling DiT instead of remounting the
    base DiT onto CUDA.
    """
    prefer_turbo = bool(getattr(model, "_prefer_turbo_restore_after_te", False)) or bool(
        getattr(model, "_train_on_turbo", False)
    )
    if prefer_turbo and hasattr(model, "apply_turbo_teacher_mode"):
        model.apply_turbo_teacher_mode(True)
        if hasattr(model, "_prefer_turbo_restore_after_te"):
            model._prefer_turbo_restore_after_te = False
        flush()
        return

    target = device
    if target is None:
        target = getattr(model, "device_torch", None)
    if target is None:
        return

    if hasattr(model, "_move_main_network"):
        model._move_main_network(target)
    elif getattr(model, "unet", None) is not None:
        model.unet.to(target)
    flush()
