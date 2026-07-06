"""PEFT-backed LoRA/DoRA network for the z_image_diffsynth arch.

Provides a `PeftNetwork` that wraps `peft.get_peft_model` and exposes the
surface that `BaseSDTrainProcess` and the z-image trainer expect from a
network (apply_to, prepare_optimizer_params, save_weights, load_weights,
share_parameters_with, force_to, multiplier/is_active/is_merged_in flags).

Scope (phase 1): basic LoRA and DoRA on quantized or unquantized Z-Image DiT.
Slider-training features (network_weight batch-split multiplier, magnitude
calibration) are deferred to phase 2.
"""

from __future__ import annotations

import os
import re
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from toolkit.network_mixins import ToolkitNetworkMixin

try:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from peft.tuners.lora.layer import LoraLayer
except ImportError as e:  # pragma: no cover - peft is in requirements.txt
    raise ImportError("peft is required for PeftNetwork: pip install peft") from e


_PEFT_BASE_PREFIX = "base_model.model."
# Linear class names we attach LoRA to. Matches toolkit LINEAR_MODULES so that
# quanto-quantized QLinear layers are also picked up.
_LINEAR_CLASS_NAMES = {"Linear", "QLinear", "LoRACompatibleLinear"}
# Module class names whose children we want to attach LoRA to. For z-image this
# comes from ZImageDiffSynthModel.target_lora_modules (["Attention","FeedForward"]).
_DEFAULT_TARGET_PARENT_CLASSES: Tuple[str, ...] = ("Attention", "FeedForward")


def _resolve_target_modules(
    root: nn.Module,
    target_parent_classes: Optional[List[str]],
) -> List[str]:
    """Return a list of LoRA target module suffixes for PEFT.

    PEFT matches ``key.endswith(f".{target_key}")`` for each ``target_key`` in
    the list (literal suffix match, not regex). We return one entry per
    (parent, child) pair using the child's path relative to its target parent
    so matches are unique even when leaf names collide (e.g. ``to_out.0`` vs
    ``layers.0``).
    """
    parents = tuple(target_parent_classes or list(_DEFAULT_TARGET_PARENT_CLASSES))
    names: set = set()
    for parent_name, parent_module in root.named_modules():
        if parent_module.__class__.__name__ not in parents:
            continue
        for child_name, child_module in parent_module.named_modules():
            if child_module.__class__.__name__ not in _LINEAR_CLASS_NAMES:
                continue
            if not child_name:
                continue
            # child_name is relative to the parent (e.g. "to_q", "to_out.0").
            # PEFT will check `<full_dotted_path>.endswith(".to_out.0")` which
            # uniquely identifies this Linear and does not match `layers.0`.
            names.add(child_name)
    if not names:
        names = {"Linear"}
    return sorted(names)


class _PeftLoraAdapter:
    """Thin adapter exposing `.lora_name` and `.named_parameters()` for a PEFT
    LoraLayer so the existing z-image param-grouping helpers
    (parse_lora_block_key / group_loras_by_block) keep working unchanged.
    """

    __slots__ = ("lora_name", "layer", "_param_cache")

    def __init__(self, lora_name: str, layer: LoraLayer) -> None:
        self.lora_name = lora_name
        self.layer = layer
        self._param_cache: Optional[List[Tuple[str, nn.Parameter]]] = None

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        if self._param_cache is None:
            # Only expose the trainable LoRA params (lora_A/lora_B/magnitude),
            # not the frozen base_layer weights. This matches what
            # toolkit.kohya_lora.prepare_optimizer_params expects.
            params: List[Tuple[str, nn.Parameter]] = []
            for adapter_name in self.layer.lora_A:
                a = self.layer.lora_A[adapter_name]
                b = self.layer.lora_B[adapter_name]
                params.append((f"lora_A.{adapter_name}.weight", a.weight))
                params.append((f"lora_B.{adapter_name}.weight", b.weight))
            mag = getattr(self.layer, "lora_magnitude_vector", None)
            if mag is not None:
                for adapter_name in mag:
                    if mag[adapter_name] is None:
                        continue
                    # magnitude vector is an nn.Parameter stored on .weight on
                    # the MagnitudeLayer; fall back to named_parameters.
                    for pname, p in mag[adapter_name].named_parameters():
                        if "magnitude" not in pname:
                            pname = f"magnitude.{pname}"
                        params.append((pname, p))
            self._param_cache = params
        return list(self._param_cache)

    def to(self, *args, **kwargs):
        return self.layer.to(*args, **kwargs)


def _peft_path_to_lora_name(peft_key: str) -> str:
    """Convert a PEFT state_dict key path to the toolkit lora_name convention.

    Input:  "base_model.model._inner_dit.layers.0.attention.to_q.lora_A.default.weight"
    Output: "transformer$$_inner_dit$$layers$$0$$attention$$to_q"
    """
    key = peft_key
    if key.startswith(_PEFT_BASE_PREFIX):
        key = key[len(_PEFT_BASE_PREFIX):]
    # strip the trailing lora_A/lora_B/lora_magnitude_vector.<adapter>.weight
    for tail in (".lora_A", ".lora_B", ".lora_magnitude_vector"):
        idx = key.find(tail + ".")
        if idx >= 0:
            key = key[:idx]
            break
    return "transformer$$" + key.replace(".", "$$")


def _lora_name_to_peft_path(lora_name: str) -> str:
    """Inverse of _peft_path_to_lora_name for load path."""
    if lora_name.startswith("transformer$$"):
        rest = lora_name[len("transformer$$"):]
        return rest.replace("$$", ".")
    if lora_name.startswith("lora_transformer__"):
        rest = lora_name[len("lora_transformer__"):]
        return rest.replace("_", ".")
    if lora_name.startswith("lora_unet__"):
        rest = lora_name[len("lora_unet__"):]
        return rest.replace("_", ".")
    return lora_name.replace("$$", ".").replace("__", ".")


class PeftNetwork(ToolkitNetworkMixin, nn.Module):
    """Network wrapper that delegates LoRA/DoRA adaptation to the `peft` library.

    Construction applies the adapter in-place via `get_peft_model`; `apply_to`
    is a no-op kept for API compatibility with BaseSDTrainProcess.
    """

    def __init__(
        self,
        text_encoder=None,
        unet: Optional[nn.Module] = None,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class=None,
        train_text_encoder: bool = False,
        train_unet: bool = True,
        use_bias: bool = False,
        is_sdxl: bool = False,
        is_v2: bool = False,
        is_v3: bool = False,
        is_pixart: bool = False,
        is_auraflow: bool = False,
        is_flux: bool = False,
        is_lumina2: bool = False,
        is_ssd: bool = False,
        is_vega: bool = False,
        network_config=None,
        network_type: str = "peft",
        transformer_only: bool = True,
        is_transformer: bool = True,
        base_model=None,
        ephemeral_lora: bool = False,
        deferred_lora_init: bool = False,
        target_lin_modules: Optional[List[str]] = None,
        peft_native_keys: bool = False,
        **kwargs,
    ) -> None:
        # Initialise nn.Module first so attribute assignment works.
        nn.Module.__init__(self)
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_sdxl=is_sdxl,
            is_v2=is_v2,
            is_ssd=is_ssd,
            is_vega=is_vega,
            network_config=network_config,
        )

        if unet is None:
            raise ValueError("PeftNetwork requires a unet (the DiT wrapper) to wrap")

        self.network_type = network_type.lower()
        if self.network_type not in ("peft", "peft_dora"):
            raise ValueError(f"PeftNetwork does not support network_type={network_type!r}")
        self.is_dora = self.network_type == "peft_dora"
        self.is_transformer = is_transformer
        self.is_pixart = is_pixart
        self.is_v3 = is_v3
        self.is_flux = is_flux
        self.is_lumina2 = is_lumina2
        self.is_auraflow = is_auraflow
        self.ephemeral_lora = ephemeral_lora
        self.deferred_lora_init = deferred_lora_init
        self.peft_native_keys = peft_native_keys

        # ToolkitNetworkMixin sets these defaults; we override where needed.
        self.module_class = None  # not used; PEFT manages its own LoraLayer
        self.full_rank = False
        self.peft_format = True  # emit lora_A/lora_B key naming
        self.can_merge_in = False  # never merge into a (possibly quantized) base via toolkit
        self.did_change_weights = False
        self.use_old_lokr_format = False
        self.block_lr = False

        # Base model weakref so save/load can call convert_lora_weights_before_*
        self.base_model_ref = weakref.ref(base_model) if base_model is not None else None
        self._multiplier = multiplier

        # Resolve target modules from the unet (DiT) and the base model's
        # target_lora_modules list (["Attention","FeedForward"] for z-image).
        parent_classes = target_lin_modules
        if parent_classes is None and base_model is not None and getattr(base_model, "target_lora_modules", None):
            parent_classes = list(base_model.target_lora_modules)
        target_modules = _resolve_target_modules(unet, parent_classes)

        # Build the PEFT LoRA config. lora_alpha controls the scaling; PEFT
        # applies scaling = lora_alpha / r internally so we do NOT fold alpha
        # manually on save.
        lora_alpha = alpha if alpha not in (None, 0) else lora_dim
        lora_config = LoraConfig(
            r=int(lora_dim),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(dropout) if dropout else 0.0,
            target_modules=target_modules,
            bias="none",
            use_dora=self.is_dora,
            # Do not specify task_type: the DiT is not a transformers ModelMixin.
        )

        # Apply in-place. get_peft_model wraps the base module and freezes it.
        # We pass get_apply_tensor_subclass=lambda: weights.config if we have a quantized model to satisfy PEFT's TorchaoLoraLinear
        kwargs_peft = {}
        if base_model is not None and hasattr(base_model, "model_config") and base_model.model_config.qtype is not None:
            from toolkit.util.quantize import get_qtype, aotype
            qtype_obj = get_qtype(base_model.model_config.qtype)
            if isinstance(qtype_obj, aotype):
                kwargs_peft["get_apply_tensor_subclass"] = lambda: qtype_obj.config

        peft_model = get_peft_model(unet, lora_config, **kwargs_peft)
        self.peft_model = peft_model
        # Keep a handle on the wrapped base for force_to / device moves.
        self._unet = unet

        # Optional text-encoder adapter (separate PeftModel on the TE).
        self.te_peft_model = None
        if train_text_encoder and text_encoder is not None:
            te_target = _resolve_target_modules(text_encoder, None)
            te_config = LoraConfig(
                r=int(lora_dim),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(dropout) if dropout else 0.0,
                target_modules=te_target,
                bias="none",
                use_dora=self.is_dora,
            )
            self.te_peft_model = get_peft_model(text_encoder, te_config, **kwargs_peft)

        # Build unet_loras / text_encoder_loras as wrapper lists so the existing
        # trainer helpers (group_loras_by_block, get_lora_optimizer_param_groups)
        # keep working.
        self.unet_loras: List[_PeftLoraAdapter] = self._collect_lora_adapters(peft_model)
        self.text_encoder_loras: List[_PeftLoraAdapter] = (
            self._collect_lora_adapters(self.te_peft_model) if self.te_peft_model is not None else []
        )

        # Initialise torch_multiplier to a sensible default.
        self.torch_multiplier = torch.tensor([float(multiplier)])

    # ------------------------------------------------------------------ helpers
    def _collect_lora_adapters(self, peft_model: nn.Module) -> List[_PeftLoraAdapter]:
        adapters: List[_PeftLoraAdapter] = []
        if peft_model is None:
            return adapters
        for name, module in peft_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            lora_name = _peft_path_to_lora_name(name)
            adapters.append(_PeftLoraAdapter(lora_name, module))
        return adapters

    def get_all_modules(self) -> List[_PeftLoraAdapter]:
        return list(self.unet_loras) + list(self.text_encoder_loras)

    # ----------------------------------------------------------- API surface
    def apply_to(self, text_encoder=None, unet=None, apply_text_encoder: bool = True, apply_unet: bool = True) -> None:
        # PEFT applies the adapter in-place during __init__, so this is a no-op.
        # Kept for compatibility with BaseSDTrainProcess which always calls it.
        return None

    def prepare_grad_etc(self, text_encoder=None, unet=None) -> None:
        # PEFT already set requires_grad on lora params during get_peft_model;
        # ensure the base is frozen.
        for p in self.peft_model.parameters():
            if not any(p is lp for lp in self._trainable_params()):
                p.requires_grad_(False)
        if self.te_peft_model is not None:
            for p in self.te_peft_model.parameters():
                p.requires_grad_(False)
            for p in self._trainable_params_te():
                p.requires_grad_(True)

    def _trainable_params(self):
        return list(self.peft_model.parameters())

    def _trainable_params_te(self):
        return [] if self.te_peft_model is None else list(self.te_peft_model.parameters())

    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad:
            # only LoRA params should be trainable
            for p in self.peft_model.parameters():
                p.requires_grad_(False)
            for p in self._trainable_params():
                p.requires_grad_(True)
            if self.te_peft_model is not None:
                for p in self.te_peft_model.parameters():
                    p.requires_grad_(False)
                for p in self._trainable_params_te():
                    p.requires_grad_(True)
        else:
            for p in self.peft_model.parameters():
                p.requires_grad_(False)
            if self.te_peft_model is not None:
                for p in self.te_peft_model.parameters():
                    p.requires_grad_(False)
        return self

    def force_to(self, device, dtype):
        self.peft_model.to(device, dtype)
        if self.te_peft_model is not None:
            self.te_peft_model.to(device, dtype)
        self.torch_multiplier = self.torch_multiplier.to(device, dtype)

    def to(self, *args, **kwargs):
        self.peft_model.to(*args, **kwargs)
        if self.te_peft_model is not None:
            self.te_peft_model.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.peft_model.train(mode)
        if self.te_peft_model is not None:
            self.te_peft_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse: bool = True):
        # Only expose trainable params so optimizer construction doesn't pick
        # up the frozen base weights.
        return iter(self._trainable_params())

    def state_dict(self, *args, **kwargs):
        sd = self.peft_model.state_dict(*args, **kwargs)
        if self.te_peft_model is not None:
            for k, v in self.te_peft_model.state_dict(*args, **kwargs).items():
                sd[f"te.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        te_keys = [k for k in state_dict if k.startswith("te.")]
        if te_keys and self.te_peft_model is not None:
            te_sd = {k[len("te."):]: v for k, v in state_dict.items() if k.startswith("te.")}
            rest = {k: v for k, v in state_dict.items() if not k.startswith("te.")}
            self.te_peft_model.load_state_dict(te_sd, strict=False)
            return self.peft_model.load_state_dict(rest, strict=strict)
        return self.peft_model.load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------- multiplier
    @property
    def multiplier(self) -> Union[float, List[float]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float]]):
        if self._multiplier == value:
            return
        self._multiplier = value
        self._update_torch_multiplier()

    @torch.no_grad()
    def _update_torch_multiplier(self):
        multiplier = self._multiplier
        try:
            device = self.torch_multiplier.device
            dtype = self.torch_multiplier.dtype
        except AttributeError:
            device = torch.device("cpu")
            dtype = torch.float32
        if isinstance(multiplier, (int, float)):
            tensor_multiplier = torch.tensor((float(multiplier),)).to(device, dtype=dtype)
        elif isinstance(multiplier, list):
            tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
        elif isinstance(multiplier, torch.Tensor):
            tensor_multiplier = multiplier.clone().detach().to(device, dtype=dtype)
        else:
            tensor_multiplier = torch.tensor((1.0,)).to(device, dtype=dtype)
        self.torch_multiplier = tensor_multiplier

    # ------------------------------------------------------- optimizer params
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None, learning_rate=None):
        lr_value = unet_lr if unet_lr is not None else (default_lr if default_lr is not None else learning_rate)
        te_lr_value = text_encoder_lr if text_encoder_lr is not None else lr_value

        all_params: List[Dict[str, Any]] = []

        # Text encoder group (optional).
        if self.text_encoder_loras:
            te_lora_params: List[nn.Parameter] = []
            te_mag_params: List[nn.Parameter] = []
            for adapter in self.text_encoder_loras:
                for pname, p in adapter.named_parameters():
                    if "magnitude" in pname:
                        te_mag_params.append(p)
                    else:
                        te_lora_params.append(p)
            if te_lora_params:
                g = {"params": te_lora_params}
                if te_lr_value is not None:
                    g["lr"] = te_lr_value
                all_params.append(g)
            if te_mag_params:
                mg = {"params": te_mag_params, "is_magnitude": True}
                if te_lr_value is not None:
                    mg["lr"] = te_lr_value
                all_params.append(mg)

        # UNet groups: defer to the base model's custom grouping if available
        # (z-image splits by DiT block for adafactor stochastic accumulation).
        base_model = self.base_model_ref() if self.base_model_ref is not None else None
        if base_model is not None and hasattr(base_model, "get_lora_optimizer_param_groups") and self.unet_loras:
            custom_groups = base_model.get_lora_optimizer_param_groups(self, lr_value, default_lr)
            if custom_groups is not None:
                all_params.extend(custom_groups)
                return all_params

        # Fallback: a single lora group + optional magnitude group.
        unet_lora_params: List[nn.Parameter] = []
        unet_mag_params: List[nn.Parameter] = []
        for adapter in self.unet_loras:
            for pname, p in adapter.named_parameters():
                if "magnitude" in pname:
                    unet_mag_params.append(p)
                else:
                    unet_lora_params.append(p)
        if unet_lora_params:
            g = {"params": unet_lora_params}
            if lr_value is not None:
                g["lr"] = lr_value
            all_params.append(g)
        if unet_mag_params:
            mg = {"params": unet_mag_params, "is_magnitude": True}
            if lr_value is not None:
                mg["lr"] = lr_value
            all_params.append(mg)

        return all_params

    # ---------------------------------------------------------- save / load
    def get_state_dict(self, extra_state_dict=None, dtype=torch.float16) -> "OrderedDict[str, torch.Tensor]":
        sd = get_peft_model_state_dict(self.peft_model)
        save_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key, value in sd.items():
            v = value.detach().clone().to("cpu").to(dtype)
            save_dict[key] = v

        if self.te_peft_model is not None:
            te_sd = get_peft_model_state_dict(self.te_peft_model)
            for key, value in te_sd.items():
                save_dict[f"te.{key}"] = value.detach().clone().to("cpu").to(dtype)

        if extra_state_dict is not None:
            for key, value in extra_state_dict.items():
                save_dict[key] = value.detach().clone().to("cpu").to(dtype)

        if not self.peft_native_keys:
            save_dict = self._convert_peft_to_diffsynth(save_dict)

        # Let the base model apply its own key remap (e.g. transformer. ->
        # diffusion_model. for z-image DiffSynth inference convention).
        if self.base_model_ref is not None and not self.peft_native_keys:
            base = self.base_model_ref()
            if base is not None and hasattr(base, "convert_lora_weights_before_save"):
                save_dict = base.convert_lora_weights_before_save(save_dict)

        return save_dict

    def save_weights(self, file, dtype=torch.float16, metadata=None, extra_state_dict: Optional[OrderedDict] = None):
        save_dict = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype)
        from toolkit.metadata import add_model_hash_to_meta

        if metadata is not None and len(metadata) == 0:
            metadata = None
        if metadata is None:
            metadata = OrderedDict()
        metadata = add_model_hash_to_meta(save_dict, metadata)

        # Delegate to base_model.save_lora when present (matches ToolkitNetworkMixin).
        if self.base_model_ref is not None:
            base = self.base_model_ref()
            if base is not None and hasattr(base, "save_lora"):
                base.save_lora(save_dict, file, metadata)
                return

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(save_dict, file, metadata)
        else:
            torch.save(save_dict, file)

    def load_weights(self, file, force_weight_mapping: bool = False):
        from safetensors.torch import load_file as _load_safetensors

        if isinstance(file, str):
            if self.base_model_ref is not None:
                base = self.base_model_ref()
                if base is not None and hasattr(base, "load_lora"):
                    weights_sd = base.load_lora(file)
                elif os.path.splitext(file)[1] == ".safetensors":
                    weights_sd = _load_safetensors(file)
                else:
                    weights_sd = torch.load(file, map_location="cpu")
            else:
                if os.path.splitext(file)[1] == ".safetensors":
                    weights_sd = _load_safetensors(file)
                else:
                    weights_sd = torch.load(file, map_location="cpu")
        else:
            weights_sd = file

        if self.base_model_ref is not None:
            base = self.base_model_ref()
            if base is not None and hasattr(base, "convert_lora_weights_before_load"):
                weights_sd = base.convert_lora_weights_before_load(weights_sd)

        if not self.peft_native_keys:
            weights_sd = self._convert_diffsynth_to_peft(weights_sd)

        # Split off text-encoder keys if present.
        te_keys = [k for k in weights_sd if k.startswith("te.")]
        if te_keys and self.te_peft_model is not None:
            te_sd = {k[len("te."):]: v for k, v in weights_sd.items() if k.startswith("te.")}
            rest = {k: v for k, v in weights_sd.items() if not k.startswith("te.")}
            try:
                set_peft_model_state_dict(self.te_peft_model, te_sd)
            except Exception as e:
                print(f"[PeftNetwork] text-encoder load warning: {e}")
            set_peft_model_state_dict(self.peft_model, rest)
        else:
            set_peft_model_state_dict(self.peft_model, weights_sd)
        return None

    def _convert_peft_to_diffsynth(self, sd: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        """Convert native PEFT state_dict keys to the toolkit peft_format convention.

        `get_peft_model_state_dict` already strips the adapter name segment, so
        input keys look like:
            base_model.model._inner_dit.layers.0.attention.to_q.lora_A.weight
        We emit the toolkit/DiffSynth convention:
            transformer._inner_dit.layers.0.attention.to_q.lora_A.weight

        This matches what `LoRASpecialNetwork.get_state_dict` produces with
        `peft_format=True` so the existing DiffSynth inference loader and
        `convert_lora_weights_before_save` (transformer. -> diffusion_model.)
        work unchanged.
        """
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key, value in sd.items():
            if key.startswith("te."):
                out[key] = value
                continue
            new_key = key
            if new_key.startswith(_PEFT_BASE_PREFIX):
                new_key = new_key[len(_PEFT_BASE_PREFIX):]
            new_key = "transformer." + new_key
            out[new_key] = value
        return out

    def _convert_diffsynth_to_peft(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inverse of _convert_peft_to_diffsynth.

        Input keys (after base_model.convert_lora_weights_before_load):
            transformer._inner_dit.layers.0.attention.to_q.lora_A.weight
        Output (PEFT native, no adapter name; set_peft_model_state_dict inserts it):
            base_model.model._inner_dit.layers.0.attention.to_q.lora_A.weight
        """
        out: Dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith("te."):
                out[key] = value
                continue
            new_key = key
            if new_key.startswith("transformer."):
                path = new_key[len("transformer."):]
            elif new_key.startswith("transformer$$"):
                path = new_key[len("transformer$$"):].replace("$$", ".")
            elif new_key.startswith("lora_transformer__"):
                path = new_key[len("lora_transformer__"):].replace("_", ".")
            elif new_key.startswith("lora_unet__"):
                path = new_key[len("lora_unet__"):].replace("_", ".")
            else:
                path = new_key.replace("$$", ".").replace("__", ".")
            new_key = _PEFT_BASE_PREFIX + path
            out[new_key] = value
        return out

    # --------------------------------------------------- sampling transformer
    def share_parameters_with(self, other: "PeftNetwork") -> None:
        """Share LoRA parameters by reference so training updates propagate to
        the sampling network live (matches LoRASpecialNetwork semantics).

        Callers invoke ``sampling_network.share_parameters_with(main_network)``,
        so ``self`` is the sampling network that should adopt ``other``'s
        (the main network's) parameters by reference.
        """
        if not isinstance(other, PeftNetwork):
            raise TypeError("PeftNetwork.share_parameters_with expects a PeftNetwork")
        sampling_layers = {a.layer: a for a in self.unet_loras}
        main_layers = {a.layer: a for a in other.unet_loras}
        if len(sampling_layers) != len(main_layers):
            raise AssertionError(
                f"lora count mismatch: sampling={len(sampling_layers)} main={len(main_layers)}"
            )

        # Pair layers by lora_name so structure must match exactly.
        sampling_by_name = {a.lora_name: a for a in self.unet_loras}
        main_by_name = {a.lora_name: a for a in other.unet_loras}
        if set(sampling_by_name.keys()) != set(main_by_name.keys()):
            missing = set(sampling_by_name.keys()) ^ set(main_by_name.keys())
            raise AssertionError(f"lora name mismatch: {missing}")

        # Sampling (self) adopts main's (other's) parameters by reference so
        # training updates on the main network propagate live to sampling.
        for name, sampling_adapter in sampling_by_name.items():
            main_adapter = main_by_name[name]
            sampling_layer = sampling_adapter.layer
            main_layer = main_adapter.layer
            for adapter_name in main_layer.lora_A:
                sampling_layer.lora_A[adapter_name].weight = main_layer.lora_A[adapter_name].weight
                sampling_layer.lora_B[adapter_name].weight = main_layer.lora_B[adapter_name].weight
            main_mag = getattr(main_layer, "lora_magnitude_vector", None)
            sampling_mag = getattr(sampling_layer, "lora_magnitude_vector", None)
            if main_mag is not None and sampling_mag is not None:
                for adapter_name in main_mag:
                    if main_mag[adapter_name] is None or sampling_mag[adapter_name] is None:
                        continue
                    # Magnitude is stored as a Parameter on the Magnitude layer.
                    for pname, p in main_mag[adapter_name].named_parameters():
                        try:
                            setattr(sampling_mag[adapter_name], pname, p)
                        except Exception:
                            # Fall back to in-place copy if direct attr set fails.
                            with torch.no_grad():
                                getattr(sampling_mag[adapter_name], pname).copy_(p)

        # Refresh the sampling network's wrapper cache so it sees shared params.
        for a in self.unet_loras:
            a._param_cache = None

    # ------------------------------------------------------------- context mgr
    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.is_active = False
        return False

    # ------------------------------------------------------ checkpointing stubs
    def enable_gradient_checkpointing(self):
        self.is_checkpointing = True
        # PEFT does not provide a per-adapter checkpointing toggle; rely on the
        # base model's own gradient_checkpointing (handled by the trainer).

    def disable_gradient_checkpointing(self):
        self.is_checkpointing = False

    def _update_checkpointing(self):
        pass

    # ----------------------------------------------------------- merge stubs
    def merge_in(self, merge_weight: float = 1.0):
        # PEFT can merge via peft_model.merge_and_unload(), but toolkit's
        # merge_in is used for in-place scaling during inference. For phase 1
        # we keep adapters unmerged and rely on PEFT's own scaling.
        if self.is_dora:
            return
        self.is_merged_in = True
        try:
            self.peft_model.merge_adapter(["default"])
        except Exception as e:
            print(f"[PeftNetwork] merge_in skipped: {e}")
            self.is_merged_in = False

    def merge_out(self, merge_weight: float = 1.0):
        if not self.is_merged_in:
            return
        try:
            self.peft_model.unmerge_adapter(["default"])
        except Exception as e:
            print(f"[PeftNetwork] merge_out skipped: {e}")
        self.is_merged_in = False
