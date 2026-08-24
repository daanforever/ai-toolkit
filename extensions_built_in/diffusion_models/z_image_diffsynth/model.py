# ZImageDiffSynthModel: BaseModel with arch "zimage_diffsynth", using DiffSynth DiT/forward and toolkit trainer.

import gc
import os
from typing import List, Optional

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from .scheduler_config import build_scheduler_config
from .scheduler_adapter import DiffSynthZImageSchedulerAdapter
from toolkit.accelerator import unwrap_model
from toolkit.paths import normalize_path
from toolkit.util.debug import memory_debug, is_debug_enabled

from . import loader as loader_mod
from . import forward as forward_mod
from . import prompt_encoding as prompt_encoding_mod
from . import diffsynth_training as diffsynth_training_mod
from . import sampling as sampling_mod
from . import lora as lora_mod
from .vae_wrapper import DiffSynthVAEWrapper

scheduler_config = build_scheduler_config(use_dynamic_shifting=False)


def _resolve_use_dynamic_shifting(model_kwargs):
    """Resolve dynamic time shifting from model_kwargs (default False)."""
    mk = model_kwargs or {}
    return bool(mk.get("use_dynamic_shifting", False))


def _resolve_use_diffsynth_prompt_encoding(model_kwargs):
    """Resolve prompt encoder branch from model_kwargs.

    If use_diffsynth_prompt_encoding is set explicitly, use it; otherwise inherit
    use_diffsynth_training_loop (default True when both are omitted).
    """
    mk = model_kwargs or {}
    if "use_diffsynth_prompt_encoding" in mk:
        return mk["use_diffsynth_prompt_encoding"]
    return mk.get("use_diffsynth_training_loop", True)


class _DiTUnetWrapper(torch.nn.Module):
    """Wraps ZImageDiT so that .config.patch_size exists for BatchProcessor / timestep scheduling.
    Forwards all other attributes and calls to the inner DiT. Uses _inner_dit so that .device/.training
    etc. are always reachable; nn.Module stores submodules in _modules, so we read from there in
    __getattr__ to avoid 'object has no attribute dit' after quantize."""

    def __init__(self, dit):
        super().__init__()
        self._inner_dit = dit
        self.config = type("_Config", (), {"patch_size": 2})()

    def forward(self, *args, **kwargs):
        return self._modules["_inner_dit"](*args, **kwargs)

    def to(self, *args, **kwargs):
        # torch.compile leaves weakrefs that break nn.Module._apply / swap_tensors (quanto).
        from .compile_blocks import move_dit_with_compiled_blocks

        move_dit_with_compiled_blocks(self._inner_dit, None, *args, **kwargs)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None):
        return self.to("cuda" if device is None else device)

    def __getattr__(self, name):
        if name in ("dit", "_inner_dit"):
            return self._modules["_inner_dit"]
        if name == "config":
            return object.__getattribute__(self, name)
        # base_model.save_device_state() and predict_noise expect unet.device / unet.training / unet.dtype;
        # inner DiT may not expose these directly.
        if name == "device":
            params = list(self._modules["_inner_dit"].parameters())
            return next(iter(params)).device if params else torch.device("cpu")
        if name == "training":
            return self._modules["_inner_dit"].training
        if name == "dtype":
            params = list(self._modules["_inner_dit"].parameters())
            return next(iter(params)).dtype if params else torch.float32
        return getattr(self._modules["_inner_dit"], name)


class ZImageDiffSynthModel(BaseModel):
    """Z-Image DiffSynth: flow-matching DiT with optional diffusers-format sampling transformer.

    Config sampling params (from SampleConfig / GenerateImageConfig) are used as follows:
    - sample_steps → gen_config.num_inference_steps (passed to pipeline)
    - guidance_scale → gen_config.guidance_scale (passed to pipeline)
    - sampler: use \"flowmatch\"; the pipeline always uses flow-match Euler (no scheduler swap).
    """
    arch = "zimage_diffsynth"
    is_flow_matching = True
    is_transformer = True
    target_lora_modules = lora_mod.TARGET_LORA_MODULES

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self._raw_dit = None
        self._main_is_diffusers = False
        self._sampling_transformer = None
        self._sampling_network = None
        self._sampling_is_diffusers = False
        # When True, we are inside our generate_images(); device moves are done
        # once there (main→CPU, sampling→GPU before loop; restore in finally).
        self._sampling_in_batch_generate = False
        # Enable gradient checkpointing by default for DiffSynth DiT to
        # reduce peak VRAM usage during training forwards.
        self.gradient_checkpointing = True
        self._dit_blocks_compiled = False

    @staticmethod
    def get_train_scheduler(use_diffsynth_loop=True, use_dynamic_shifting=False):
        """use_diffsynth_loop=True: same timesteps/add_noise/weight as DiffSynth Z-Image.sh."""
        if use_dynamic_shifting:
            return CustomFlowMatchEulerDiscreteScheduler(
                **build_scheduler_config(use_dynamic_shifting=True)
            )
        if use_diffsynth_loop:
            return DiffSynthZImageSchedulerAdapter()
        return CustomFlowMatchEulerDiscreteScheduler(**build_scheduler_config(False))

    def get_bucket_divisibility(self):
        return 16 * 2

    def _place_training_dit(self, device):
        """Move frozen DiT (quantized payload included) onto ``device``; fail if it stays off."""
        target = device if isinstance(device, torch.device) else torch.device(device)
        dit = getattr(self, "_raw_dit", None)
        if dit is not None:
            from .compile_blocks import move_dit_with_compiled_blocks

            move_dit_with_compiled_blocks(dit, None, target)
        elif self.model is not None:
            self.model.to(target)
        self._assert_dit_on_device(target)

    def _assert_dit_on_device(self, device):
        from toolkit.util.device import devices_equal, quantized_payload_device

        target = device if isinstance(device, torch.device) else torch.device(device)
        module = getattr(self, "_raw_dit", None) or self.model
        if module is None:
            return
        p = self._first_frozen_base_param(module)
        if p is None:
            try:
                p = next(module.parameters())
            except StopIteration:
                return
        if not devices_equal(p.device, target):
            raise RuntimeError(
                f"Z-Image DiffSynth DiT param on {p.device}, expected {target}"
            )
        payload = quantized_payload_device(p)
        if payload is not None and not devices_equal(payload, target):
            raise RuntimeError(
                f"Z-Image DiffSynth DiT quantized payload on {payload}, expected {target}"
            )

    def _move_main_network(self, device):
        """Pin DiT + training LoRA to CUDA; preserve network.dtype. Never move to CPU."""
        with memory_debug(self.print_and_status_update, "Move main network"):
            target = device if isinstance(device, torch.device) else torch.device(device)
            if target.type == "cpu":
                return
            self._place_training_dit(target)
            net = getattr(self, "network", None)
            if net is None or not hasattr(net, "force_to"):
                return
            net = unwrap_model(net)
            # Preserve network.dtype (e.g. fp32 for optimizer); do not use model.dtype.
            p = next((p for p in net.parameters() if p.requires_grad), None)
            dtype = p.dtype if p is not None else self.torch_dtype
            net.force_to(target, dtype)
            if is_debug_enabled():
                self.print_and_status_update(
                    f"\n[zimage_diffsynth] main network force_to {device}"
                )

    @staticmethod
    def _first_frozen_base_param(module):
        """First frozen non-LoRA parameter (base DiT weight), for device placement checks."""
        for name, p in module.named_parameters():
            if "lora" not in name.lower() and not p.requires_grad:
                return p
        return None

    def _move_sampling_transformer(self, device):
        """Move _sampling_transformer (sampling DiT base). Do not force_to _sampling_network."""
        with memory_debug(self.print_and_status_update, "Move sampling transformer"):
            st = getattr(self, "_sampling_transformer", None)
            if st is None:
                return
            target = device if isinstance(device, torch.device) else torch.device(device)
            base_p = self._first_frozen_base_param(st)
            need_move = base_p is None or base_p.device != target
            if not need_move:
                return
            if is_debug_enabled():
                self.print_and_status_update(
                    f"\n[zimage_diffsynth] moving sampling transformer to {device}"
                )
            try:
                # Wrapper.to unwraps compiled blocks; plain Module uses nn.Module.to.
                st.to(device)
            except Exception as e:
                self.print_and_status_update(
                    f"[zimage_diffsynth] failed moving sampling transformer to {device}: {e}"
                )

    def _flush_cuda(self):
        """Release CUDA cache and run GC so VRAM is actually freed after model moves."""
        if isinstance(self.device_torch, torch.device) and self.device_torch.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _log_device_state(self, label: str):
        """Log device of key modules (for debug). Only runs when config.debug is enabled."""
        if not is_debug_enabled():
            return

        def _device_str(obj):
            if obj is None:
                return "None"
            try:
                p = next(obj.parameters(), None)
                return str(p.device) if p is not None else "no_params"
            except Exception:
                try:
                    return str(getattr(obj, "device", "?"))
                except Exception:
                    return "?"

        te = self.text_encoder
        if isinstance(te, list):
            te_dev = ",".join(_device_str(e) for e in (te or []))
        else:
            te_dev = _device_str(te)

        parts = [
            f"model={_device_str(self.model)}",
            f"network={_device_str(getattr(self, "network", None))}",
            f"sampling_transformer={_device_str(getattr(self, "_sampling_transformer", None))}",
            f"sampling_network={_device_str(getattr(self, "_sampling_network", None))}",
            f"vae={_device_str(self.vae)}",
            f"text_encoder={te_dev}",
        ]
        self.print_and_status_update(
            f"\n[DEBUG zimage_diffsynth device state] {label}: " + ", ".join(parts)
        )

    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        self.print_and_status_update("Loading ZImage DiffSynth model")
        model_path = normalize_path(self.model_config.name_or_path)
        base_path = normalize_path(self.model_config.extras_name_or_path or model_path)
        sampling_path = getattr(
            self.model_config, "sampling_name_or_path", None
        )
        if sampling_path:
            sampling_path = normalize_path(sampling_path)

        def log(msg):
            self.print_and_status_update(msg)

        # Same as z_image: qfloat8 is not compatible with quantize path; use float8
        if getattr(self.model_config, "quantize", False) and getattr(
            self.model_config, "qtype", None
        ) == "qfloat8":
            self.model_config.qtype = "float8"

        with memory_debug(self.print_and_status_update, "Load components"):
            model_kwargs = getattr(self.model_config, "model_kwargs", None) or {}
            loader_mode = model_kwargs.get("loader", "auto")
            components = loader_mod.load_components(
                model_path,
                base_path,
                dtype=dtype,
                device=device,
                log_fn=log,
                quantize_te=getattr(self.model_config, "quantize_te", False),
                qtype_te=getattr(self.model_config, "qtype_te", "float8"),
                sampling_transformer_path=sampling_path,
                quantize_transformer=getattr(self.model_config, "quantize", False),
                base_model=self,
                loader_mode=loader_mode,
            )

        # Optionally disable refiner stacks via model.model_kwargs to reduce VRAM
        # (noise_refiner ~10 GB, context_refiner ~4 GB). Replace with empty ModuleList
        # so code that iterates over them (e.g. model_fn_z_image_turbo) still runs.
        # Only applies to DiffSynth DiT (not Diffusers ZImageTransformer2DModel).
        kwargs = getattr(self.model_config, "model_kwargs", None) or {}
        self._disable_noise_refiner = kwargs.get("disable_noise_refiner", True)
        self._disable_context_refiner = kwargs.get("disable_context_refiner", True)
        self._raw_dit = components["dit"]
        self._main_is_diffusers = components.get("dit_is_diffusers", False)
        if not self._main_is_diffusers:
            for _ref_name, _do_disable in (
                ("noise_refiner", self._disable_noise_refiner),
                ("context_refiner", self._disable_context_refiner),
            ):
                if _do_disable and hasattr(self._raw_dit, _ref_name):
                    try:
                        setattr(self._raw_dit, _ref_name, torch.nn.ModuleList([]))
                        self.print_and_status_update(f"Disabled DiT module: {_ref_name}")
                    except Exception:
                        pass
        self.model = _DiTUnetWrapper(self._raw_dit)
        self.vae = components["vae_wrapper"]
        # For zimage_diffsynth VAE is needed on GPU during both training and sampling.
        # Move it to the target VAE device immediately so save_device_state() records
        # GPU as the baseline device and restore_device_state() keeps it there.
        try:
            self.vae.to(self.vae_device_torch)
        except Exception:
            pass
        self.text_encoder = [components["text_encoder"]]
        self.tokenizer = [components["tokenizer"]]
        self._sampling_is_diffusers = components.get("sampling_is_diffusers", False)
        sampling_dit = components.get("sampling_dit")
        if sampling_dit is not None and not self._sampling_is_diffusers:
            # Apply same refiner disabling as main DiT so structure and LoRA names match.
            for _ref_name, _do_disable in (
                ("noise_refiner", self._disable_noise_refiner),
                ("context_refiner", self._disable_context_refiner),
            ):
                if _do_disable and hasattr(sampling_dit, _ref_name):
                    try:
                        setattr(sampling_dit, _ref_name, torch.nn.ModuleList([]))
                    except Exception:
                        pass
        # Always wrap sampling DiT in _DiTUnetWrapper so LoRA module names match the main model
        # (main uses _DiTUnetWrapper → lora_unet__inner_dit_*). For diffusers sampling we unwrap
        # when building ZImagePipeline in get_generation_pipeline.
        if sampling_dit is not None:
            self._sampling_transformer = _DiTUnetWrapper(sampling_dit)
        else:
            self._sampling_transformer = None

        # Decide whether to use the original DiffSynth training loop behaviour.
        # Flag lives in model_kwargs for this model; default is True so that
        # existing configs (without the flag) keep DiffSynth-compatible behaviour.
        use_diffsynth = True
        use_dynamic_shifting = False
        try:
            model_kwargs = getattr(self.model_config, "model_kwargs", {}) or {}
            use_diffsynth = model_kwargs.get("use_diffsynth_training_loop", True)
            use_dynamic_shifting = _resolve_use_dynamic_shifting(model_kwargs)
        except Exception:
            # On any unexpected shape, keep defaults.
            use_diffsynth = True
            use_dynamic_shifting = False

        self.noise_scheduler = ZImageDiffSynthModel.get_train_scheduler(
            use_diffsynth_loop=use_diffsynth,
            use_dynamic_shifting=use_dynamic_shifting,
        )
        self.pipeline = None
        self._move_main_network("cpu")
        self._move_sampling_transformer("cpu")
        self.print_and_status_update("Model loaded")

    def get_model_to_train(self):
        return self.model

    @property
    def transformer(self):
        return self.model

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch=None,
        **kwargs,
    ):
        # Ensure DiT weights live on the same device as the latents we are
        # about to run on. During training the device-state presets will
        # already have moved the wrapper (and inner DiT) to the correct
        # device; for one-off calls like the smoke test this also brings a
        # freshly-loaded, CPU-resident DiT onto self.device_torch.
        # Never follow CPU latents onto the DiT while the trainer device is CUDA;
        # that silently runs the frozen base on CPU (~30 s/it).
        if (
            isinstance(self.device_torch, torch.device)
            and self.device_torch.type == "cuda"
        ):
            self._place_training_dit(self.device_torch)
        elif isinstance(latent_model_input, torch.Tensor):
            self._place_training_dit(latent_model_input.device)
        else:
            self._place_training_dit(self.device_torch)
        # Z-Image DiT is trained on latent-space tensors with a fixed
        # channel count (e.g. 16). If we accidentally receive 3‑channel BCHW
        # tensors here (RGB-like), they will eventually hit dit.all_x_embedder
        # with the wrong input dimension and cause an opaque Quanto matmul error.
        #
        # Guard early and surface a clear message that points to likely causes:
        # - cached latents created by another model / latent_space_version
        # - RGB tensors cached as "latents" by external tools.
        if isinstance(latent_model_input, torch.Tensor) and latent_model_input.dim() == 4:
            in_channels = getattr(self._raw_dit, "in_channels", None)
            if in_channels is not None:
                c = latent_model_input.shape[1]
                if c != in_channels:
                    raise RuntimeError(
                        f"Z-Image DiffSynth DiT expected latents with {in_channels} channels "
                        f"(B x {in_channels} x H x W), but got B x {c} x H x W instead. "
                        "This usually means your dataset latent cache was created with a different "
                        "model/latent_space_version or that RGB tensors were saved as latents. "
                        "Please disable cache_latents/cache_latents_to_disk for this dataset or "
                        "regenerate latents using the current zimage_diffsynth model."
                    )

        text_embeds = text_embeddings.text_embeds
        attention_mask = text_embeddings.attention_mask
        if isinstance(text_embeds, torch.Tensor) and len(text_embeds.shape) == 3:
            if attention_mask is not None:
                text_embeds = [
                    text_embeds[i][attention_mask[i].bool()]
                    for i in range(text_embeds.shape[0])
                ]
            else:
                text_embeds = [text_embeds[i] for i in range(text_embeds.shape[0])]
        elif isinstance(text_embeds, list) and text_embeds and len(text_embeds[0].shape) == 3:
            new_text_embeds = []
            for i, t in enumerate(text_embeds):
                if t is None:
                    continue
                if attention_mask is not None:
                    if isinstance(attention_mask, (list, tuple)):
                        mask = attention_mask[i]
                    else:
                        mask = (
                            attention_mask[i]
                            if attention_mask.dim() == 3
                            else attention_mask
                        )
                    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
                        new_text_embeds += [
                            t[j][mask[j].bool()] for j in range(t.shape[0])
                        ]
                    elif isinstance(mask, torch.Tensor):
                        new_text_embeds += [
                            t[j][mask.bool()] for j in range(t.shape[0])
                        ]
                    else:
                        new_text_embeds += [t[j][mask] for j in range(t.shape[0])]
                else:
                    new_text_embeds += list(t.unbind(dim=0))
            text_embeds = new_text_embeds

        # Geometric pad validity → patch grid (attention only; not user ROI mask)
        image_valid_patches = None
        if (
            batch is not None
            and getattr(batch, "image_valid_mask_tensor", None) is not None
            and isinstance(latent_model_input, torch.Tensor)
            and latent_model_input.dim() == 4
        ):
            from .spatial_attention import pixel_valid_to_patch_valid

            valid = batch.image_valid_mask_tensor
            if valid.dim() == 3:
                valid = valid.unsqueeze(1)
            B, _, lh, lw = latent_model_input.shape
            if valid.shape[0] != B:
                raise ValueError(
                    f"image_valid_mask_tensor batch {valid.shape[0]} != latents batch {B}"
                )
            valid = valid.to(device=latent_model_input.device)
            patch_grid = pixel_valid_to_patch_valid(valid, lh, lw, patch_size=2)
            # Only activate adapter when at least one fully padded patch exists
            patches = [patch_grid[b] for b in range(B)]
            if any(not p.all().item() for p in patches):
                image_valid_patches = patches
            else:
                image_valid_patches = None

        # Diffusers ZImageTransformer2DModel: official DreamBooth / z_image convention.
        if getattr(self, "_main_is_diffusers", False):
            # Same train/model dtype gate as DiffSynth run_forward: activations may
            # be train.dtype (fp32) while float8-quantized DiT dequantizes to model.dtype (bf16).
            train_dtype = getattr(self, "train_torch_dtype", None) or self.torch_dtype
            model_dtype = self.torch_dtype
            if isinstance(latent_model_input, torch.Tensor) and latent_model_input.dtype != model_dtype:
                latent_model_input = latent_model_input.to(model_dtype)
            if isinstance(text_embeds, torch.Tensor):
                if text_embeds.dtype != model_dtype:
                    text_embeds = text_embeds.to(model_dtype)
            elif isinstance(text_embeds, list):
                text_embeds = [
                    p.to(model_dtype) if isinstance(p, torch.Tensor) and p.dtype != model_dtype else p
                    for p in text_embeds
                ]

            latent_list = list(latent_model_input.unsqueeze(2).unbind(dim=0))
            timestep_model_input = (1000 - timestep) / 1000
            use_autocast = (
                latent_model_input.device.type == "cuda"
                and model_dtype in (torch.float16, torch.bfloat16)
            )
            from .spatial_attention import spatial_attention_context

            with torch.autocast(
                device_type="cuda",
                dtype=model_dtype if use_autocast else torch.float32,
                enabled=use_autocast,
            ):
                with spatial_attention_context(self._raw_dit, image_valid_patches):
                    model_out_list = self._raw_dit(
                        latent_list,
                        timestep_model_input,
                        text_embeds,
                        return_dict=False,
                    )[0]
            noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
            noise_pred = -noise_pred.squeeze(2)
            if noise_pred.dtype != train_dtype:
                noise_pred = noise_pred.to(train_dtype)
            return noise_pred

        use_gradient_checkpointing = getattr(self, "gradient_checkpointing", False)
        train_dtype = getattr(self, "train_torch_dtype", None) or self.torch_dtype
        # Pass raw DiT to DiffSynth model_fn (expects real DiT with t_embedder, etc.).
        noise_pred = forward_mod.run_forward(
            self._raw_dit,
            latent_model_input,
            timestep,
            text_embeds,
            model_dtype=self.torch_dtype,
            use_gradient_checkpointing=use_gradient_checkpointing,
            train_dtype=train_dtype,
            image_valid_patches=image_valid_patches,
        )
        return noise_pred

    def get_turbo_teacher_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch=None,
        **kwargs,
    ):
        """
        Frozen Z-Image-Turbo velocity pred on ``_sampling_transformer`` (no train LoRA).

        I/O matches ``get_noise_prediction`` (Diffusers list/CFHW + negate, or DiffSynth
        ``run_forward``). Moves sampling DiT to device for the forward, then offloads.
        """
        st = getattr(self, "_sampling_transformer", None)
        if st is None:
            raise ValueError(
                "turbo_teacher_weight>0 requires a loaded sampling transformer "
                "(set model.sampling_name_or_path to Z-Image-Turbo)."
            )

        text_embeds = text_embeddings.text_embeds
        attention_mask = text_embeddings.attention_mask
        if isinstance(text_embeds, torch.Tensor) and len(text_embeds.shape) == 3:
            if attention_mask is not None:
                text_embeds = [
                    text_embeds[i][attention_mask[i].bool()]
                    for i in range(text_embeds.shape[0])
                ]
            else:
                text_embeds = [text_embeds[i] for i in range(text_embeds.shape[0])]
        elif isinstance(text_embeds, list) and text_embeds and len(text_embeds[0].shape) == 3:
            new_text_embeds = []
            for i, t in enumerate(text_embeds):
                if t is None:
                    continue
                if attention_mask is not None:
                    if isinstance(attention_mask, (list, tuple)):
                        mask = attention_mask[i]
                    else:
                        mask = (
                            attention_mask[i]
                            if attention_mask.dim() == 3
                            else attention_mask
                        )
                    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
                        new_text_embeds += [
                            t[j][mask[j].bool()] for j in range(t.shape[0])
                        ]
                    elif isinstance(mask, torch.Tensor):
                        new_text_embeds += [
                            t[j][mask.bool()] for j in range(t.shape[0])
                        ]
                    else:
                        new_text_embeds += [t[j][mask] for j in range(t.shape[0])]
                else:
                    new_text_embeds += list(t.unbind(dim=0))
            text_embeds = new_text_embeds

        image_valid_patches = None
        if (
            batch is not None
            and getattr(batch, "image_valid_mask_tensor", None) is not None
            and isinstance(latent_model_input, torch.Tensor)
            and latent_model_input.dim() == 4
        ):
            from .spatial_attention import pixel_valid_to_patch_valid

            valid = batch.image_valid_mask_tensor
            if valid.dim() == 3:
                valid = valid.unsqueeze(1)
            B, _, lh, lw = latent_model_input.shape
            if valid.shape[0] != B:
                raise ValueError(
                    f"image_valid_mask_tensor batch {valid.shape[0]} != latents batch {B}"
                )
            valid = valid.to(device=latent_model_input.device)
            patch_grid = pixel_valid_to_patch_valid(valid, lh, lw, patch_size=2)
            patches = [patch_grid[b] for b in range(B)]
            if any(not p.all().item() for p in patches):
                image_valid_patches = patches

        dit = getattr(st, "_inner_dit", None)
        if dit is None:
            dit = unwrap_model(st)
            if hasattr(dit, "_inner_dit"):
                dit = dit._inner_dit

        train_dtype = getattr(self, "train_torch_dtype", None) or self.torch_dtype
        model_dtype = self.torch_dtype
        target_device = self.device_torch
        if (
            isinstance(target_device, torch.device)
            and target_device.type == "cuda"
        ):
            pass
        elif isinstance(latent_model_input, torch.Tensor):
            target_device = latent_model_input.device
        else:
            target_device = self.device_torch

        try:
            self._move_sampling_transformer(target_device)
            with torch.no_grad():
                if getattr(self, "_sampling_is_diffusers", False):
                    if (
                        isinstance(latent_model_input, torch.Tensor)
                        and latent_model_input.dtype != model_dtype
                    ):
                        latent_model_input = latent_model_input.to(model_dtype)
                    if isinstance(text_embeds, torch.Tensor):
                        if text_embeds.dtype != model_dtype:
                            text_embeds = text_embeds.to(model_dtype)
                    elif isinstance(text_embeds, list):
                        text_embeds = [
                            p.to(model_dtype)
                            if isinstance(p, torch.Tensor) and p.dtype != model_dtype
                            else p
                            for p in text_embeds
                        ]

                    latent_list = list(latent_model_input.unsqueeze(2).unbind(dim=0))
                    timestep_model_input = (1000 - timestep) / 1000
                    use_autocast = (
                        latent_model_input.device.type == "cuda"
                        and model_dtype in (torch.float16, torch.bfloat16)
                    )
                    from .spatial_attention import spatial_attention_context

                    with torch.autocast(
                        device_type="cuda",
                        dtype=model_dtype if use_autocast else torch.float32,
                        enabled=use_autocast,
                    ):
                        with spatial_attention_context(dit, image_valid_patches):
                            model_out_list = dit(
                                latent_list,
                                timestep_model_input,
                                text_embeds,
                                return_dict=False,
                            )[0]
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
                    noise_pred = -noise_pred.squeeze(2)
                    if noise_pred.dtype != train_dtype:
                        noise_pred = noise_pred.to(train_dtype)
                else:
                    noise_pred = forward_mod.run_forward(
                        dit,
                        latent_model_input,
                        timestep,
                        text_embeds,
                        model_dtype=model_dtype,
                        use_gradient_checkpointing=False,
                        train_dtype=train_dtype,
                        image_valid_patches=image_valid_patches,
                    )
                return noise_pred.detach()
        finally:
            self._move_sampling_transformer("cpu")
            if getattr(self.model_config, "low_vram", False):
                self._flush_cuda()

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        te = self.text_encoder[0]
        tok = self.tokenizer[0]
        if next(te.parameters()).device != self.device_torch:
            try:
                te.to(self.device_torch)
            except RuntimeError as e:
                if "Couldn't swap" in str(e) or "swap_tensors" in str(e):
                    # Quantized TE: .to() can fail when already on device or shared storage
                    pass
                else:
                    raise
        try:
            mk = getattr(self.model_config, "model_kwargs", {}) or {}
            use_diffsynth_prompt_encoding = _resolve_use_diffsynth_prompt_encoding(mk)
        except Exception:
            use_diffsynth_prompt_encoding = True
        encode_dtype = getattr(self, "train_torch_dtype", None) or self.torch_dtype
        if use_diffsynth_prompt_encoding:
            return diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i(
                tok,
                te,
                prompt,
                self.device_torch,
                dtype=encode_dtype,
            )
        return prompt_encoding_mod.encode_prompt(
            tok,
            te,
            prompt,
            self.device_torch,
            dtype=encode_dtype,
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_generation_pipeline(self):
        return sampling_mod.get_generation_pipeline(self)

    def decode_latents(
        self,
        latents: torch.Tensor,
        device=None,
        dtype=None,
    ):
        """
        Decode latents for preview / sampling.

        For the DiffSynth Z-Image training loop we sometimes operate directly in
        pixel space (flow-matching on RGB images). In that case SDTrainer passes
        tensors with 3 channels here when saving noised-input previews. Feeding
        those through the VAE decoder (which expects latent_channels, e.g. 16)
        triggers a channel-mismatch error.

        To keep previews working without affecting the main training path, we
        treat 3-channel inputs as already-decoded images and only apply a dtype /
        device cast. For true latents (latent_channels, e.g. 16) we fall back to
        the BaseModel implementation which uses the wrapped VAE.
        """
        if latents.dim() == 4 and latents.shape[1] == 3:
            if device is None:
                device = self.device
            if dtype is None:
                dtype = self.torch_dtype
            return latents.to(device, dtype=dtype)

        return super().decode_latents(latents, device=device, dtype=dtype)

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        def _run():
            sc = self.get_bucket_divisibility()
            gen_config.width = int(gen_config.width // sc * sc)
            gen_config.height = int(gen_config.height // sc * sc)
            cond = conditional_embeds.text_embeds
            cond_mask = conditional_embeds.attention_mask
            uncond = unconditional_embeds.text_embeds
            uncond_mask = unconditional_embeds.attention_mask
            if isinstance(cond, torch.Tensor) and len(cond.shape) == 3:
                if cond_mask is not None:
                    cond = [cond[i][cond_mask[i].bool()] for i in range(cond.shape[0])]
                else:
                    cond = [cond[i] for i in range(cond.shape[0])]
            if isinstance(uncond, torch.Tensor) and len(uncond.shape) == 3:
                if uncond_mask is not None:
                    uncond = [uncond[i][uncond_mask[i].bool()] for i in range(uncond.shape[0])]
                else:
                    uncond = [uncond[i] for i in range(uncond.shape[0])]
            return pipeline(
                prompt_embeds=cond,
                negative_prompt_embeds=uncond,
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=gen_config.guidance_scale,
                latents=gen_config.latents,
                generator=generator,
                **extra,
            ).images[0]

        use_sampling = (
            self._sampling_transformer is not None
            and isinstance(self.device_torch, torch.device)
            and self.device_torch.type == "cuda"
        )
        if not use_sampling:
            return _run()
        # Batch path: BaseModel already moved main→CPU, sampling→GPU once; no per-prompt moves.
        if self._sampling_in_batch_generate:
            return _run()
        # Standalone path (e.g. smoke test): move sampling to GPU, main to CPU, then restore after.
        try:
            if is_debug_enabled():
                self.print_and_status_update(
                    "\n[zimage_diffsynth] standalone sampling: moving main transformer to "
                    "CPU and sampling transformer to GPU"
                )
            self.model.to("cpu", dtype=self.torch_dtype)
            self._flush_cuda()
            self._move_sampling_transformer(self.device_torch)
            return _run()
        finally:
            if is_debug_enabled():
                self.print_and_status_update(
                    "\n[zimage_diffsynth] standalone sampling: restoring main "
                    "transformer to GPU and sampling transformer to CPU"
                )
            self._move_sampling_transformer("cpu")
            self.model.to(self.device_torch, dtype=self.torch_dtype)
            self._move_main_network(self.device_torch)
            self._flush_cuda()

    def generate_images(
        self,
        image_configs: List[GenerateImageConfig],
        sampler=None,
    ):
        saved_network = None
        use_sampling = (
            hasattr(self, "_sampling_transformer")
            and self._sampling_transformer is not None
            and isinstance(self.device_torch, torch.device)
            and self.device_torch.type == "cuda"
        )
        try:
            if (
                hasattr(self, "_sampling_transformer")
                and self._sampling_transformer is not None
                and hasattr(self, "_sampling_network")
                and self._sampling_network is not None
                and hasattr(self, "network")
                and self.network is not None
            ):
                saved_network = self.network
                self.network = self._sampling_network

            if use_sampling:
                if is_debug_enabled():
                    self.print_and_status_update(
                        "\n[zimage_diffsynth] batch generate: enabling sampling transformer "
                        "on GPU and using sampling network"
                    )
                self._sampling_in_batch_generate = True
            try:
                return super().generate_images(image_configs, sampler)
            finally:
                if use_sampling:
                    if is_debug_enabled():
                        self.print_and_status_update(
                            "\n[zimage_diffsynth] batch generate: restoring main "
                            "transformer to GPU and moving sampling transformer to CPU"
                        )
                    self._sampling_in_batch_generate = False
                    # Restore after batch: main back on GPU, sampling back on CPU (no memory spike).
                    with memory_debug(
                        self.print_and_status_update,
                        "zimage_diffsynth after batch restore",
                    ):
                        if saved_network is not None:
                            self.network = saved_network
                        self._move_sampling_transformer("cpu")
                        self.model.to(self.device_torch, dtype=self.torch_dtype)
                        self._move_main_network(self.device_torch)
                    
                    self._flush_cuda()
                    self._log_device_state("after batch restore")
        finally:
            if saved_network is not None:
                self.network = saved_network

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        dit = unwrap_model(self.model)
        if hasattr(dit, "_inner_dit"):
            dit = dit._inner_dit
        elif hasattr(dit, "dit"):
            dit = dit.dit
        save_dir = os.path.join(output_path, "transformer")
        os.makedirs(save_dir, exist_ok=True)
        if getattr(self, "_main_is_diffusers", False) and hasattr(
            dit, "save_pretrained"
        ):
            dit.save_pretrained(save_directory=save_dir, safe_serialization=True)
        else:
            from safetensors.torch import save_file

            save_file(dit.state_dict(), os.path.join(save_dir, "model.safetensors"))
        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "zimage_diffsynth"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        # Main DiT layers plus any refiner stacks that were not disabled via model_kwargs.
        names = ["layers"]
        if not getattr(self, "_disable_noise_refiner", True):
            names.append("noise_refiner")
        if not getattr(self, "_disable_context_refiner", True):
            names.append("context_refiner")
        return names

    def compile_dit_blocks(self):
        """Per-block torch.compile on DiffSynth DiT ModuleLists (after quantize + LoRA)."""
        if getattr(self, "_dit_blocks_compiled", False):
            return
        from .compile_blocks import compile_dit_module_lists

        names = self.get_transformer_block_names() or []
        log = self.print_and_status_update
        totals = {"ok": 0, "failed": 0, "skipped": 0}

        def _accumulate(stats):
            for k in totals:
                totals[k] += stats.get(k, 0)

        if (
            self._raw_dit is not None
            and names
            and not getattr(self, "_main_is_diffusers", False)
        ):
            log("[zimage_diffsynth] Compiling main DiT blocks (torch.compile dynamic=True)")
            _accumulate(compile_dit_module_lists(self._raw_dit, names, log_fn=log))

        sampling = getattr(self, "_sampling_transformer", None)
        if (
            sampling is not None
            and not getattr(self, "_sampling_is_diffusers", False)
            and names
        ):
            inner = getattr(sampling, "_inner_dit", None)
            if inner is None:
                inner = unwrap_model(sampling)
                if hasattr(inner, "_inner_dit"):
                    inner = inner._inner_dit
            if inner is not None:
                log("[zimage_diffsynth] Compiling sampling DiT blocks")
                try:
                    inner.to(self.device_torch)
                except Exception:
                    pass
                _accumulate(compile_dit_module_lists(inner, names, log_fn=log))
                self._move_sampling_transformer("cpu")

        self._dit_blocks_compiled = True
        log(
            f"[zimage_diffsynth] DiT compile summary: ok={totals['ok']} "
            f"failed={totals['failed']} skipped={totals['skipped']}"
        )

    def get_lora_optimizer_param_groups(self, network, unet_lr, default_lr):
        unet_loras = getattr(network, "unet_loras", None)
        if not unet_loras:
            return None

        grouped_loras = lora_mod.group_loras_by_block(unet_loras)
        if not grouped_loras:
            return None

        if "other" in grouped_loras:
            self.print_and_status_update(
                f"[zimage_diffsynth] LoRA block grouping fallback=other count={len(grouped_loras['other']['loras'])}"
            )

        lr_value = unet_lr if unet_lr is not None else default_lr
        param_groups = []
        for block_key, entry in grouped_loras.items():
            block_loras = entry["loras"]
            block_index = entry["block_index"]
            lora_params = []
            magnitude_params = []
            for lora in block_loras:
                for name, param in lora.named_parameters():
                    if "magnitude" in name:
                        magnitude_params.append(param)
                    else:
                        lora_params.append(param)

            if lora_params:
                group = {"params": lora_params, "name": block_key}
                if lr_value is not None:
                    group["lr"] = lr_value
                if "layer" in block_key and block_index is not None:
                    group["index"] = block_index
                param_groups.append(group)

            if magnitude_params:
                mag_group = {
                    "params": magnitude_params,
                    "is_magnitude": True,
                    "name": f"{block_key}_magnitude",
                }
                if lr_value is not None:
                    mag_group["lr"] = lr_value
                param_groups.append(mag_group)

        return param_groups or None

    def convert_lora_weights_before_save(self, state_dict):
        return lora_mod.convert_lora_weights_before_save(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return lora_mod.convert_lora_weights_before_load(state_dict)

    def convert_accuracy_recovery_weights_before_load(self, state_dict):
        """Used by quantize_model when loading the ARA so LoRASpecialNetwork.load_weights matches keys."""
        return lora_mod.convert_accuracy_recovery_weights_before_load(state_dict)
