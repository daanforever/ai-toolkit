# ZImageDiffSynthModel: BaseModel with arch "zimage_diffsynth", using DiffSynth DiT/forward and toolkit trainer.

import os
from typing import List, Optional

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from .scheduler_adapter import DiffSynthZImageSchedulerAdapter
from toolkit.accelerator import unwrap_model
from toolkit.paths import normalize_path
from toolkit.util.debug import memory_debug, is_debug_enabled

from . import loader as loader_mod
from . import forward as forward_mod
from . import prompt_encoding as prompt_encoding_mod
from . import sampling as sampling_mod
from . import lora as lora_mod
from .vae_wrapper import DiffSynthVAEWrapper

scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


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

    def __getattr__(self, name):
        if name == "dit":
            return self._modules["_inner_dit"]
        if name in ("_inner_dit", "config"):
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
        self._sampling_transformer = None
        self._sampling_network = None

    @staticmethod
    def get_train_scheduler(use_diffsynth_loop=True):
        """use_diffsynth_loop=True: same timesteps/add_noise/weight as DiffSynth Z-Image.sh."""
        if use_diffsynth_loop:
            return DiffSynthZImageSchedulerAdapter()
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2

    def _move_main_network(self, device):
        if not hasattr(self, "network") or self.network is None:
            return
        try:
            self.network.to(device)
        except Exception:
            pass

    def _move_sampling_network(self, device):
        if not hasattr(self, "_sampling_network") or self._sampling_network is None:
            return
        try:
            self._sampling_network.to(device)
        except Exception:
            pass

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

        with memory_debug(self.print_and_status_update, "Load components"):
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
            )

        self._raw_dit = components["dit"]
        self.model = _DiTUnetWrapper(self._raw_dit)
        self.vae = components["vae_wrapper"]
        self.text_encoder = [components["text_encoder"]]
        self.tokenizer = [components["tokenizer"]]
        sampling_dit = components.get("sampling_dit")
        # Wrap sampling DiT in the same wrapper as main so LoRA module names match
        # (main uses _DiTUnetWrapper → "dit.noise_refiner..."; unwrapped would be "noise_refiner...").
        self._sampling_transformer = _DiTUnetWrapper(sampling_dit) if sampling_dit is not None else None

        use_diffsynth = getattr(self.model_config, "use_diffsynth_training_loop", True)
        self.noise_scheduler = ZImageDiffSynthModel.get_train_scheduler(use_diffsynth_loop=use_diffsynth)
        self.pipeline = None
        self._move_main_network("cpu")
        self._move_sampling_network("cpu")
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
        **kwargs,
    ):
        # Ensure DiT weights live on the same device as the latents we are
        # about to run on. During training the device-state presets will
        # already have moved the wrapper (and inner DiT) to the correct
        # device; for one-off calls like the smoke test this also brings a
        # freshly-loaded, CPU-resident DiT onto self.device_torch.
        if isinstance(latent_model_input, torch.Tensor):
            target_device = latent_model_input.device
        else:
            target_device = self.device_torch
        try:
            self._raw_dit.to(target_device)
        except Exception:
            # If for some reason .to(...) is not supported on the inner DiT,
            # fall back to its current placement and let the error surface.
            pass
        # Z-Image DiffSynth DiT is trained on latent-space tensors with a fixed
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

        self._move_main_network(target_device)

        use_gradient_checkpointing = getattr(
            self, "gradient_checkpointing", False
        )
        text_embeds = text_embeddings.text_embeds
        if isinstance(text_embeds, torch.Tensor) and len(text_embeds.shape) == 3:
            text_embeds = [text_embeds[i] for i in range(text_embeds.shape[0])]
        # Pass raw DiT to DiffSynth model_fn (expects real DiT with t_embedder, etc.)
        noise_pred = forward_mod.run_forward(
            self._raw_dit,
            latent_model_input,
            timestep,
            text_embeds,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        return noise_pred

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
        return prompt_encoding_mod.encode_prompt(
            tok,
            te,
            prompt,
            self.device_torch,
            dtype=self.torch_dtype,
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
        def _run_generation():
            sc = self.get_bucket_divisibility()
            gen_config.width = int(gen_config.width // sc * sc)
            gen_config.height = int(gen_config.height // sc * sc)
            cond = conditional_embeds.text_embeds
            uncond = unconditional_embeds.text_embeds
            if isinstance(cond, torch.Tensor) and len(cond.shape) == 3:
                cond = [cond[i] for i in range(cond.shape[0])]
            if isinstance(uncond, torch.Tensor) and len(uncond.shape) == 3:
                uncond = [uncond[i] for i in range(uncond.shape[0])]
            img = pipeline(
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
            return img

        use_sampling_transformer = (
            self._sampling_transformer is not None
            and isinstance(self.device_torch, torch.device)
            and self.device_torch.type == "cuda"
        )

        if not use_sampling_transformer:
            return _run_generation()

        try:
            if self._raw_dit is not None:
                self._raw_dit.to("cpu")
            self._move_main_network("cpu")
            self._move_sampling_network(self.device_torch)
            self._sampling_transformer.to(self.device_torch)
            return _run_generation()
        finally:
            self._sampling_transformer.to("cpu")
            self._move_sampling_network("cpu")
            if self._raw_dit is not None:
                self._raw_dit.to(self.device_torch)
            self._move_main_network(self.device_torch)

    def generate_images(
        self,
        image_configs: List[GenerateImageConfig],
        sampler=None,
    ):
        saved_network = None
        use_sampling_transformer = (
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

            if not use_sampling_transformer:
                return super().generate_images(image_configs, sampler)

            try:
                if self._raw_dit is not None:
                    self._raw_dit.to("cpu")
                self._move_main_network("cpu")
                self._move_sampling_network(self.device_torch)
                self._sampling_transformer.to(self.device_torch)
                return super().generate_images(image_configs, sampler)
            finally:
                self._sampling_transformer.to("cpu")
                self._move_sampling_network("cpu")
                if self._raw_dit is not None:
                    self._raw_dit.to(self.device_torch)
                self._move_main_network(self.device_torch)
        finally:
            if saved_network is not None:
                self.network = saved_network

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        import torch
        dit = unwrap_model(self.model)
        if hasattr(dit, "dit"):
            dit = dit.dit
        save_dir = os.path.join(output_path, "transformer")
        os.makedirs(save_dir, exist_ok=True)
        from safetensors.torch import save_file
        state = dit.state_dict()
        save_file(state, os.path.join(save_dir, "model.safetensors"))
        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_base_model_version(self):
        return "zimage_diffsynth"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers", "noise_refiner", "context_refiner"]

    def convert_lora_weights_before_save(self, state_dict):
        return lora_mod.convert_lora_weights_before_save(state_dict)

    def convert_lora_weights_before_load(self, state_dict):
        return lora_mod.convert_lora_weights_before_load(state_dict)
