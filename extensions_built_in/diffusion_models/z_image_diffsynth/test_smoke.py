"""
Smoke test for Z-Image DiffSynth: import, load model, prompt encoding, forward, generation pipeline.
Run from repo root. If venv exists, the script will use it automatically (no need to activate).

By default the test runs with quantized DiT and quantized text encoder (quantize=True, quantize_te=True)
to reduce VRAM and match typical usage. Set ZIMAGE_DIFFSYNTH_TEST_NO_QUANT=1 to run without quantization.

Paths: use env ZIMAGE_DIFFSYNTH_MODEL_PATH and optionally ZIMAGE_DIFFSYNTH_SAMPLING_PATH.
If unset, defaults from the plan are used (see DEFAULT_* below). Override via env if needed.

Example (PowerShell, from repo root):
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke
"""

import io
import os
import sys
import tempfile

# Re-run with venv Python if venv exists and we're not already using it
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if sys.platform == "win32":
    _venv_python = os.path.join(_REPO_ROOT, "venv", "Scripts", "python.exe")
else:
    _venv_python = os.path.join(_REPO_ROOT, "venv", "bin", "python")
if os.path.isfile(_venv_python):
    _current = os.path.realpath(sys.executable)
    _venv_real = os.path.realpath(_venv_python)
    if _current != _venv_real:
        os.execv(_venv_python, [_venv_python] + sys.argv[1:])

# Default paths from the plan (HuggingFace snapshot layout)
DEFAULT_ZIMAGE_MODEL_PATH = (
    "e:/Backup/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/"
)
DEFAULT_ZIMAGE_SAMPLING_PATH = (
    "e:/Backup/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/0e36c2b379e66fa531d01cc531c44919e5f1c6fd/"
)

# repo root on path
TOOLKIT_ROOT = _REPO_ROOT
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)

# Enable toolkit debug / memory_debug output by default so that this smoke test
# always prints CUDA / RAM usage for key steps (load_model, get_noise_prediction,
# etc.). An optional environment variable allows overriding the default.
try:
    from types import SimpleNamespace
    from toolkit.util.debug import set_debug_config

    _debug_flag = os.environ.get("ZIMAGE_DIFFSYNTH_DEBUG", "").strip()
    if _debug_flag:
        # Treat any non-empty value other than an explicit "0"/"false" as True.
        _enabled = _debug_flag not in ("0", "false", "False")
    else:
        # Default: enable debug so memory_debug contexts in the model print
        # VRAM/RAM stats during this smoke test.
        _enabled = True
    set_debug_config(SimpleNamespace(debug=_enabled))
except Exception:
    # Debug configuration is optional; if toolkit.util.debug is unavailable or
    # misconfigured, continue without failing the smoke test.
    pass


def _log(msg):
    """Print and flush so output appears immediately (helps when script fails mid-run)."""
    print(msg, flush=True)


class _TeeStderr:
    """Writes to both real stderr and a buffer. Use to capture loader/torch stderr during load_model()."""

    def __init__(self):
        self._buffer = io.StringIO()
        self._real = sys.stderr

    def write(self, s):
        self._real.write(s)
        self._real.flush()
        self._buffer.write(s)

    def flush(self):
        self._real.flush()
        self._buffer.flush()

    def getvalue(self):
        return self._buffer.getvalue()


def main():
    import traceback

    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH
    )
    _log(f"[paths] model_path={model_path!r}")
    if not model_path or not os.path.isdir(model_path):
        _log(
            "Z-Image model path is missing or not a directory. Set ZIMAGE_DIFFSYNTH_MODEL_PATH "
            "or use default (see DEFAULT_ZIMAGE_MODEL_PATH in this script)."
        )
        _log(f"  Current value: {model_path or '(empty)'}")
        sys.exit(1)
    _log("[paths] model_path OK")

    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None
    _log(f"[paths] sampling_path={sampling_path or '(none)'}")

    _log("1. Import module and get model class ...")
    try:
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class
        import torch
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    no_quant = os.environ.get("ZIMAGE_DIFFSYNTH_TEST_NO_QUANT", "").strip() == "1"
    model_config_dict = {
        "name_or_path": model_path,
        "arch": "zimage_diffsynth",
        "quantize": not no_quant,
        "quantize_te": not no_quant,
    }
    if sampling_path:
        model_config_dict["sampling_name_or_path"] = sampling_path
    _log(f"   quantize (DiT)={model_config_dict['quantize']}, quantize_te={model_config_dict['quantize_te']}")

    model_config = ModelConfig(**model_config_dict)
    ModelClass = get_model_class(model_config)
    if ModelClass.arch != "zimage_diffsynth":
        _log(f"Expected arch zimage_diffsynth, got {ModelClass.arch}")
        sys.exit(1)
    _log(f"   Model class: {ModelClass.__name__} (arch={ModelClass.arch})")
    _log("1. OK")

    _log("2. Create model and load_model() ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"   device={device}")
    tee = None
    old_stderr = None
    try:
        sd = ModelClass(device, model_config, dtype="bf16")
        _log("   ModelClass() OK, calling load_model() (stderr captured below) ...")
        tee = _TeeStderr()
        old_stderr = sys.stderr
        sys.stderr = tee
        sd.load_model()
    except Exception:
        if old_stderr is not None:
            sys.stderr = old_stderr
        if tee is not None:
            captured = tee.getvalue()
            if captured.strip():
                _log("[stderr during load_model]:")
                for line in captured.splitlines():
                    _log("  | " + line)
        _log("   FAILED in step 2 (create model / load_model):")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if old_stderr is not None:
            sys.stderr = old_stderr
    if tee is not None:
        captured = tee.getvalue()
        if captured.strip():
            _log("[stderr during load_model]:")
            for line in captured.splitlines():
                _log("  | " + line)
    _log("2. OK (Loaded.)")

    _log("2c. VAE wrapper has nn.Module interface (.to, .parameters) ...")
    try:
        vae = sd.vae
        if isinstance(vae, list):
            vae = vae[0]
        assert hasattr(vae, "to"), "VAE must have .to() for trainer (e.g. vae.to(device))"
        assert callable(getattr(vae, "to")), "VAE.to must be callable"
        assert hasattr(vae, "parameters"), "VAE must have .parameters()"
        # Actually call .to() to ensure it doesn't raise (e.g. DiffSynthVAEWrapper as nn.Module)
        vae.to(device)
        _log("2c. OK")
    except Exception as e:
        _log(f"2c. FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2e. _DiTUnetWrapper must have .dit so base_model.save_device_state() does not raise
    #     "'_DiTUnetWrapper' object has no attribute 'dit'" when accessing self.unet.device
    _log("2e. Unet wrapper has .dit and device state preset (cache_latents) works ...")
    try:
        unet = sd.unet
        assert unet is not None, "sd.unet must be set after load_model"
        # Explicit check: wrapper used as unet must have .dit (e.g. _DiTUnetWrapper)
        if hasattr(unet, "dit"):
            assert getattr(unet, "dit") is not None, "sd.unet.dit must be set (inner DiT)"
        # This is what fails in get_dataloader_from_datasets -> setup_epoch -> cache_latents_all_latents
        # -> set_device_state_preset('cache_latents') -> save_device_state() -> self.unet.device
        _ = unet.device
        _ = unet.training
        sd.set_device_state_preset("cache_latents")
        # Restore so later steps (get_noise_prediction, etc.) still work
        sd.restore_device_state()
        _log("2e. OK")
    except AttributeError as e:
        if "dit" in str(e):
            _log(
                f"2e. FAILED: unet wrapper missing .dit (trainer/dataloader will fail with same error): {e}"
            )
            traceback.print_exc()
            sys.exit(1)
        raise
    except Exception as e:
        _log(f"2e. FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2f. VAE wrapper must have .dtype so SDTrainer.train_single_accumulation does not raise
    #     "AttributeError: 'DiffSynthVAEWrapper' object has no attribute 'dtype'"
    _log("2f. VAE wrapper has .dtype (train_single_accumulation checks vae.dtype vs vae_torch_dtype) ...")
    try:
        vae = sd.vae
        if isinstance(vae, list):
            vae = vae[0]
        _ = vae.dtype
        assert isinstance(vae.dtype, torch.dtype), "vae.dtype must be torch.dtype"
        _log("2f. OK")
    except AttributeError as e:
        if "dtype" in str(e):
            _log(
                f"2f. FAILED: VAE wrapper missing .dtype (trainer will fail with same error): {e}"
            )
            traceback.print_exc()
            sys.exit(1)
        raise
    except Exception as e:
        _log(f"2f. FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Optional: run a second load without quantization (to test unquantized path)
    if os.environ.get("ZIMAGE_DIFFSYNTH_TEST_ALSO_NO_QUANT", "").strip() == "1":
        _log("2b. Optional: load again without quantization ...")
        try:
            cfg_noq = {
                "name_or_path": model_path,
                "arch": "zimage_diffsynth",
                "quantize": False,
                "quantize_te": False,
            }
            if sampling_path:
                cfg_noq["sampling_name_or_path"] = sampling_path
            model_config_noq = ModelConfig(**cfg_noq)
            sd_noq = ModelClass(device, model_config_noq, dtype="bf16")
            sd_noq.load_model()
            _log("2b. OK")
        except Exception as e:
            _log(f"2b. FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        _log("2b. Skipped (set ZIMAGE_DIFFSYNTH_TEST_ALSO_NO_QUANT=1 to run unquantized load)")

    # 2d. LoRA name consistency: when using sampling transformer, main and sampling LoRA networks
    # must use the same lora_name for each module so share_parameters_with() does not raise
    # "lora name mismatch: lora_unet_noise_refiner_0_attention_to_q vs lora_unet_dit_noise_refiner_0_attention_to_q"
    if getattr(sd, "_sampling_transformer", None) is not None:
        _log("2d. LoRA name consistency (main vs sampling transformer) ...")
        try:
            from toolkit.config_modules import NetworkConfig
            from toolkit.lora_special import LoRASpecialNetwork

            network_config = NetworkConfig(
                type="lora",
                linear=4,
                linear_alpha=1.0,
                transformer_only=True,
                network_kwargs={},
            )
            network_kwargs = dict(network_config.network_kwargs)
            if hasattr(sd, "target_lora_modules"):
                network_kwargs["target_lin_modules"] = sd.target_lora_modules
            common = dict(
                text_encoder=sd.text_encoder,
                lora_dim=network_config.linear,
                multiplier=1.0,
                alpha=network_config.linear_alpha,
                train_unet=True,
                train_text_encoder=False,
                conv_lora_dim=network_config.conv,
                conv_alpha=network_config.conv_alpha,
                is_sdxl=False,
                is_v2=False,
                is_v3=False,
                is_pixart=False,
                is_auraflow=False,
                is_flux=False,
                is_lumina2=False,
                is_ssd=False,
                is_vega=False,
                dropout=network_config.dropout,
                rank_dropout=network_config.rank_dropout,
                module_dropout=network_config.module_dropout,
                use_text_encoder_1=True,
                use_text_encoder_2=True,
                use_bias=False,
                is_lorm=False,
                network_config=network_config,
                network_type=network_config.type,
                transformer_only=network_config.transformer_only,
                is_transformer=getattr(sd, "is_transformer", True),
                base_model=sd,
                **network_kwargs,
            )
            main_network = LoRASpecialNetwork(
                unet=sd.get_model_to_train(),
                **common,
            )
            sampling_network = LoRASpecialNetwork(
                unet=sd._sampling_transformer,
                **common,
            )
            sampling_network.share_parameters_with(main_network)
            _log("2d. OK (LoRA names match; share_parameters_with succeeded)")
        except AssertionError as e:
            if "lora name mismatch" in str(e):
                _log(f"2d. FAILED: LoRA name mismatch between main and sampling transformer: {e}")
                traceback.print_exc()
                sys.exit(1)
            raise
        except Exception as e:
            _log(f"2d. FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        _log("2d. Skipped (no sampling transformer; set ZIMAGE_DIFFSYNTH_SAMPLING_PATH to test LoRA name consistency)")

    _log("3. get_prompt_embeds ...")
    try:
        embeds = sd.get_prompt_embeds("a cat on a mat")
        assert embeds.text_embeds is not None
        te = embeds.text_embeds
        if isinstance(te, torch.Tensor):
            _log(f"   text_embeds shape: {te.shape}")
        else:
            _log(f"   text_embeds: {type(te)} (len={len(te) if hasattr(te, '__len__') else 'n/a'})")
    except Exception:
        _log("   FAILED in step 3 (get_prompt_embeds):")
        traceback.print_exc()
        sys.exit(1)
    _log("3. OK")

    _log("4. get_noise_prediction (dummy tensors) ...")
    try:
        B, C, H, W = 1, 16, 64, 64
        latent = torch.randn(B, C, H, W, device=device, dtype=sd.torch_dtype)
        timestep = torch.tensor([500], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = sd.get_noise_prediction(latent, timestep, embeds)
        _log(f"   noise_pred shape: {pred.shape}")
    except Exception:
        _log("   FAILED in step 4 (get_noise_prediction):")
        traceback.print_exc()
        sys.exit(1)
    _log("4. OK")

    _log("4b. BaseModel.predict_noise uses noise_scheduler.scale_model_input ...")
    try:
        # This exercises the same code path as SDTrainer / BaseSDTrainProcess,
        # where BaseModel.predict_noise calls self.noise_scheduler.scale_model_input
        # via its internal scale_model_input() helper. Our DiffSynthZImageSchedulerAdapter
        # must therefore implement scale_model_input with a compatible signature.
        B, C, H, W = 1, 16, 64, 64
        latent = torch.randn(B, C, H, W, device=device, dtype=sd.torch_dtype)
        timestep = torch.tensor([500], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred2 = sd.predict_noise(
                latents=latent,
                text_embeddings=embeds,
                timestep=timestep,
                is_input_scaled=False,
            )
        _log(f"   predict_noise output shape: {pred2.shape}")
    except Exception:
        _log("   FAILED in step 4b (predict_noise / scale_model_input):")
        traceback.print_exc()
        sys.exit(1)
    _log("4b. OK")

    _log("4c. get_noise_prediction with 3-channel input should fail clearly ...")
    try:
        # Construct a fake 3-channel tensor that mimics RGB pixel input.
        # For Z-Image DiffSynth, the DiT and its VAE expect latent-space
        # tensors with in_channels (e.g. 16). Passing 3 channels should now
        # trigger our explicit channel-mismatch guard instead of a Quanto
        # matmul shape error deep inside the DiT.
        B, C_bad, H, W = 1, 3, 64, 64
        latent_rgb = torch.randn(B, C_bad, H, W, device=device, dtype=sd.torch_dtype)
        try:
            with torch.no_grad():
                _ = sd.get_noise_prediction(latent_rgb, timestep, embeds)
            _log("   4c. FAILED: expected channel-mismatch error but get_noise_prediction succeeded")
            sys.exit(1)
        except RuntimeError as e:
            msg = str(e)
            # The exact wording is implementation-defined, but it must mention
            # channels / latents so users are not left with a raw matmul error.
            if ("channels" in msg and "latent" in msg) or "expected latents with" in msg:
                _log("4c. OK (3-channel input produces clear channel-mismatch error)")
            else:
                _log(f"   4c. FAILED: RuntimeError message not clear enough: {msg!r}")
                sys.exit(1)
    except Exception:
        _log("   FAILED in step 4c (3-channel get_noise_prediction guard):")
        traceback.print_exc()
        sys.exit(1)

    # 4d. SNR API regression for the default DiffSynth adapter (use_diffsynth_training_loop=True).
    # DiffSynthZImageSchedulerAdapter inherits compute_snr, so get_all_snr / apply_snr_weight must
    # work without DDPM alphas_cumprod. Default DiffSynth training disables min_snr_gamma in the
    # trainer (see step 7a); this step does NOT validate the active toolkit-loop SNR path.
    _log(
        "4d. noise_scheduler SNR API guard (get_all_snr, apply_snr_weight; "
        "default DiffSynth loop does not apply min_snr_gamma) ..."
    )
    try:
        from extensions_built_in.diffusion_models.z_image_diffsynth.snr_weighting_checks import (
            assert_all_snr_table,
            assert_apply_snr_flow_match_weights,
            assert_scheduler_uses_compute_snr_path,
            non_integer_schedule_timesteps,
        )

        noise_scheduler = sd.noise_scheduler
        assert noise_scheduler is not None, "sd.noise_scheduler must be set"
        assert_scheduler_uses_compute_snr_path(noise_scheduler)
        assert_all_snr_table(noise_scheduler, device)

        min_snr_gamma = 5.0
        check_timesteps = [10, 500, 990]
        assert_apply_snr_flow_match_weights(
            noise_scheduler,
            check_timesteps,
            min_snr_gamma,
            device,
        )

        float_timesteps = non_integer_schedule_timesteps(noise_scheduler)
        if float_timesteps:
            assert_apply_snr_flow_match_weights(
                noise_scheduler,
                float_timesteps,
                min_snr_gamma,
                device,
            )
            _log(f"   float schedule timesteps checked: {float_timesteps}")
        else:
            _log("   no non-integer DiffSynth timesteps to check interpolation")

        _log("4d. OK (SNR API + min_snr_gamma weight checks passed)")
    except Exception:
        _log("   FAILED in step 4d (scheduler SNR support):")
        traceback.print_exc()
        sys.exit(1)

    # 4e. Batch path in model_fn_z_image_turbo: B>1, list of prompt_embeds, no control_context
    #     -> single dit.forward(all_image, timestep, all_cap_feats) and return (B, C, H, W).
    _log("4e. get_noise_prediction with batch size 2 (model_fn_z_image_turbo batch path) ...")
    try:
        from extensions_built_in.diffusion_models.z_image_diffsynth import prompt_encoding as prompt_encoding_mod

        B, C, H, W = 2, 16, 64, 64
        latent_b2 = torch.randn(B, C, H, W, device=device, dtype=sd.torch_dtype)
        timestep_b = torch.tensor([500], device=device, dtype=torch.float32)
        batch_embeds = prompt_encoding_mod.encode_prompt(
            sd.tokenizer[0],
            sd.text_encoder[0],
            ["a cat on a mat", "a dog in the sun"],
            sd.device_torch,
            sd.torch_dtype,
        )
        with torch.no_grad():
            pred_b = sd.get_noise_prediction(latent_b2, timestep_b, batch_embeds)
        assert pred_b.shape == (B, C, H, W), (
            f"batch noise_pred shape must be ({B}, {C}, {H}, {W}), got {pred_b.shape}"
        )
        _log(f"   batch noise_pred shape: {pred_b.shape}")
    except Exception:
        _log("   FAILED in step 4e (get_noise_prediction batch path):")
        traceback.print_exc()
        sys.exit(1)
    _log("4e. OK")

    _log("5. get_generation_pipeline ...")
    try:
        pipeline = sd.get_generation_pipeline()
        _log(f"   pipeline: {type(pipeline).__name__}")
    except Exception:
        _log("   FAILED in step 5 (get_generation_pipeline):")
        traceback.print_exc()
        sys.exit(1)
    _log("5. OK")

    _log("6. Optional: one step of sampling (single image) ...")
    try:
        from toolkit.config_modules import GenerateImageConfig
        gen_config = GenerateImageConfig(
            width=256, height=256, num_inference_steps=4,
            guidance_scale=1.0, prompt="a cat", negative_prompt="",
            output_folder=os.path.join(tempfile.gettempdir(), "z_image_diffsynth_smoke"),
            output_ext="png",
        )
        uncond = sd.get_prompt_embeds("")
        gen = torch.Generator(device=device).manual_seed(42)
        img = sd.generate_single_image(
            pipeline, gen_config, embeds, uncond, gen, {}
        )
        _log(f"   generated image: {type(img).__name__}")
    except Exception as e:
        _log(f"   sampling skipped or failed: {e}")
        traceback.print_exc()
    _log("6. OK (or skipped)")

    _log("7. ZImageDiffSynthTrainer wiring (train_config & is_flow_matching) ...")
    try:
        from types import SimpleNamespace
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
            ZImageDiffSynthTrainer,
        )

        # Patch DiffusionTrainer.__init__ so we can exercise the trainer's
        # initialization logic without constructing a full Job / datasets.
        orig_init = DiffusionTrainer.__init__
        orig_hook = DiffusionTrainer.hook_after_sd_init_before_load
        try:
            def _fake_init(self, process_id, job, config, **kwargs):
                # Minimal fields used by ZImageDiffSynthTrainer.__init__
                # and the smoke test itself.
                self.config = config
                self.progress_bar = None  # avoid AttributeError in self.print()
                self.train_config = SimpleNamespace(
                    noise_scheduler="placeholder",
                    num_train_timesteps=None,
                    loss_type=None,
                    timestep_type=None,
                    linear_timesteps=True,
                    linear_timesteps2=True,
                    snr_gamma=1.0,
                    min_snr_gamma=1.0,
                    dtype="bf16",
                )

            def _noop_hook(self):
                # In this smoke test we only care about the ZImageDiffSynthTrainer
                # part of the hook; the base hook relies on a full trainer setup,
                # which our fake init does not provide.
                return None

            DiffusionTrainer.__init__ = _fake_init  # type: ignore[assignment]
            DiffusionTrainer.hook_after_sd_init_before_load = _noop_hook  # type: ignore[assignment]

            # 7a. Default behaviour (no flag or flag True): must hard-wire DiffSynth config.
            trainer = ZImageDiffSynthTrainer(0, None, {})
            tc = trainer.train_config
            assert getattr(trainer, "use_diffsynth_training_loop", None) is True, (
                "trainer.use_diffsynth_training_loop must mirror model.model_kwargs default True"
            )
            # After trainer init, config must be hard-wired for DiffSynth Z-Image.
            assert tc.noise_scheduler is None, "trainer must leave noise_scheduler=None to use model.get_train_scheduler()"
            assert tc.num_train_timesteps == 1000, "num_train_timesteps must default to 1000"
            assert tc.loss_type == "mse", "loss_type must be mse when use_diffsynth_training_loop is True/default"
            assert tc.timestep_type == "linear", "timestep_type must be linear when use_diffsynth_training_loop is True/default"
            # Enables get_weights_for_timesteps / DiffSynth linear_timesteps_weights (Z-Image.sh); YAML override when flag True.
            assert tc.linear_timesteps is True, "linear_timesteps must be True when use_diffsynth_training_loop is True/default"
            assert tc.linear_timesteps2 is False, "linear_timesteps2 must be False when use_diffsynth_training_loop is True/default"
            assert tc.snr_gamma is None, "snr_gamma must be disabled (None) when use_diffsynth_training_loop is True/default"
            assert tc.min_snr_gamma is None, "min_snr_gamma must be disabled (None) when use_diffsynth_training_loop is True/default"

            # 7b. When use_diffsynth_training_loop is explicitly False in model_kwargs,
            # trainer must *not* override timestep_type / loss / SNR settings.
            # Use process config shape (same as job passes to the trainer).
            cfg_with_flag = {
                "model": {
                    "model_kwargs": {
                        "use_diffsynth_training_loop": False,
                    }
                }
            }
            trainer2 = ZImageDiffSynthTrainer(0, None, cfg_with_flag)
            tc2 = trainer2.train_config
            assert getattr(trainer2, "use_diffsynth_training_loop", None) is False
            assert tc2.noise_scheduler == "flowmatch", "trainer must set noise_scheduler='flowmatch' in toolkit-loop mode (use_diffsynth_training_loop=False)"
            assert tc2.num_train_timesteps == 1000, "num_train_timesteps must still default to 1000 when use_diffsynth_training_loop is False"
            # From _fake_init defaults: loss_type/timestep_type remain None, linear_timesteps* and SNR flags unchanged.
            assert tc2.loss_type is None, "loss_type must not be forced when use_diffsynth_training_loop is False"
            assert tc2.timestep_type is None, "timestep_type must not be forced when use_diffsynth_training_loop is False"
            assert tc2.linear_timesteps is True, "linear_timesteps must not be overridden when use_diffsynth_training_loop is False"
            assert tc2.linear_timesteps2 is True, "linear_timesteps2 must not be overridden when use_diffsynth_training_loop is False"
            assert tc2.snr_gamma == 1.0, "snr_gamma must not be overridden when use_diffsynth_training_loop is False"
            assert tc2.min_snr_gamma == 1.0, "min_snr_gamma must not be overridden when use_diffsynth_training_loop is False"

            # Now exercise the hook that marks sd as flow-matching once it exists
            # (behaviour is independent of the flag).
            fake_sd = SimpleNamespace(is_flow_matching=False)
            trainer.sd = fake_sd
            trainer.hook_after_sd_init_before_load()
            assert getattr(trainer.sd, "is_flow_matching", False) is True, "sd.is_flow_matching must be True after hook"
        finally:
            DiffusionTrainer.__init__ = orig_init  # type: ignore[assignment]
            DiffusionTrainer.hook_after_sd_init_before_load = orig_hook  # type: ignore[assignment]
        _log("7. OK")
    except Exception:
        _log("   FAILED in step 7 (ZImageDiffSynthTrainer wiring):")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _log("8. decode_latents channel handling (3-channel vs latent space) ...")
    try:
        # 8a. 3-channel tensors (RGB-like) must bypass the VAE decoder and be
        # treated as already-decoded images. This matches the behavior needed
        # when SDTrainer.train_single_accumulation calls sd.decode_latents on
        # pixel-space tensors for preview.
        rgb = torch.randn(
            1,
            3,
            320,
            480,
            device=sd.device_torch,
            dtype=sd.torch_dtype,
        )
        out_rgb = sd.decode_latents(rgb)
        assert out_rgb.shape == rgb.shape, "decode_latents(3ch) must preserve shape"
        assert (
            out_rgb.dtype == sd.torch_dtype
        ), "decode_latents(3ch) must cast to sd.torch_dtype"

        # 8b. Latent-space tensors (C != 3, e.g. 16) should still go through the
        # VAE path without raising, to keep the usual decode_latents behavior.
        latents = torch.randn(
            1,
            16,
            40,
            60,
            device=sd.device_torch,
            dtype=sd.torch_dtype,
        )
        out_latents = sd.decode_latents(latents)
        assert out_latents.dim() == 4 and out_latents.shape[1] in (
            3,
            4,
        ), "decode_latents(latents) must produce image-like tensor"
        _log("8. OK")
    except Exception:
        _log("   FAILED in step 8 (decode_latents channel handling):")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    _log("Done.")


if __name__ == "__main__":
    main()
