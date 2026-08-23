# ZImageDiffSynthTrainer: DiffusionTrainer for arch zimage_diffsynth (DiffSynth DiT/forward).

from toolkit.extension import Extension
from toolkit.print import print_acc
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer


def _read_use_diffsynth_training_loop_from_config(config) -> bool:
    use_diffsynth_training_loop = True
    if isinstance(config, dict):
        try:
            model_cfg = config.get("model", {}) or {}
            model_kwargs = model_cfg.get("model_kwargs", {}) or {}
            use_diffsynth_training_loop = model_kwargs.get(
                "use_diffsynth_training_loop", True
            )
        except Exception:
            use_diffsynth_training_loop = True
    return bool(use_diffsynth_training_loop)


def _read_use_dynamic_shifting_from_config(config) -> bool:
    if isinstance(config, dict):
        try:
            model_cfg = config.get("model", {}) or {}
            model_kwargs = model_cfg.get("model_kwargs", {}) or {}
            return bool(model_kwargs.get("use_dynamic_shifting", False))
        except Exception:
            return False
    return False


def _model_kwargs_from_config(config) -> dict:
    if isinstance(config, dict):
        try:
            model_cfg = config.get("model", {}) or {}
            return dict(model_cfg.get("model_kwargs", {}) or {})
        except Exception:
            return {}
    return {}


def _write_model_kwarg(trainer, config, key: str, value) -> None:
    """Persist a model_kwargs entry on both live model_config and raw config dict."""
    try:
        mk = dict(getattr(trainer.model_config, "model_kwargs", None) or {})
        mk[key] = value
        trainer.model_config.model_kwargs = mk
    except Exception:
        pass
    if isinstance(config, dict):
        try:
            model_cfg = config.setdefault("model", {})
            model_kwargs = model_cfg.setdefault("model_kwargs", {})
            model_kwargs[key] = value
        except Exception:
            pass


class ZImageDiffSynthTrainer(DiffusionTrainer):
    """
    Trainer for Z-Image DiffSynth (arch zimage_diffsynth).

    It keeps the generic DiffusionTrainer loop, but hard-wires the parts of
    train_config that should match the original DiffSynth Z-Image training:
    - flow-matching noise scheduler
    - MSE loss on rectified-flow target
    - linear timesteps (used together with DiffSynth-style training weights)
    """

    def __init__(self, process_id, job, config, **kwargs):
        # Let DiffusionTrainer / SDTrainer do their normal initialization first.
        super().__init__(process_id, job, config, **kwargs)

        tc = self.train_config

        cfg = getattr(self, "config", None)
        use_diffsynth_training_loop = _read_use_diffsynth_training_loop_from_config(cfg)
        use_dynamic_shifting = _read_use_dynamic_shifting_from_config(cfg)

        if getattr(tc, "timestep_type", None) == "turbo_prior":
            mk = _model_kwargs_from_config(cfg)
            try:
                mk_live = dict(getattr(self.model_config, "model_kwargs", None) or {})
                mk = {**mk, **mk_live}
            except Exception:
                pass

            if "use_diffsynth_prompt_encoding" in mk:
                if not bool(mk["use_diffsynth_prompt_encoding"]):
                    raise ValueError(
                        "timestep_type=turbo_prior requires use_diffsynth_prompt_encoding=true "
                        "(explicit false is not allowed)."
                    )
            else:
                _write_model_kwarg(self, cfg, "use_diffsynth_prompt_encoding", True)

            if use_diffsynth_training_loop:
                raise ValueError(
                    "timestep_type=turbo_prior requires use_diffsynth_training_loop=false "
                    "(toolkit TimestepSampler); true is not allowed."
                )

            if getattr(tc, "content_or_style", None) in (
                "gaussian",
                "gaussian_bimodal",
            ):
                raise ValueError(
                    "timestep_type=turbo_prior is incompatible with content_or_style="
                    f"{tc.content_or_style!r}; use balanced (or another non-gaussian mode)."
                )

            if use_dynamic_shifting:
                raise ValueError(
                    "timestep_type=turbo_prior requires use_dynamic_shifting=false "
                    "(official Turbo uses static shift); true is not allowed."
                )

        self.use_diffsynth_training_loop = use_diffsynth_training_loop
        self._requested_use_dynamic_shifting = use_dynamic_shifting
        self.use_dynamic_shifting = use_dynamic_shifting
        # Set in _aggregate_flow_matching_mse_loss when DiffSynth weighting is already applied.
        self._skip_post_timestep_weighting_once = False

        # Always train Z-Image in flow-matching mode with 1000 train timesteps.
        # We let ZImageDiffSynthModel.get_train_scheduler() provide the actual
        # DiffSynth-compatible scheduler via BaseSDTrainProcess when
        # train_config.noise_scheduler is None.
        tc.noise_scheduler = None
        tc.num_train_timesteps = getattr(tc, "num_train_timesteps", 1000) or 1000

        if use_diffsynth_training_loop:
            self.print("ZImage DiffSynth: using DiffSynth training loop (linear timesteps, MSE, no SNR reweighting).")
            # Use MSE loss on rectified-flow target (noise - latents) as in DiffSynth.
            tc.loss_type = "mse"

            # Let our FlowMatch scheduler control weighting; keep timesteps linear.
            tc.timestep_type = "linear"
            # True: keeps train_config aligned with DiffSynth-style timestep weights (same family as
            # Z-Image.sh / FlowMatchSFTLoss). In the usual MSE path, weights are applied inside
            # ZImageDiffSynthTrainer._aggregate_flow_matching_mse_loss → aggregate_flow_matching_mse_diffsynth
            # (get_weights_for_timesteps, then multiply per batch after spatial mean)—not only via SDTrainer._apply.
            # If we fall back to super()._aggregate_flow_matching_mse_loss (e.g. do_prior_divergence), SDTrainer
            # uses _apply_flow_timestep_element_weights, which also keys off linear_timesteps for flow matching.
            tc.linear_timesteps = True
            tc.linear_timesteps2 = False

            # Disable SNR re-weighting — DiffSynth already applies its own weighting.
            tc.snr_gamma = None
            tc.min_snr_gamma = None
        else:
            self.print("ZImage DiffSynth: not using DiffSynth training loop (respecting train_config: timestep_type, content_or_style, SNR).")
            # In toolkit-loop mode, the shared BatchProcessor chooses how it prepares
            # the training timestep tensor based on `train_config.noise_scheduler`
            # string. Without overriding the default `None`, it would fall back to
            # the inference-style `set_timesteps()` path and produce a different
            # timestep grid than `TimestepSampler`/gaussian weights expect.
            tc.noise_scheduler = "flowmatch"
        self._refresh_dynamic_shifting_runtime(log_transitions=True)

    def _set_use_dynamic_shifting_state(self, enabled: bool):
        enabled = bool(enabled)
        self.use_dynamic_shifting = enabled

        try:
            mk = dict(getattr(self.model_config, "model_kwargs", None) or {})
            mk["use_dynamic_shifting"] = enabled
            self.model_config.model_kwargs = mk
        except Exception:
            pass

        cfg = getattr(self, "config", None)
        if isinstance(cfg, dict):
            try:
                model_cfg = cfg.setdefault("model", {})
                model_kwargs = model_cfg.setdefault("model_kwargs", {})
                model_kwargs["use_dynamic_shifting"] = enabled
            except Exception:
                pass

        sd = getattr(self, "sd", None)
        scheduler = getattr(sd, "noise_scheduler", None)
        if scheduler is None:
            return
        try:
            if hasattr(scheduler, "config"):
                scheduler.config.use_dynamic_shifting = enabled
        except Exception:
            pass
        try:
            if hasattr(scheduler, "use_dynamic_shifting"):
                scheduler.use_dynamic_shifting = enabled
        except Exception:
            pass

    def _refresh_dynamic_shifting_runtime(self, *, log_transitions: bool = False):
        requested = bool(getattr(self, "_requested_use_dynamic_shifting", False))
        timestep_type = getattr(self.train_config, "timestep_type", None)
        allow_dynamic_shifting = (
            (not self.use_diffsynth_training_loop)
            and timestep_type in ("shift", "flux_shift")
        )
        should_enable = requested and allow_dynamic_shifting
        previous = bool(getattr(self, "use_dynamic_shifting", False))

        if previous != should_enable:
            self._set_use_dynamic_shifting_state(should_enable)
            if requested:
                if should_enable:
                    print_acc(
                        "ZImage DiffSynth: enabling use_dynamic_shifting for training and sampling "
                        f"(use_diffsynth_training_loop={self.use_diffsynth_training_loop}, "
                        f"train.timestep_type={timestep_type!r})."
                    )
                else:
                    print_acc(
                        "ZImage DiffSynth: ignoring use_dynamic_shifting for both training and sampling "
                        f"(use_diffsynth_training_loop={self.use_diffsynth_training_loop}, "
                        f"train.timestep_type={timestep_type!r}; requires 'shift' or 'flux_shift')."
                    )
            return

        if log_transitions and requested and not should_enable:
            print_acc(
                "ZImage DiffSynth: ignoring use_dynamic_shifting for both training and sampling "
                f"(use_diffsynth_training_loop={self.use_diffsynth_training_loop}, "
                f"train.timestep_type={timestep_type!r}; requires 'shift' or 'flux_shift')."
            )

    def apply_runtime_timestep_type(self):
        super().apply_runtime_timestep_type()
        self._refresh_dynamic_shifting_runtime()

    def hook_before_train_loop(self):
        """
        Run SDTrainer.setup (including optional text-encoder unload), then place main vs sampling
        transformers before DiffusionTrainer clears runtime DB and before baseline sampling /
        ``set_device_state`` at loop start.
        """
        super(DiffusionTrainer, self).hook_before_train_loop()
        sd = getattr(self, "sd", None)
        if sd is not None:
            dev = self.device_torch
            sd._move_main_network(dev)
            if getattr(self, "_compile_dit_blocks", False) and hasattr(
                sd, "compile_dit_blocks"
            ):
                sd.compile_dit_blocks()
        self.internal_hook_before_train_loop()

    def hook_after_sd_init_before_load(self):
        """
        Called from BaseSDTrainProcess immediately after self.sd is constructed,
        but before sd.load_model(). We use this hook to mark the model as
        flow-matching while preserving DiffusionTrainer's existing behavior.
        """
        # Defer per-block DiT compile to hook_before_train_loop (after quantize + LoRA).
        # Suppress BaseSDTrainProcess whole-unet torch.compile which is ineffective here.
        self._compile_dit_blocks = bool(
            getattr(self.model_config, "compile", False)
        )
        if self._compile_dit_blocks:
            self.model_config.compile = False

        # Preserve DiffusionTrainer logic (status updates, hooks, etc.)
        super().hook_after_sd_init_before_load()

        # Make sure the model is treated as flow-matching by SDTrainer logic.
        sd = getattr(self, "sd", None)
        if sd is not None and hasattr(sd, "is_flow_matching"):
            sd.is_flow_matching = True

    def _aggregate_flow_matching_mse_loss(
        self,
        pred,
        target,
        timesteps,
        mask_multiplier,
        noise_pred,
        prior_pred,
        batch,
    ):
        if not self.use_diffsynth_training_loop:
            self._skip_post_timestep_weighting_once = False
            return super()._aggregate_flow_matching_mse_loss(
                pred,
                target,
                timesteps,
                mask_multiplier,
                noise_pred,
                prior_pred,
                batch,
            )
        if self.train_config.do_prior_divergence and prior_pred is not None:
            self._skip_post_timestep_weighting_once = False
            return super()._aggregate_flow_matching_mse_loss(
                pred,
                target,
                timesteps,
                mask_multiplier,
                noise_pred,
                prior_pred,
                batch,
            )
        from . import diffsynth_training as dst

        weighting_scheme = (
            "weighted" if self.train_config.timestep_weighting == "weighted" else "linear"
        )
        w = self.sd.noise_scheduler.get_weights_for_timesteps(
            timesteps,
            v2=self.train_config.linear_timesteps2,
            timestep_type=weighting_scheme,
        )
        # This path applies DiffSynth timestep weights inside aggregate_flow_matching_mse_diffsynth.
        # Skip the generic post-loss weighting pass in SDTrainer.calculate_loss for this batch.
        self._skip_post_timestep_weighting_once = True
        return dst.aggregate_flow_matching_mse_diffsynth(
            pred,
            target,
            timesteps,
            w,
            mask_multiplier,
            noise_pred,
            train_turbo=self.train_config.train_turbo,
            log_writer=self.writer,
            step_num=self.step_num,
            is_main_process=self.accelerator.is_main_process,
            log_every=getattr(self.logging_config, "log_every", None),
        )

    def _apply_flow_timestep_element_weights(self, loss, timesteps):
        if getattr(self, "_skip_post_timestep_weighting_once", False):
            self._skip_post_timestep_weighting_once = False
            return loss
        return super()._apply_flow_timestep_element_weights(loss, timesteps)


class ZImageDiffSynthTrainerExtension(Extension):
    uid = "z_image_diffsynth_trainer"
    name = "Z-Image DiffSynth Trainer"

    @classmethod
    def get_process(cls):
        return ZImageDiffSynthTrainer
