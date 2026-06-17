# ZImageDiffSynthTrainer: DiffusionTrainer for arch zimage_diffsynth (DiffSynth DiT/forward).

from toolkit.extension import Extension
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
        self.use_diffsynth_training_loop = use_diffsynth_training_loop
        self.use_dynamic_shifting = use_dynamic_shifting

        if use_dynamic_shifting and use_diffsynth_training_loop:
            self.print(
                "ZImage DiffSynth: use_dynamic_shifting is ignored when "
                "use_diffsynth_training_loop is true; set use_diffsynth_training_loop: false "
                "and train.timestep_type: shift for Flux-style dynamic time shifting."
            )
        elif use_dynamic_shifting:
            tt = getattr(tc, "timestep_type", None)
            if tt not in ("shift", "flux_shift"):
                self.print(
                    "ZImage DiffSynth: use_dynamic_shifting requires train.timestep_type "
                    "'shift' or 'flux_shift' (toolkit loop); dynamic mu is only applied in "
                    "that scheduler path."
                )

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
        self.internal_hook_before_train_loop()

    def hook_after_sd_init_before_load(self):
        """
        Called from BaseSDTrainProcess immediately after self.sd is constructed,
        but before sd.load_model(). We use this hook to mark the model as
        flow-matching while preserving DiffusionTrainer's existing behavior.
        """
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

        w = self.sd.noise_scheduler.get_weights_for_timesteps(
            timesteps,
            v2=self.train_config.linear_timesteps2,
            timestep_type=self.train_config.timestep_type,
        )
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


class ZImageDiffSynthTrainerExtension(Extension):
    uid = "z_image_diffsynth_trainer"
    name = "Z-Image DiffSynth Trainer"

    @classmethod
    def get_process(cls):
        return ZImageDiffSynthTrainer
