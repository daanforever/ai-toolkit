# ZImageDiffSynthTrainer: DiffusionTrainer for arch zimage_diffsynth (DiffSynth DiT/forward).

from toolkit.extension import Extension
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer


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

        # Decide whether to use the original DiffSynth training loop behaviour.
        # Default is True for backwards compatibility; when model_kwargs contains
        # use_diffsynth_training_loop: false we fall back to the generic toolkit
        # behaviour (respect timestep_type, content_or_style, SNR settings, etc.).
        use_diffsynth_training_loop = True
        cfg = getattr(self, "config", None)
        if isinstance(cfg, dict):
            try:
                # self.config is the current process config (one element of job config.process).
                model_cfg = cfg.get("model", {}) or {}
                model_kwargs = model_cfg.get("model_kwargs", {}) or {}
                use_diffsynth_training_loop = model_kwargs.get(
                    "use_diffsynth_training_loop", True
                )
            except Exception:
                # On any unexpected shape, keep the default (True) so existing
                # configs and smoke tests remain unchanged.
                use_diffsynth_training_loop = True

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
            tc.linear_timesteps = False
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


class ZImageDiffSynthTrainerExtension(Extension):
    uid = "z_image_diffsynth_trainer"
    name = "Z-Image DiffSynth Trainer"

    @classmethod
    def get_process(cls):
        return ZImageDiffSynthTrainer
