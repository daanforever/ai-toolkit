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
        super().__init__(process_id, job, config, **kwargs)

        tc = self.train_config

        # Always train Z-Image in flow-matching mode with 1000 train timesteps.
        tc.noise_scheduler = "flowmatch"
        tc.num_train_timesteps = getattr(tc, "num_train_timesteps", 1000) or 1000

        # Use MSE loss on rectified-flow target (noise - latents) as in DiffSynth.
        tc.loss_type = "mse"

        # Let our FlowMatch scheduler control weighting; keep timesteps linear.
        tc.timestep_type = "linear"
        tc.linear_timesteps = False
        tc.linear_timesteps2 = False

        # Disable SNR re-weighting — DiffSynth already applies its own weighting.
        tc.snr_gamma = None
        tc.min_snr_gamma = None

        # Make sure the model is treated as flow-matching by SDTrainer logic.
        if hasattr(self.sd, "is_flow_matching"):
            self.sd.is_flow_matching = True


class ZImageDiffSynthTrainerExtension(Extension):
    uid = "z_image_diffsynth_trainer"
    name = "Z-Image DiffSynth Trainer"

    @classmethod
    def get_process(cls):
        return ZImageDiffSynthTrainer
