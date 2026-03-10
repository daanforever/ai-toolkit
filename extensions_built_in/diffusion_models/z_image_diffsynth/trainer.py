# ZImageDiffSynthTrainer: DiffusionTrainer for arch zimage_diffsynth (DiffSynth DiT/forward).

from toolkit.extension import Extension
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer


class ZImageDiffSynthTrainer(DiffusionTrainer):
    """Trainer for Z-Image DiffSynth (arch zimage_diffsynth). Uses same training loop as DiffusionTrainer."""

    pass


class ZImageDiffSynthTrainerExtension(Extension):
    uid = "z_image_diffsynth_trainer"
    name = "Z-Image DiffSynth Trainer"

    @classmethod
    def get_process(cls):
        return ZImageDiffSynthTrainer
