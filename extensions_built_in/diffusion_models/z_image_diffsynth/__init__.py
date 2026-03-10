# Z-Image DiffSynth: model and trainer extension (arch zimage_diffsynth, type z_image_diffsynth_trainer)

from .model import ZImageDiffSynthModel
from .trainer import ZImageDiffSynthTrainerExtension

AI_TOOLKIT_MODELS = [ZImageDiffSynthModel]
AI_TOOLKIT_EXTENSIONS = [ZImageDiffSynthTrainerExtension]

__all__ = [
    "ZImageDiffSynthModel",
    "ZImageDiffSynthTrainerExtension",
    "AI_TOOLKIT_MODELS",
    "AI_TOOLKIT_EXTENSIONS",
]
