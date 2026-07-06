"""
Smoke test for Z-Image DiffSynth with PEFT DoRA.
This file wraps the original test_train.py to execute the training, saving, and loading
phases using the PEFT library's DoRA implementation (network type: "peft_dora")
instead of the in-house DoRAModule.
"""

import os
import sys

# Set the environment variable to configure the network type to PEFT DoRA
os.environ["ZIMAGE_TEST_TRAIN_NETWORK_TYPE"] = "peft_dora"

# Ensure repo root is on sys.path so we can import the original test module
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extensions_built_in.diffusion_models.z_image_diffsynth.test_train import main

if __name__ == "__main__":
    main()
