"""Unit tests for model/network/train dtype split (no GPU model load)."""

import os
import sys
import types
import unittest

import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from toolkit.config_modules import NetworkConfig, ModelConfig, TrainConfig
from toolkit.train_tools import get_torch_dtype


def _install_fake_model_fn(fn):
    fake_pipelines = types.ModuleType("diffsynth.pipelines.z_image")
    fake_pipelines.model_fn_z_image_turbo = fn
    fake_diffsynth = types.ModuleType("diffsynth")
    fake_pipelines_pkg = types.ModuleType("diffsynth.pipelines")
    sys.modules["diffsynth"] = fake_diffsynth
    sys.modules["diffsynth.pipelines"] = fake_pipelines_pkg
    sys.modules["diffsynth.pipelines.z_image"] = fake_pipelines


def _uninstall_fake_model_fn():
    for k in ("diffsynth", "diffsynth.pipelines", "diffsynth.pipelines.z_image"):
        sys.modules.pop(k, None)


class TestNetworkConfigDtype(unittest.TestCase):
    def test_dtype_default_none(self):
        cfg = NetworkConfig(type="lora", linear=8)
        self.assertIsNone(cfg.dtype)

    def test_dtype_explicit(self):
        cfg = NetworkConfig(type="lora", linear=8, dtype="fp32")
        self.assertEqual(cfg.dtype, "fp32")


class TestModelDtypeFallback(unittest.TestCase):
    def test_model_dtype_respected_when_set(self):
        train = TrainConfig(dtype="fp32")
        model_raw = {"name_or_path": "x", "dtype": "bf16"}
        if model_raw.get("dtype") is None:
            model_raw["dtype"] = train.dtype
        mc = ModelConfig(**model_raw)
        self.assertEqual(mc.dtype, "bf16")
        self.assertEqual(get_torch_dtype(mc.dtype), torch.bfloat16)

    def test_model_dtype_falls_back_to_train(self):
        train = TrainConfig(dtype="bf16")
        model_raw = {"name_or_path": "x"}
        if model_raw.get("dtype") is None:
            model_raw["dtype"] = train.dtype
        mc = ModelConfig(**model_raw)
        self.assertEqual(mc.dtype, "bf16")


class TestRunForwardDtypeGate(unittest.TestCase):
    def test_compute_gate_returns_train_dtype(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        dit = TinyDiT()
        latents = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        embeds = torch.randn(1, 8, 8, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        test_case = self

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            test_case.assertEqual(latents.dtype, torch.bfloat16)
            if isinstance(prompt_embeds, torch.Tensor):
                test_case.assertEqual(prompt_embeds.dtype, torch.bfloat16)
            return torch.zeros_like(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                train_dtype=torch.float32,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)

    def test_matched_dtypes_noop(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        dit = TinyDiT()
        latents = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
        embeds = torch.randn(1, 8, 8, dtype=torch.bfloat16)
        timestep = torch.tensor([500.0])
        test_case = self

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            test_case.assertEqual(latents.dtype, torch.bfloat16)
            return latents + 1

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                train_dtype=torch.bfloat16,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
