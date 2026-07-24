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
    def _tiny_dit(self, param_dtype=torch.bfloat16):
        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, dtype=param_dtype))

        return TinyDiT()

    def test_compute_gate_returns_train_dtype(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit()
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
                model_dtype=torch.bfloat16,
                train_dtype=torch.float32,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)

    def test_matched_dtypes_noop(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit()
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
                model_dtype=torch.bfloat16,
                train_dtype=torch.bfloat16,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.bfloat16)

    def test_model_dtype_overrides_param_dtype(self):
        """Compute gate follows model_dtype, not DiT parameter dtype."""
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit(param_dtype=torch.float32)
        latents = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        embeds = torch.randn(1, 8, 8, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        test_case = self

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            test_case.assertEqual(latents.dtype, torch.bfloat16)
            test_case.assertEqual(prompt_embeds.dtype, torch.bfloat16)
            return torch.zeros_like(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=torch.bfloat16,
                train_dtype=torch.float32,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(next(dit.parameters()).dtype, torch.float32)

    def test_list_prompt_embeds_cast_to_model_dtype(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit()
        latents = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        embeds = [
            torch.randn(8, 8, dtype=torch.float32),
            torch.randn(4, 8, dtype=torch.float32),
        ]
        timestep = torch.tensor([500.0])
        test_case = self

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            test_case.assertEqual(latents.dtype, torch.bfloat16)
            test_case.assertIsInstance(prompt_embeds, list)
            test_case.assertEqual(len(prompt_embeds), 2)
            for emb in prompt_embeds:
                test_case.assertEqual(emb.dtype, torch.bfloat16)
            return torch.zeros_like(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=torch.bfloat16,
                train_dtype=torch.float32,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)

    def test_train_dtype_none_uses_latents_dtype(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit()
        latents = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        embeds = torch.randn(1, 8, 8, dtype=torch.bfloat16)
        timestep = torch.tensor([500.0])
        test_case = self

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            test_case.assertEqual(latents.dtype, torch.bfloat16)
            test_case.assertEqual(prompt_embeds.dtype, torch.bfloat16)
            return torch.zeros_like(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=torch.bfloat16,
                train_dtype=None,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)

    def test_cpu_bf16_model_dtype_disables_autocast(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import forward as forward_mod

        dit = self._tiny_dit()
        latents = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        embeds = torch.randn(1, 8, 8, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        observed = []

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            observed.append(torch.is_autocast_enabled("cuda"))
            return torch.zeros_like(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=torch.bfloat16,
                train_dtype=torch.float32,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(observed, [False])


class TestCudaModelDtypeAutocast(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 hardware required",
    )
    def test_float8_lora_checkpoint_keeps_config_autocast(self):
        from optimum.quanto import freeze
        from torch.utils.checkpoint import checkpoint

        from extensions_built_in.diffusion_models.z_image_diffsynth import (
            forward as forward_mod,
        )
        from toolkit.util.quantize import get_qtype, quantize

        model_dtype = get_torch_dtype(ModelConfig(name_or_path="x", dtype="bf16").dtype)
        train_dtype = get_torch_dtype(TrainConfig(dtype="fp32").dtype)
        network_dtype = get_torch_dtype(
            NetworkConfig(type="lora", linear=8, dtype="fp32").dtype
        )

        device = torch.device("cuda")
        base = nn.Linear(8, 8).to(device=device, dtype=model_dtype)
        quantize(base, weights=get_qtype("float8"))
        freeze(base)

        autocast_observations = []

        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.base = base
                self.lora_down = nn.Linear(8, 4, bias=False).to(
                    device=device,
                    dtype=network_dtype,
                )
                self.lora_up = nn.Linear(4, 8, bias=False).to(
                    device=device,
                    dtype=network_dtype,
                )

            def forward(self, x):
                autocast_observations.append(
                    (
                        torch.is_autocast_enabled("cuda"),
                        torch.get_autocast_dtype("cuda"),
                    )
                )
                base_out = self.base(x)
                lora_input = x.to(self.lora_down.weight.dtype)
                lora_out = self.lora_up(self.lora_down(lora_input))
                return base_out + lora_out

        dit = TinyDiT()
        latents = torch.randn(
            1,
            4,
            2,
            8,
            device=device,
            dtype=train_dtype,
        )
        embeds = torch.randn(1, 8, 8, device=device, dtype=train_dtype)
        timestep = torch.tensor([500.0], device=device)

        def fake_model_fn(
            dit,
            latents=None,
            timestep=None,
            prompt_embeds=None,
            use_gradient_checkpointing=False,
            **kwargs,
        ):
            if use_gradient_checkpointing:
                return checkpoint(dit, latents, use_reentrant=False)
            return dit(latents)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=model_dtype,
                use_gradient_checkpointing=True,
                train_dtype=train_dtype,
            )
            out.square().mean().backward()
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, train_dtype)
        self.assertGreaterEqual(len(autocast_observations), 2)
        self.assertTrue(
            all(enabled for enabled, _ in autocast_observations)
        )
        self.assertTrue(
            all(
                dtype == model_dtype
                for _, dtype in autocast_observations
            )
        )
        for parameter in (dit.lora_down.weight, dit.lora_up.weight):
            self.assertEqual(parameter.dtype, network_dtype)
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.dtype, network_dtype)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp32_model_dtype_disables_autocast(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import (
            forward as forward_mod,
        )

        model_dtype = get_torch_dtype(ModelConfig(name_or_path="x", dtype="fp32").dtype)
        train_dtype = get_torch_dtype(TrainConfig(dtype="fp32").dtype)
        device = torch.device("cuda")
        observed = []

        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, device=device, dtype=model_dtype))

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            observed.append(
                (
                    torch.is_autocast_enabled("cuda"),
                    latents.dtype,
                )
            )
            return torch.zeros_like(latents)

        dit = TinyDiT()
        latents = torch.randn(1, 4, 2, 2, device=device, dtype=train_dtype)
        embeds = torch.randn(1, 8, 8, device=device, dtype=train_dtype)
        timestep = torch.tensor([500.0], device=device)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=model_dtype,
                train_dtype=train_dtype,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, train_dtype)
        self.assertEqual(len(observed), 1)
        enabled, latents_dtype = observed[0]
        self.assertFalse(enabled)
        self.assertEqual(latents_dtype, torch.float32)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 hardware required",
    )
    def test_fp16_model_dtype_uses_fp16_autocast(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth import (
            forward as forward_mod,
        )

        model_dtype = get_torch_dtype(ModelConfig(name_or_path="x", dtype="fp16").dtype)
        train_dtype = get_torch_dtype(TrainConfig(dtype="fp32").dtype)
        device = torch.device("cuda")
        observed = []

        class TinyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, device=device, dtype=model_dtype))

        def fake_model_fn(dit, latents=None, timestep=None, prompt_embeds=None, **kwargs):
            observed.append(
                (
                    torch.is_autocast_enabled("cuda"),
                    torch.get_autocast_dtype("cuda"),
                    latents.dtype,
                )
            )
            return torch.zeros_like(latents)

        dit = TinyDiT()
        latents = torch.randn(1, 4, 2, 2, device=device, dtype=train_dtype)
        embeds = torch.randn(1, 8, 8, device=device, dtype=train_dtype)
        timestep = torch.tensor([500.0], device=device)

        _install_fake_model_fn(fake_model_fn)
        try:
            out = forward_mod.run_forward(
                dit,
                latents,
                timestep,
                embeds,
                model_dtype=model_dtype,
                train_dtype=train_dtype,
            )
        finally:
            _uninstall_fake_model_fn()

        self.assertEqual(out.dtype, train_dtype)
        self.assertEqual(len(observed), 1)
        enabled, autocast_dtype, latents_dtype = observed[0]
        self.assertTrue(enabled)
        self.assertEqual(autocast_dtype, torch.float16)
        self.assertEqual(latents_dtype, torch.float16)


class TestDiffusersGetNoisePredictionDtypeGate(unittest.TestCase):
    """Diffusers _raw_dit path must cast fp32 train activations to model_dtype."""

    def test_diffusers_path_casts_latents_and_embeds_to_model_dtype(self):
        from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
            ZImageDiffSynthModel,
        )
        from toolkit.prompt_utils import PromptEmbeds

        test_case = self

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_channels = 4
                self.w = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

            def forward(self, latent_list, timestep, text_embeds, return_dict=False):
                test_case.assertIsInstance(latent_list, list)
                for x in latent_list:
                    test_case.assertEqual(x.dtype, torch.bfloat16)
                test_case.assertIsInstance(text_embeds, list)
                for emb in text_embeds:
                    test_case.assertEqual(emb.dtype, torch.bfloat16)
                return ([torch.zeros_like(x) for x in latent_list],)

        obj = object.__new__(ZImageDiffSynthModel)
        obj._main_is_diffusers = True
        obj.torch_dtype = torch.bfloat16
        obj.train_torch_dtype = torch.float32
        obj.device_torch = torch.device("cpu")
        obj.gradient_checkpointing = False
        obj._raw_dit = FakeDiT()
        obj.model = nn.Identity()

        latents = torch.randn(2, 4, 2, 2, dtype=torch.float32)
        embeds = PromptEmbeds(torch.randn(2, 8, 8, dtype=torch.float32))
        timestep = torch.tensor([500.0, 400.0])

        out = ZImageDiffSynthModel.get_noise_prediction(
            obj, latents, timestep, embeds
        )
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
