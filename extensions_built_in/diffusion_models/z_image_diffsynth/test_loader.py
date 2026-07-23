"""
Unit tests for model_kwargs.loader mode selection (main + sampling).

Run from repo root with venv:
  venv\\Scripts\\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_loader.py -q
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from extensions_built_in.diffusion_models.z_image_diffsynth import loader as loader_mod


def test_normalize_loader_mode_known_and_unknown():
    logs = []
    assert loader_mod._normalize_loader_mode("diffusers", logs.append) == "diffusers"
    assert loader_mod._normalize_loader_mode("DiffSynth", logs.append) == "diffsynth"
    assert loader_mod._normalize_loader_mode(None, logs.append) == "auto"
    assert loader_mod._normalize_loader_mode("nope", logs.append) == "auto"
    assert any("falling back" in m for m in logs)


def test_load_transformer_by_mode_diffsynth_only():
    logs = []
    fake_dit = MagicMock(name="dit")
    with (
        patch.object(loader_mod, "load_dit_from_folder", return_value=fake_dit) as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards"
        ) as mock_df,
    ):
        dit, is_df = loader_mod._load_transformer_by_mode(
            "/fake/transformer",
            torch.float32,
            torch.device("cpu"),
            "diffsynth",
            logs.append,
            "transformer",
        )
    assert dit is fake_dit
    assert is_df is False
    mock_ds.assert_called_once()
    mock_df.assert_not_called()


def test_load_transformer_by_mode_diffusers_only():
    logs = []
    fake_tr = MagicMock(name="diffusers_tr")
    with (
        patch.object(loader_mod, "load_dit_from_folder") as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards",
            return_value=fake_tr,
        ) as mock_df,
    ):
        dit, is_df = loader_mod._load_transformer_by_mode(
            "/fake/transformer",
            torch.float32,
            torch.device("cpu"),
            "diffusers",
            logs.append,
            "transformer",
        )
    assert dit is fake_tr
    assert is_df is True
    mock_df.assert_called_once()
    mock_ds.assert_not_called()


def test_load_transformer_by_mode_diffusers_failure_raises():
    logs = []
    with (
        patch.object(loader_mod, "load_dit_from_folder") as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards",
            side_effect=FileNotFoundError("missing"),
        ),
    ):
        with pytest.raises(RuntimeError, match="diffusers"):
            loader_mod._load_transformer_by_mode(
                "/fake/transformer",
                torch.float32,
                torch.device("cpu"),
                "diffusers",
                logs.append,
                "transformer",
            )
    mock_ds.assert_not_called()


def test_load_transformer_by_mode_auto_falls_back_to_diffsynth():
    logs = []
    fake_dit = MagicMock(name="dit")
    with (
        patch.object(loader_mod, "load_dit_from_folder", return_value=fake_dit) as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards",
            side_effect=ValueError("not diffusers"),
        ) as mock_df,
    ):
        dit, is_df = loader_mod._load_transformer_by_mode(
            "/fake/transformer",
            torch.float32,
            torch.device("cpu"),
            "auto",
            logs.append,
            "transformer",
        )
    assert dit is fake_dit
    assert is_df is False
    mock_df.assert_called_once()
    mock_ds.assert_called_once()


def _stub_te_vae_side_effects():
    """Common patches for TE/VAE/tokenizer so load_components can finish."""
    return {
        "ensure": patch.object(loader_mod, "_ensure_diffsynth_path"),
        "tok": patch.object(
            loader_mod.AutoTokenizer,
            "from_pretrained",
            return_value=MagicMock(name="tok"),
        ),
        "te": patch.object(
            loader_mod.Qwen3ForCausalLM,
            "from_pretrained",
            return_value=MagicMock(name="te"),
        ),
        "vae": patch.object(
            loader_mod.AutoencoderKL,
            "from_pretrained",
            return_value=MagicMock(name="vae"),
        ),
        "wrap": patch.object(
            loader_mod,
            "DiffSynthVAEWrapper",
            return_value=MagicMock(name="vae_wrapper"),
        ),
        "norm": patch(
            "extensions_built_in.diffusion_models.z_image_diffsynth.loader.normalize_path",
            side_effect=lambda p: p,
        ),
        "isdir": patch("os.path.isdir", return_value=True),
    }


def test_load_components_diffsynth_loads_main_and_sampling_via_dit():
    fake_main = MagicMock(name="main")
    fake_samp = MagicMock(name="samp")
    call_folders = []

    def _dit_side_effect(folder, dtype, device):
        call_folders.append(folder)
        return fake_samp if len(call_folders) == 1 else fake_main

    patches = _stub_te_vae_side_effects()
    with (
        patches["ensure"],
        patches["tok"],
        patches["te"],
        patches["vae"],
        patches["wrap"],
        patches["norm"],
        patches["isdir"],
        patch.object(loader_mod, "load_dit_from_folder", side_effect=_dit_side_effect) as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards"
        ) as mock_df,
    ):
        out = loader_mod.load_components(
            "/model",
            "/base",
            dtype=torch.float32,
            device=torch.device("cpu"),
            sampling_transformer_path="/sampling",
            loader_mode="diffsynth",
        )

    assert out["dit"] is fake_main
    assert out["sampling_dit"] is fake_samp
    assert out["dit_is_diffusers"] is False
    assert out["sampling_is_diffusers"] is False
    assert mock_ds.call_count == 2
    mock_df.assert_not_called()


def test_load_components_diffusers_loads_main_and_sampling_via_shards():
    fake_main = MagicMock(name="main_df")
    fake_samp = MagicMock(name="samp_df")
    n = {"i": 0}

    def _df_side_effect(*args, **kwargs):
        n["i"] += 1
        return fake_samp if n["i"] == 1 else fake_main

    patches = _stub_te_vae_side_effects()
    with (
        patches["ensure"],
        patches["tok"],
        patches["te"],
        patches["vae"],
        patches["wrap"],
        patches["norm"],
        patches["isdir"],
        patch.object(loader_mod, "load_dit_from_folder") as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards",
            side_effect=_df_side_effect,
        ) as mock_df,
    ):
        out = loader_mod.load_components(
            "/model",
            "/base",
            dtype=torch.float32,
            device=torch.device("cpu"),
            sampling_transformer_path="/sampling",
            loader_mode="diffusers",
        )

    assert out["dit"] is fake_main
    assert out["sampling_dit"] is fake_samp
    assert out["dit_is_diffusers"] is True
    assert out["sampling_is_diffusers"] is True
    assert mock_df.call_count == 2
    mock_ds.assert_not_called()


def test_load_components_auto_fallback_main_when_diffusers_fails():
    fake_main = MagicMock(name="main_ds")

    patches = _stub_te_vae_side_effects()
    with (
        patches["ensure"],
        patches["tok"],
        patches["te"],
        patches["vae"],
        patches["wrap"],
        patches["norm"],
        patches["isdir"],
        patch.object(loader_mod, "load_dit_from_folder", return_value=fake_main) as mock_ds,
        patch(
            "extensions_built_in.diffusion_models.z_image.loading.load_zimage_transformer_from_shards",
            side_effect=OSError("no shards"),
        ) as mock_df,
    ):
        out = loader_mod.load_components(
            "/model",
            "/base",
            dtype=torch.float32,
            device=torch.device("cpu"),
            loader_mode="auto",
        )

    assert out["dit"] is fake_main
    assert out["dit_is_diffusers"] is False
    assert out["sampling_dit"] is None
    mock_df.assert_called_once()
    mock_ds.assert_called_once()


def test_model_load_model_passes_loader_kwarg():
    """ZImageDiffSynthModel.load_model must read model_kwargs['loader'], not sampling_loader."""
    import inspect
    from extensions_built_in.diffusion_models.z_image_diffsynth import model as model_mod

    src = inspect.getsource(model_mod.ZImageDiffSynthModel.load_model)
    assert 'model_kwargs.get("loader"' in src
    assert "sampling_loader" not in src
    assert "loader_mode=" in src
