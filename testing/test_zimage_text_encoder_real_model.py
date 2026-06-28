"""
Real-model checks for Z-Image text encoder parity between toolkit workflow and DiffSynth-Studio.

This test intentionally does NOT load transformer/sampling_transformer:
only tokenizer + text_encoder are used.

Run (repo root, venv):
  venv\\Scripts\\python.exe -m pytest testing/test_zimage_text_encoder_real_model.py -v -s

Env:
  ZIMAGE_DIFFSYNTH_MODEL_PATH (optional, falls back to DEFAULT_ZIMAGE_MODEL_PATH from test_smoke)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from optimum.quanto import freeze
from transformers import AutoTokenizer, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.z_image_diffsynth import (  # noqa: E402
    diffsynth_training as diffsynth_training_mod,
)
from extensions_built_in.diffusion_models.z_image_diffsynth import loader as loader_mod  # noqa: E402
from extensions_built_in.diffusion_models.z_image_diffsynth import prompt_encoding as prompt_encoding_mod  # noqa: E402
from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from toolkit.util.quantize import get_qtype, quantize  # noqa: E402


def _resolve_model_path() -> str:
    return os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH


def _pad_and_stack(embeddings_list: list[torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    max_len = max(t.shape[0] for t in embeddings_list)
    dim = embeddings_list[0].shape[1]
    padded: list[torch.Tensor] = []
    for tensor in embeddings_list:
        out = tensor.to(dtype)
        if out.shape[0] < max_len:
            pad = torch.zeros((max_len - out.shape[0], dim), dtype=dtype, device=out.device)
            out = torch.cat([out, pad], dim=0)
        padded.append(out)
    return torch.stack(padded, dim=0)


def _flatten_omni_embeddings(omni_embeddings: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for row in omni_embeddings:
        if len(row) == 1:
            rows.append(row[0])
        else:
            rows.append(torch.cat(row, dim=0))
    return rows


@pytest.fixture(scope="module")
def text_encoder_bundle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this real-model test")

    model_path = _resolve_model_path()
    if not model_path or not os.path.isdir(model_path):
        pytest.skip(
            "Z-Image model path is missing. Set ZIMAGE_DIFFSYNTH_MODEL_PATH or update DEFAULT_ZIMAGE_MODEL_PATH."
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path,
        subfolder="text_encoder",
        dtype=dtype,
    )
    text_encoder.to(device)

    # Mirror real z_image_diffsynth configs where text encoder is quantized.
    quantize(text_encoder, weights=get_qtype("qfloat8"))
    freeze(text_encoder)
    text_encoder.eval()

    loader_mod._ensure_diffsynth_path()
    from diffsynth.pipelines.z_image import ZImageUnit_PromptEmbedder

    prompt_unit = ZImageUnit_PromptEmbedder()

    try:
        yield {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "device": device,
            "prompt_unit": prompt_unit,
        }
    finally:
        del text_encoder
        torch.cuda.empty_cache()


def test_workflow_chat_template_matches_diffsynth_encode_prompt(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt_unit = text_encoder_bundle["prompt_unit"]

    prompts = [
        "portrait photo of a tiny orange fox, ultra-detailed fur",
        "street photo of a blue tram during rain, cinematic lighting",
    ]

    with torch.inference_mode():
        workflow = prompt_encoding_mod.encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=list(prompts),
            device=device,
            dtype=torch.float32,
        ).text_embeds

        pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
        ds_embeddings = prompt_unit.encode_prompt(
            pipe,
            prompt=list(prompts),
            device=device,
        )
        ds_stacked = _pad_and_stack(ds_embeddings, dtype=torch.float32)

    assert workflow.shape == ds_stacked.shape
    assert torch.allclose(workflow, ds_stacked, rtol=1e-5, atol=1e-5)


def test_workflow_literal_matches_diffsynth_encode_prompt_omni(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt_unit = text_encoder_bundle["prompt_unit"]

    prompts = [
        "anime character with silver hair and red scarf",
        "mountain landscape at dawn with thin fog",
    ]

    with torch.inference_mode():
        workflow = diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=list(prompts),
            device=device,
            dtype=torch.float32,
        ).text_embeds

        pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
        ds_omni = prompt_unit.encode_prompt_omni(
            pipe,
            prompt=list(prompts),
            edit_image=None,
            device=device,
        )
        ds_stacked = _pad_and_stack(_flatten_omni_embeddings(ds_omni), dtype=torch.float32)

    assert workflow.shape == ds_stacked.shape
    assert torch.allclose(workflow, ds_stacked, rtol=1e-5, atol=1e-5)


def test_caption_changes_embeddings_in_workflow_and_diffsynth_paths(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt_unit = text_encoder_bundle["prompt_unit"]

    prompts = [
        "a red cube on white table",
        "a green sphere on white table",
    ]

    with torch.inference_mode():
        workflow = prompt_encoding_mod.encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=list(prompts),
            device=device,
            dtype=torch.float32,
        ).text_embeds

        pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
        ds_embeddings = prompt_unit.encode_prompt(
            pipe,
            prompt=list(prompts),
            device=device,
        )
        ds_stacked = _pad_and_stack(ds_embeddings, dtype=torch.float32)

    workflow_delta = (workflow[0] - workflow[1]).abs().max().item()
    ds_delta = (ds_stacked[0] - ds_stacked[1]).abs().max().item()

    assert workflow_delta > 1e-6
    assert ds_delta > 1e-6
