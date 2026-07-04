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
from extensions_built_in.diffusion_models.z_image_diffsynth.model import (  # noqa: E402
    ZImageDiffSynthModel,
)
from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (  # noqa: E402
    DEFAULT_ZIMAGE_MODEL_PATH,
)
from toolkit.util.quantize import get_qtype, quantize  # noqa: E402

EMBED_DTYPE = torch.bfloat16
# bf16 parity tolerance (direct encode helpers for DiffSynth parity tests).
EMBED_RTOL = 1e-2
EMBED_ATOL = 1e-2

# ZImageDiffSynthModel.get_prompt_embeds passes dtype=torch.float32 to encode helpers.
PRODUCTION_ENCODE_DTYPE = torch.float32
PRODUCTION_RTOL = 1e-5
PRODUCTION_ATOL = 1e-5


def _resolve_model_path() -> str:
    return os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH


def _pad_and_stack(embeddings_list: list[torch.Tensor], dtype: torch.dtype = EMBED_DTYPE) -> torch.Tensor:
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


def _make_pipe(tokenizer, text_encoder):
    return SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)


def _make_prompt_embeds_model(bundle: dict, model_kwargs: dict | None = None) -> ZImageDiffSynthModel:
    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = bundle["device"]
    model.tokenizer = [bundle["tokenizer"]]
    model.text_encoder = [bundle["text_encoder"]]
    model.model_config = SimpleNamespace(model_kwargs=model_kwargs or {})
    return model


def _as_prompt_list(prompt: str | list[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    return list(prompt)


def _encode_chat(tokenizer, text_encoder, device, prompt):
    return prompt_encoding_mod.encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=_as_prompt_list(prompt),
        device=device,
        dtype=EMBED_DTYPE,
    ).text_embeds


def _encode_literal(tokenizer, text_encoder, device, prompt):
    return diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=_as_prompt_list(prompt),
        device=device,
        dtype=EMBED_DTYPE,
    ).text_embeds


def _encode_like_get_prompt_embeds_chat(tokenizer, text_encoder, device, prompt):
    """Mirror ZImageDiffSynthModel.get_prompt_embeds chat branch (dtype=torch.float32)."""
    return prompt_encoding_mod.encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        device=device,
        dtype=PRODUCTION_ENCODE_DTYPE,
    ).text_embeds


def _encode_like_get_prompt_embeds_literal(tokenizer, text_encoder, device, prompt):
    """Mirror ZImageDiffSynthModel.get_prompt_embeds literal branch (dtype=torch.float32)."""
    return diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        device=device,
        dtype=PRODUCTION_ENCODE_DTYPE,
    ).text_embeds


def _allclose_production(a: torch.Tensor, b: torch.Tensor) -> None:
    """Compare tensors using tolerance suited to their dtype (production paths)."""
    if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16:
        assert torch.allclose(a, b, rtol=EMBED_RTOL, atol=EMBED_ATOL)
    else:
        assert torch.allclose(a, b, rtol=PRODUCTION_RTOL, atol=PRODUCTION_ATOL)


def _chat_formatted_string(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _literal_wrapped_string(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _load_quantized_text_encoder(model_path: str, device: torch.device) -> Qwen3ForCausalLM:
    """Mirror loader.load_components text-encoder path (quantize_te=True)."""
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
    )
    text_encoder.to(device)
    quantize(text_encoder, weights=get_qtype("qfloat8"))
    freeze(text_encoder)
    text_encoder.eval()
    return text_encoder


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
            "model_path": model_path,
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
        workflow = _encode_chat(tokenizer, text_encoder, device, list(prompts))

        pipe = _make_pipe(tokenizer, text_encoder)
        ds_embeddings = prompt_unit.encode_prompt(
            pipe,
            prompt=list(prompts),
            device=device,
        )
        ds_stacked = _pad_and_stack(ds_embeddings, dtype=EMBED_DTYPE)

    assert workflow.shape == ds_stacked.shape
    assert workflow.dtype == EMBED_DTYPE
    assert torch.allclose(workflow, ds_stacked, rtol=EMBED_RTOL, atol=EMBED_ATOL)


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
        workflow = _encode_literal(tokenizer, text_encoder, device, list(prompts))

        pipe = _make_pipe(tokenizer, text_encoder)
        ds_omni = prompt_unit.encode_prompt_omni(
            pipe,
            prompt=list(prompts),
            edit_image=None,
            device=device,
        )
        ds_stacked = _pad_and_stack(_flatten_omni_embeddings(ds_omni), dtype=EMBED_DTYPE)

    assert workflow.shape == ds_stacked.shape
    assert workflow.dtype == EMBED_DTYPE
    assert torch.allclose(workflow, ds_stacked, rtol=EMBED_RTOL, atol=EMBED_ATOL)


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
        workflow = _encode_chat(tokenizer, text_encoder, device, list(prompts))

        pipe = _make_pipe(tokenizer, text_encoder)
        ds_embeddings = prompt_unit.encode_prompt(
            pipe,
            prompt=list(prompts),
            device=device,
        )
        ds_stacked = _pad_and_stack(ds_embeddings, dtype=EMBED_DTYPE)

    workflow_delta = (workflow[0] - workflow[1]).abs().max().item()
    ds_delta = (ds_stacked[0] - ds_stacked[1]).abs().max().item()

    assert workflow_delta > 1.0
    assert ds_delta > 1.0


def test_literal_and_chat_paths_are_equivalent_for_zimage(text_encoder_bundle):
    """Z-Image chat template matches literal <|im_start|>user… chain for this checkpoint."""
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt = "a red cube on white table"

    chat_str = _chat_formatted_string(tokenizer, prompt)
    lit_str = _literal_wrapped_string(prompt)
    assert chat_str == lit_str

    with torch.inference_mode():
        chat = _encode_chat(tokenizer, text_encoder, device, prompt)
        literal = _encode_literal(tokenizer, text_encoder, device, prompt)

    assert chat.dtype == EMBED_DTYPE
    assert literal.dtype == EMBED_DTYPE
    assert torch.allclose(chat, literal, rtol=EMBED_RTOL, atol=EMBED_ATOL)


def test_empty_prompt_chat_parity_with_diffsynth(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt_unit = text_encoder_bundle["prompt_unit"]

    with torch.inference_mode():
        workflow = _encode_chat(tokenizer, text_encoder, device, "")
        pipe = _make_pipe(tokenizer, text_encoder)
        ds_stacked = _pad_and_stack(
            prompt_unit.encode_prompt(pipe, prompt="", device=device),
            dtype=EMBED_DTYPE,
        )

    assert workflow.shape == ds_stacked.shape
    assert workflow.shape[0] == 1
    assert workflow.shape[1] <= 512
    assert workflow.dtype == EMBED_DTYPE
    assert torch.isfinite(workflow).all()
    assert torch.allclose(workflow, ds_stacked, rtol=EMBED_RTOL, atol=EMBED_ATOL)


def test_empty_prompt_literal_parity_with_diffsynth(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt_unit = text_encoder_bundle["prompt_unit"]

    with torch.inference_mode():
        workflow = _encode_literal(tokenizer, text_encoder, device, "")
        pipe = _make_pipe(tokenizer, text_encoder)
        ds_omni = prompt_unit.encode_prompt_omni(
            pipe, prompt="", edit_image=None, device=device
        )
        ds_stacked = _pad_and_stack(_flatten_omni_embeddings(ds_omni), dtype=EMBED_DTYPE)

    assert workflow.shape == ds_stacked.shape
    assert workflow.shape[0] == 1
    assert workflow.shape[1] <= 512
    assert workflow.dtype == EMBED_DTYPE
    assert torch.isfinite(workflow).all()
    assert torch.allclose(workflow, ds_stacked, rtol=EMBED_RTOL, atol=EMBED_ATOL)


@pytest.mark.parametrize("encode_fn_name", ["chat", "literal"])
def test_dup_batch_rows_match_each_other(text_encoder_bundle, encode_fn_name):
    """Rows with the same caption in one batch must match (batch padding is symmetric)."""
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    encode_fn = _encode_chat if encode_fn_name == "chat" else _encode_literal
    prompt = "a tiny orange fox in the snow"

    with torch.inference_mode():
        dup_batch = encode_fn(tokenizer, text_encoder, device, [prompt, prompt])

    assert dup_batch.shape[0] == 2
    assert torch.allclose(dup_batch[0], dup_batch[1], rtol=EMBED_RTOL, atol=EMBED_ATOL)


@pytest.mark.parametrize("encode_fn_name", ["chat", "literal"])
def test_encode_prompt_is_deterministic(text_encoder_bundle, encode_fn_name):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    encode_fn = _encode_chat if encode_fn_name == "chat" else _encode_literal
    prompt = "portrait photo, soft light"

    with torch.inference_mode():
        first = encode_fn(tokenizer, text_encoder, device, prompt)
        second = encode_fn(tokenizer, text_encoder, device, prompt)

    assert torch.equal(first, second)


def test_get_prompt_embeds_via_model_literal_flag(text_encoder_bundle):
    model = _make_prompt_embeds_model(
        text_encoder_bundle,
        {"use_diffsynth_prompt_encoding": True, "use_diffsynth_training_loop": False},
    )
    prompt = "mountain landscape at dawn"
    tok = text_encoder_bundle["tokenizer"]
    te = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]

    with torch.inference_mode():
        via_model = model.get_prompt_embeds(prompt).text_embeds
        direct = _encode_like_get_prompt_embeds_literal(tok, te, device, prompt)

    assert via_model.dtype == torch.float32
    assert direct.dtype == torch.float32
    _allclose_production(via_model, direct)


def test_get_prompt_embeds_via_model_chat_flag(text_encoder_bundle):
    model = _make_prompt_embeds_model(
        text_encoder_bundle,
        {"use_diffsynth_prompt_encoding": False, "use_diffsynth_training_loop": True},
    )
    prompt = "mountain landscape at dawn"
    tok = text_encoder_bundle["tokenizer"]
    te = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]

    with torch.inference_mode():
        via_model = model.get_prompt_embeds(prompt).text_embeds
        direct = _encode_like_get_prompt_embeds_chat(tok, te, device, prompt)

    # encode_prompt requests float32 but keeps TE bf16 when no intra-batch pad (production today).
    assert via_model.dtype == direct.dtype
    _allclose_production(via_model, direct)


def test_get_prompt_embeds_production_output_dtypes(text_encoder_bundle):
    """Document actual output dtypes from get_prompt_embeds (model requests float32)."""
    prompt = "a cat on a mat"
    tok = text_encoder_bundle["tokenizer"]
    te = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]

    model_literal = _make_prompt_embeds_model(
        text_encoder_bundle,
        {"use_diffsynth_prompt_encoding": True},
    )
    model_chat = _make_prompt_embeds_model(
        text_encoder_bundle,
        {"use_diffsynth_prompt_encoding": False},
    )

    with torch.inference_mode():
        literal_embeds = model_literal.get_prompt_embeds(prompt).text_embeds
        chat_embeds = model_chat.get_prompt_embeds(prompt).text_embeds
        literal_direct = _encode_like_get_prompt_embeds_literal(tok, te, device, prompt)
        chat_direct = _encode_like_get_prompt_embeds_chat(tok, te, device, prompt)

    assert literal_embeds.dtype == torch.float32
    assert literal_direct.dtype == torch.float32
    assert chat_embeds.dtype == torch.float32
    assert chat_direct.dtype == torch.float32


def test_literal_caption_changes_embeddings(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]

    prompts = [
        "a red cube on white table",
        "a green sphere on white table",
    ]

    with torch.inference_mode():
        embeds = _encode_literal(tokenizer, text_encoder, device, prompts)

    assert (embeds[0] - embeds[1]).abs().max().item() > 1.0


def test_long_prompt_truncation_finite(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    long_prompt = "word " * 2000

    with torch.inference_mode():
        chat = _encode_chat(tokenizer, text_encoder, device, long_prompt)
        literal = _encode_literal(tokenizer, text_encoder, device, long_prompt)

    for embeds in (chat, literal):
        assert embeds.shape[1] <= 512
        assert torch.isfinite(embeds).all()


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "a",
        "日本語テスト",
        "emoji 🦊 portrait",
        "line one\nline two",
    ],
)
def test_encode_edge_prompts_finite_shapes(text_encoder_bundle, prompt):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]

    with torch.inference_mode():
        chat = _encode_chat(tokenizer, text_encoder, device, prompt)
        literal = _encode_literal(tokenizer, text_encoder, device, prompt)

    for embeds in (chat, literal):
        assert embeds.ndim == 3
        assert embeds.shape[0] == 1
        assert embeds.shape[1] <= 512
        assert embeds.dtype == EMBED_DTYPE
        assert torch.isfinite(embeds).all()


def test_chat_padded_tail_is_zero(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompts = ["hi", "a much longer caption with extra words to widen the batch pad"]

    def _valid_len_for_chat(prompt_item: str) -> int:
        messages = [{"role": "user", "content": prompt_item}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text_inputs = tokenizer(
            [formatted],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return int(text_inputs.attention_mask[0].sum().item())

    valid_lens = [_valid_len_for_chat(p) for p in prompts]

    with torch.inference_mode():
        embeds = _encode_chat(tokenizer, text_encoder, device, prompts)

    assert valid_lens[0] < valid_lens[1]
    assert embeds.shape[1] == valid_lens[1]
    assert torch.all(embeds[0, valid_lens[0] :, :] == 0)
    assert torch.any(embeds[0, : valid_lens[0], :].abs() > 0)


def test_loader_tokenizer_matches_fixture(text_encoder_bundle):
    model_path = text_encoder_bundle["model_path"]
    tokenizer_fixture = text_encoder_bundle["tokenizer"]
    tokenizer_reload = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    prompt = "portrait photo of a fox"

    ids_fixture = tokenizer_fixture(prompt, return_tensors="pt").input_ids
    ids_reload = tokenizer_reload(prompt, return_tensors="pt").input_ids
    assert torch.equal(ids_fixture, ids_reload)


def test_loader_quantized_te_matches_fixture(text_encoder_bundle):
    model_path = text_encoder_bundle["model_path"]
    device = text_encoder_bundle["device"]
    tokenizer = text_encoder_bundle["tokenizer"]
    prompt = "street photo during rain"

    with torch.inference_mode():
        fixture_embeds = _encode_chat(
            tokenizer, text_encoder_bundle["text_encoder"], device, prompt
        )
        te_reload = _load_quantized_text_encoder(model_path, device)
        reload_embeds = _encode_chat(tokenizer, te_reload, device, prompt)
        del te_reload
        torch.cuda.empty_cache()

    assert torch.allclose(fixture_embeds, reload_embeds, rtol=EMBED_RTOL, atol=EMBED_ATOL)


def test_literal_padded_tail_is_zero(text_encoder_bundle):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompts = ["hi", "a much longer caption with extra words to widen the batch pad"]

    def _valid_len_for_literal(prompt_item: str) -> int:
        formatted = (
            "<|im_start|>user\n"
            + prompt_item
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        text_inputs = tokenizer(
            [formatted],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return int(text_inputs.attention_mask[0].sum().item())

    valid_lens = [_valid_len_for_literal(p) for p in prompts]

    with torch.inference_mode():
        embeds = _encode_literal(tokenizer, text_encoder, device, prompts)

    assert valid_lens[0] < valid_lens[1]
    assert embeds.shape[1] == valid_lens[1]
    assert torch.all(embeds[0, valid_lens[0] :, :] == 0)
    assert torch.any(embeds[0, : valid_lens[0], :].abs() > 0)


def test_get_prompt_embeds_with_list_input(text_encoder_bundle):
    model = _make_prompt_embeds_model(
        text_encoder_bundle,
        {"use_diffsynth_prompt_encoding": True},
    )
    prompts = ["a cat", "a very fluffy adorable brown dog playing in the garden"]

    with torch.inference_mode():
        via_model_batch = model.get_prompt_embeds(prompts).text_embeds
        single_1 = model.get_prompt_embeds(prompts[0]).text_embeds
        single_2 = model.get_prompt_embeds(prompts[1]).text_embeds

    # The batch should have shape (2, max_len, dim) where max_len is the length of prompt 2.
    assert via_model_batch.shape[0] == 2
    assert via_model_batch.shape[1] == single_2.shape[1]

    # First item in batch should match single_1 padded with zeros to match single_2 sequence length.
    expected_1 = single_1[0]
    pad_len = single_2.shape[1] - single_1.shape[1]
    if pad_len > 0:
        pad = torch.zeros((pad_len, single_1.shape[2]), dtype=single_1.dtype, device=single_1.device)
        expected_1 = torch.cat([expected_1, pad], dim=0)

    # Since the text encoder computes in bfloat16/qfloat8 and uses transformers SDPA under different batch sizes,
    # batch size 1 vs batch size 2 has slight numerical divergence. We use cosine similarity to verify parity.
    len_1 = single_1.shape[1]
    cos_1 = torch.nn.functional.cosine_similarity(via_model_batch[0, :len_1], single_1[0], dim=-1)
    assert cos_1.min().item() > 0.99

    cos_2 = torch.nn.functional.cosine_similarity(via_model_batch[1], single_2[0], dim=-1)
    assert cos_2.min().item() > 0.99


@pytest.mark.parametrize("max_len", [32, 64])
@pytest.mark.parametrize("encode_fn_name", ["chat", "literal"])
def test_custom_max_sequence_length(text_encoder_bundle, max_len, encode_fn_name):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt = "word " * 100

    encode_fn = (
        prompt_encoding_mod.encode_prompt
        if encode_fn_name == "chat"
        else diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i
    )

    with torch.inference_mode():
        embeds = encode_fn(
            tokenizer,
            text_encoder,
            prompt,
            device,
            dtype=EMBED_DTYPE,
            max_sequence_length=max_len,
        ).text_embeds

    assert embeds.shape[1] == max_len


@pytest.mark.parametrize("encode_fn_name", ["chat", "literal"])
def test_string_vs_list_of_string_parity(text_encoder_bundle, encode_fn_name):
    tokenizer = text_encoder_bundle["tokenizer"]
    text_encoder = text_encoder_bundle["text_encoder"]
    device = text_encoder_bundle["device"]
    prompt = "a majestic eagle flying high over snowy mountain peaks"

    encode_fn = (
        prompt_encoding_mod.encode_prompt
        if encode_fn_name == "chat"
        else diffsynth_training_mod.encode_prompt_diffsynth_literal_t2i
    )

    with torch.inference_mode():
        embeds_str = encode_fn(tokenizer, text_encoder, prompt, device, dtype=EMBED_DTYPE).text_embeds
        embeds_list = encode_fn(tokenizer, text_encoder, [prompt], device, dtype=EMBED_DTYPE).text_embeds

    assert torch.equal(embeds_str, embeds_list)

