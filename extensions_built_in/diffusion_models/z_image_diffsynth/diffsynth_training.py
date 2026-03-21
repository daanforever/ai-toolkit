# DiffSynth-aligned prompt encoding (T2I, no edit_image) and flow-MSE aggregation for Z-Image adapter.

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn.functional as F

from toolkit.prompt_utils import PromptEmbeds
from toolkit.util.tensorboard_timestep_weights import log_timestep_weights


def encode_prompt_diffsynth_literal_t2i(
    tokenizer,
    text_encoder: torch.nn.Module,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_sequence_length: int = 512,
) -> PromptEmbeds:
    """
    Same chain as DiffSynth ZImageUnit_PromptEmbedder.encode_prompt_omni with no condition
    images: literal <|im_start|>user ... user text ... then tokenizer batch → TE hidden_states[-2]
    → masked embeddings per row → pad/stack like prompt_encoding.encode_prompt.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    wrapped: List[List[str]] = []
    for prompt_item in prompt:
        wrapped.append(
            [
                "<|im_start|>user\n"
                + prompt_item
                + "<|im_end|>\n<|im_start|>assistant\n"
            ]
        )

    flattened: List[str] = []
    lengths: List[int] = []
    for row in wrapped:
        lengths.append(len(row))
        flattened.extend(row)

    text_inputs = tokenizer(
        flattened,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list: List[torch.Tensor] = []
    start_idx = 0
    for n_chunks in lengths:
        parts: List[torch.Tensor] = []
        end_idx = start_idx + n_chunks
        for j in range(start_idx, end_idx):
            parts.append(prompt_embeds[j][prompt_masks[j]])
        start_idx = end_idx
        if len(parts) == 1:
            embeddings_list.append(parts[0])
        else:
            embeddings_list.append(torch.cat(parts, dim=0))

    max_len = max(t.shape[0] for t in embeddings_list)
    dim = embeddings_list[0].shape[1]
    padded: List[torch.Tensor] = []
    for t in embeddings_list:
        if t.shape[0] < max_len:
            pad = torch.zeros((max_len - t.shape[0], dim), dtype=dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        padded.append(t.to(dtype))
    text_embeds = torch.stack(padded, dim=0)
    return PromptEmbeds([text_embeds, None])


def aggregate_flow_matching_mse_diffsynth(
    pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    timestep_weight_b: torch.Tensor,
    mask_multiplier: Union[torch.Tensor, float],
    noise_pred: torch.Tensor,
    *,
    train_turbo: bool,
    log_writer,
    step_num: int,
    is_main_process: bool,
    log_every,
) -> torch.Tensor:
    """
    Matches FlowMatchSFTLoss-style scaling for B=1, mask=1:
    mean over (C,H,W) of squared error, then multiply by scalar training weight.
    Order: per batch element b: loss_b = w_b * mean_spatial((pred_b - target_b)^2 * mask_b).
    """
    sq = F.mse_loss(pred.float(), target.float(), reduction="none")
    mm = mask_multiplier
    if train_turbo:
        mm = mm[:, 3:, :, :]
        mm = F.interpolate(mm, size=(pred.shape[2], pred.shape[3]), mode="nearest")

    if len(noise_pred.shape) == 5:
        mm = mm.unsqueeze(2)
        mm = mm.repeat(1, 1, noise_pred.shape[2], 1, 1)

    sq = sq * mm

    if len(noise_pred.shape) == 5:
        per_sample = sq.mean(dim=(1, 2, 3, 4))
    else:
        per_sample = sq.mean(dim=(1, 2, 3))

    w = timestep_weight_b.to(device=per_sample.device, dtype=per_sample.dtype)
    if w.dim() == 0:
        w = w.unsqueeze(0)
    out = per_sample * w

    if log_writer is not None and is_main_process:
        log_timestep_weights(
            log_writer,
            step_num,
            timesteps,
            w,
            log_every=log_every,
        )

    return out
