# Encode prompt in DiffSynth ZImage style; return PromptEmbeds (text_embeds only, pooled_embeds=None).

from typing import List, Union, Optional
import torch

from toolkit.prompt_utils import PromptEmbeds


def encode_prompt(
    tokenizer,
    text_encoder: torch.nn.Module,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
) -> PromptEmbeds:
    """
    Encode prompt using tokenizer + text_encoder (DiffSynth ZImageUnit_PromptEmbedder logic).
    Returns PromptEmbeds(text_embeds=..., pooled_embeds=None) for Z-Image.
    """
    if isinstance(prompt, str):
        prompt = [prompt]
    for i, prompt_item in enumerate(prompt):
        messages = [{"role": "user", "content": prompt_item}]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
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

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    # Pad to same length and stack to [batch, seq_len, dim] for toolkit
    max_len = max(t.shape[0] for t in embeddings_list)
    dim = embeddings_list[0].shape[1]
    if dtype is None:
        dtype = torch.float32
    padded = []
    for t in embeddings_list:
        if t.shape[0] < max_len:
            pad = torch.zeros((max_len - t.shape[0], dim), dtype=dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    text_embeds = torch.stack(padded, dim=0)
    return PromptEmbeds([text_embeds, None])
