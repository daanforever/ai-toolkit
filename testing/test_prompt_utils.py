import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.prompt_utils import PromptEmbeds

def test_prompt_embeds_to_preserves_bool_mask():
    text_embeds = torch.randn(2, 10, 768, dtype=torch.float32)
    pooled_embeds = torch.randn(2, 768, dtype=torch.float32)
    attention_mask = torch.ones(2, 10, dtype=torch.bool)
    attention_mask[0, 5:] = False
    attention_mask[1, 8:] = False

    embeds = PromptEmbeds([text_embeds, pooled_embeds], attention_mask=attention_mask)

    # Move to bfloat16 and cpu
    embeds = embeds.to(device="cpu", dtype=torch.bfloat16)

    assert embeds.text_embeds.dtype == torch.bfloat16
    assert embeds.pooled_embeds.dtype == torch.bfloat16
    # attention_mask must remain boolean
    assert embeds.attention_mask.dtype == torch.bool
    assert embeds.attention_mask.device == torch.device("cpu")

    # Check indexing works
    indexed = [embeds.text_embeds[i][embeds.attention_mask[i]] for i in range(embeds.text_embeds.shape[0])]
    assert indexed[0].shape == (5, 768)
    assert indexed[1].shape == (8, 768)


def test_prompt_embeds_to_positional_dtype():
    text_embeds = torch.randn(2, 10, 768, dtype=torch.float32)
    attention_mask = torch.ones(2, 10, dtype=torch.bool)

    embeds = PromptEmbeds(text_embeds, attention_mask=attention_mask)

    # Move using positional dtype
    embeds = embeds.to(torch.float16)

    assert embeds.text_embeds.dtype == torch.float16
    assert embeds.attention_mask.dtype == torch.bool


def test_indexing_with_float_mask_fallback():
    # Even if attention_mask somehow becomes a float tensor (e.g. from older cached states or external sources),
    # our .bool() casting in model.py and z_image.py should prevent IndexError.
    text_embeds = torch.randn(2, 10, 768, dtype=torch.float32)
    attention_mask_float = torch.ones(2, 10, dtype=torch.float16)
    attention_mask_float[0, 5:] = 0.0
    attention_mask_float[1, 8:] = 0.0

    # This would raise IndexError without .bool()
    with pytest.raises(IndexError):
        # Verify that raw float indexing indeed raises IndexError in PyTorch
        _ = [text_embeds[i][attention_mask_float[i]] for i in range(text_embeds.shape[0])]

    # Verify that casting to bool fixes it
    indexed = [text_embeds[i][attention_mask_float[i].bool()] for i in range(text_embeds.shape[0])]
    assert indexed[0].shape == (5, 768)
    assert indexed[1].shape == (8, 768)
