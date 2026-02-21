"""
Compare two ways of getting shuffled caption embeddings for ZImage:
  A) encode_prompt(caption) then shuffle_sequence (segment shuffle in token space)
  B) encode_prompt(caption_shuffled) (shuffle caption string then encode)

Usage (from repo root):
  python scripts/compare_shuffle_sequence_vs_encode_shuffled.py
  python scripts/compare_shuffle_sequence_vs_encode_shuffled.py --model_path /path/to/zimage
"""
import argparse
import os
import random
import sys
from typing import List, Union

import torch

# Add project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from toolkit.prompt_utils import PromptEmbeds
from toolkit.config_modules import ModelConfig

DEFAULT_MODEL_PATH = (
    "e:/Backup/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/"
)


def _load_zimage(model_path: str, device: str = "cuda"):
    from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel

    config = ModelConfig(
        name_or_path=model_path,
        extras_name_or_path=model_path,
    )
    sd = ZImageModel(device=device, model_config=config)
    sd.load_model()
    return sd


def get_segment_boundaries_from_prompt(
    tokenizer, prompt: Union[str, List[str]], seq_len: int
) -> List[int]:
    """
    Compute segment boundaries (phrase boundaries at commas) in token space.
    Returns list of segment end indices: segment i = [boundaries[i] : boundaries[i+1]].
    """
    caption = prompt[0] if isinstance(prompt, list) else prompt
    caption = str(caption)
    enc = tokenizer(caption, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    else:
        input_ids = list(input_ids)
    n_tokens = len(input_ids)

    comma_ids = set()
    for s in [",", ", ", " ,"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        comma_ids.update(ids if isinstance(ids, list) else ids.tolist())

    positions = [i for i, tid in enumerate(input_ids) if tid in comma_ids]
    end = min(seq_len, n_tokens)
    after_comma = sorted(set(p + 1 for p in positions))
    boundaries = [0] + [b for b in after_comma if 0 < b < end] + [end]
    if boundaries[-1] != end:
        boundaries.append(end)
    return boundaries


def shuffle_sequence(
    pe: PromptEmbeds,
    segment_order_override: List[int] = None,
    debug: bool = False,
) -> None:
    """
    Shuffle by whole segments (phrase boundaries). First segment fixed, rest randomly reordered.
    In-place. Optional segment_order_override for reproducible comparison (e.g. [1, 0] for two segments).
    """
    boundaries = getattr(pe, "segment_boundaries", None)
    if boundaries is None or len(boundaries) < 2:
        if debug:
            print("  [shuffle_sequence] No segment boundaries, skip.")
        return
    if len(boundaries) == 2:
        if debug:
            print("  [shuffle_sequence] 1 segment, no shuffle.")
        return

    n_segments = len(boundaries) - 1
    if segment_order_override is not None:
        segment_order = segment_order_override
    else:
        segment_order = [0] + list(range(1, n_segments))
        random.shuffle(segment_order[1:])

    def apply_segment_perm(t: torch.Tensor) -> torch.Tensor:
        if t.dim() < 2 or t.shape[1] <= 1:
            return t
        seq_len = t.shape[1]
        device = t.device
        perm = torch.zeros(seq_len, dtype=torch.long, device=device)
        for new_seg_idx in range(n_segments):
            old_seg_idx = segment_order[new_seg_idx]
            start_new = boundaries[new_seg_idx]
            end_new = min(boundaries[new_seg_idx + 1], seq_len)
            start_old = boundaries[old_seg_idx]
            for i in range(start_new, end_new):
                offset = i - start_new
                perm[i] = start_old + offset
        return t[:, perm, ...].contiguous()

    if isinstance(pe.text_embeds, (list, tuple)):
        te_list = list(pe.text_embeds)
        attn_list = (
            list(pe.attention_mask)
            if pe.attention_mask is not None
            and isinstance(pe.attention_mask, (list, tuple))
            else None
        )
        attn_is_tuple = (
            isinstance(pe.attention_mask, tuple) if pe.attention_mask is not None else False
        )
        for i, t in enumerate(te_list):
            if t.dim() >= 2:
                te_list[i] = apply_segment_perm(t)
                if (
                    attn_list is not None
                    and i < len(attn_list)
                    and attn_list[i] is not None
                    and attn_list[i].dim() >= 2
                ):
                    attn_list[i] = apply_segment_perm(attn_list[i])
        pe.text_embeds = tuple(te_list) if isinstance(pe.text_embeds, tuple) else te_list
        if attn_list is not None:
            pe.attention_mask = tuple(attn_list) if attn_is_tuple else attn_list
    else:
        pe.text_embeds = apply_segment_perm(pe.text_embeds)
        if pe.attention_mask is not None and pe.attention_mask.dim() >= 2:
            pe.attention_mask = apply_segment_perm(pe.attention_mask)

    if debug:
        print(f"  [shuffle_sequence] Shuffled {n_segments} segments, order={segment_order}")


def compare_embeddings(
    pe_a: PromptEmbeds,
    pe_b: PromptEmbeds,
    name_a: str = "A",
    name_b: str = "B",
) -> None:
    """Print shape and similarity metrics between two PromptEmbeds (single batch)."""

    def _t(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    ta = _t(pe_a.text_embeds)
    tb = _t(pe_b.text_embeds)
    if ta.dim() == 3:
        ta, tb = ta.squeeze(0), tb.squeeze(0)
    seq_a, seq_b = ta.shape[0], tb.shape[0]
    print(f"  Shape {name_a}: {ta.shape}, {name_b}: {tb.shape}")

    if seq_a != seq_b:
        print(f"  WARNING: sequence length mismatch ({seq_a} vs {seq_b}); comparing up to min.")
    seq = min(seq_a, seq_b)
    va = ta[:seq].float().reshape(-1)
    vb = tb[:seq].float().reshape(-1)

    # Simple checks first (in case they are identical)
    exact_equal = torch.equal(ta[:seq], tb[:seq])
    print(f"  Exact equal (tensors): {exact_equal}")
    allclose_1e5 = torch.allclose(va, vb, rtol=0, atol=1e-5)
    allclose_1e6 = torch.allclose(va, vb, rtol=0, atol=1e-6)
    print(f"  allclose(atol=1e-5): {allclose_1e5}, allclose(atol=1e-6): {allclose_1e6}")

    cos = torch.nn.functional.cosine_similarity(
        va.unsqueeze(0), vb.unsqueeze(0), dim=1
    ).item()
    l2 = torch.norm(va - vb).item()
    max_diff = (va - vb).abs().max().item()
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  L2 distance:       {l2:.6f}")
    print(f"  Max |diff|:        {max_diff:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare shuffle_sequence vs encode_prompt(caption_shuffled)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("ZIMAGE_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="ZImage model path",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="red flower, outdoor",
        help="Caption (comma-separated segments)",
    )
    parser.add_argument(
        "--caption_shuffled",
        type=str,
        default="outdoor, red flower",
        help="Shuffled caption for method B",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle_sequence (when not using override)",
    )
    parser.add_argument(
        "--fixed_order",
        action="store_true",
        help="Use fixed segment order [1,0] for 2 segments (reproducible match with caption_shuffled)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print segment boundaries and shuffle info",
    )
    args = parser.parse_args()

    if not args.model_path or not os.path.exists(args.model_path):
        print(
            "Provide a valid --model_path or set ZIMAGE_MODEL_PATH to a ZImage model directory."
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading ZImage model...")
    sd = _load_zimage(args.model_path, device=device)

    caption = args.caption
    caption_shuffled = args.caption_shuffled

    with torch.no_grad():
        # Method A: encode original caption, then shuffle segments in token space
        pe_original = sd.encode_prompt(caption)
        if isinstance(pe_original.text_embeds, (list, tuple)):
            seq_len = pe_original.text_embeds[0].shape[1]
        else:
            seq_len = pe_original.text_embeds.shape[1]

        tokenizer = sd.tokenizer[0]
        boundaries = get_segment_boundaries_from_prompt(tokenizer, caption, seq_len)
        if args.debug:
            print(f"  segment_boundaries: {boundaries}")

        pe_shuffle_sequence = pe_original.clone()
        pe_shuffle_sequence.segment_boundaries = boundaries
        segment_order_override = (
            [1, 0] if args.fixed_order and len(boundaries) == 3 else None
        )
        if segment_order_override and args.debug:
            print(f"  segment_order_override: {segment_order_override}")
        random.seed(args.seed)
        shuffle_sequence(
            pe_shuffle_sequence,
            segment_order_override=segment_order_override,
            debug=args.debug,
        )

        # Method B: encode pre-shuffled caption string
        pe_encode_shuffled = sd.encode_prompt(caption_shuffled)

    print("\n--- Method A: encode(caption) + shuffle_sequence ---")
    print(f"  caption: {caption}")
    print("\n--- Method B: encode(caption_shuffled) ---")
    print(f"  caption_shuffled: {caption_shuffled}")
    print("\n--- Comparison A vs B ---")
    compare_embeddings(
        pe_shuffle_sequence,
        pe_encode_shuffled,
        "A (shuffle_sequence)",
        "B (encode_shuffled)",
    )
    print()


if __name__ == "__main__":
    main()
