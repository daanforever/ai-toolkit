"""
Disk cache for sample prompts (conditional/unconditional embeddings).
Uses the same _t_e_cache directory and format as text embedding cache (PromptEmbeds.save_multi / load).
"""
import base64
import hashlib
import json
import os
from typing import Dict, Any, Tuple

from toolkit.dataloader_mixins import get_t_e_cache_dir
from toolkit.prompt_utils import PromptEmbeds


def get_sample_prompt_hash(info_dict: Dict[str, Any]) -> str:
    """Compute hash string for sample prompt cache invalidation (same algorithm as text embedding cache)."""
    hash_input = json.dumps(info_dict, sort_keys=True).encode('utf-8')
    hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
    return hash_str.replace('=', '')


def get_sample_prompt_path(cache_dir: str, index: int, hash_str: str) -> str:
    """Return path to cached sample prompt file: cache_dir/sample_{index}_{hash}.safetensors."""
    return os.path.join(cache_dir, f'sample_{index}_{hash_str}.safetensors')


def load_sample_prompt_pair(path: str) -> Tuple[PromptEmbeds, PromptEmbeds]:
    """Load conditional (v0) and unconditional (v1) PromptEmbeds from a single multi-variant file."""
    conditional = PromptEmbeds.load(path, variant_index=0)
    unconditional = PromptEmbeds.load(path, variant_index=1)
    return conditional, unconditional
