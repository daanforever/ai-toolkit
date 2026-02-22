"""
Custom loading of ZImageTransformer2DModel from shards or single file,
bypassing the slow diffusers path (init_empty_weights + load_model_dict_into_meta).
"""
import json
import os
from typing import List, Optional

import torch
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file as safetensors_load_file


def load_zimage_transformer_from_shards(
    pretrained_model_name_or_path: str,
    *,
    subfolder: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> ZImageTransformer2DModel:
    """
    Load ZImageTransformer2DModel from a local folder (sharded or single safetensors),
    using from_config + load_state_dict(..., assign=True) for faster loading.

    Supports only local paths. For Hub model IDs use ZImageTransformer2DModel.from_pretrained.
    """
    # 1) Config
    load_config_kwargs = {}
    for key in ("cache_dir", "local_files_only", "token", "revision"):
        if key in kwargs:
            load_config_kwargs[key] = kwargs[key]
    config, _unused, _commit_hash = ZImageTransformer2DModel.load_config(
        pretrained_model_name_or_path,
        subfolder=subfolder or "",
        return_unused_kwargs=True,
        return_commit_hash=True,
        **load_config_kwargs,
    )

    # 2) Resolve folder and shards vs single file (local only)
    cached_folder = os.path.join(
        os.path.expanduser(pretrained_model_name_or_path), subfolder or ""
    )
    if not os.path.isdir(cached_folder):
        raise ValueError(
            "load_zimage_transformer_from_shards currently only supports local paths. "
            f"Folder not found: {cached_folder}. "
            "For Hub model IDs use ZImageTransformer2DModel.from_pretrained."
        )

    index_path = os.path.join(cached_folder, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_paths: List[str] = [
            os.path.join(cached_folder, name) for name in shard_names
        ]
        is_sharded = True
    else:
        single_path = os.path.join(cached_folder, SAFETENSORS_WEIGHTS_NAME)
        if not os.path.isfile(single_path):
            raise FileNotFoundError(
                f"Neither index {SAFE_WEIGHTS_INDEX_NAME} nor single weights "
                f"{SAFETENSORS_WEIGHTS_NAME} found in {cached_folder}"
            )
        shard_paths = [single_path]
        is_sharded = False

    # 3) Create model (no init_empty_weights)
    model = ZImageTransformer2DModel.from_config(config)

    # 4) Load weights
    for path in shard_paths:
        state_dict = safetensors_load_file(path, device="cpu")
        if is_sharded:
            model.load_state_dict(state_dict, assign=True, strict=False)
        else:
            if hasattr(model, "_fix_state_dict_keys_on_load"):
                model._fix_state_dict_keys_on_load(state_dict)
            model.load_state_dict(state_dict, assign=True, strict=True)

    # 5) Dtype and return
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model
