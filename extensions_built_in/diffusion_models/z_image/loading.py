"""
Custom loading of ZImageTransformer2DModel from shards or single file,
using the same low-memory path as diffusers (meta device + load_model_dict_into_meta)
with local path/shard resolution and a single entry point.
"""
import gc
import json
import os
import time
from typing import List, Optional

import torch
from accelerate import init_empty_weights
from diffusers.models.model_loading_utils import load_model_dict_into_meta, load_state_dict
from diffusers.models.modeling_utils import no_init_weights
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFETENSORS_WEIGHTS_NAME
from diffusers.utils import logging as diffusers_logging


def load_zimage_transformer_from_shards(
    pretrained_model_name_or_path: str,
    *,
    subfolder: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> ZImageTransformer2DModel:
    """
    Load ZImageTransformer2DModel from a local folder (sharded or single safetensors).

    Uses meta device and load_model_dict_into_meta per shard (same as from_pretrained
    with low_cpu_mem_usage), so peak RAM is on the order of one shard, not model + shard.

    Supports only local paths. For Hub model IDs use ZImageTransformer2DModel.from_pretrained.
    """
    start = time.perf_counter()
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

    # 3) Create model on meta device (low peak RAM)
    with no_init_weights():
        with init_empty_weights():
            model = ZImageTransformer2DModel.from_config(config)

    # 4) Load weights per shard via load_model_dict_into_meta
    if len(shard_paths) > 1:
        shard_iter = diffusers_logging.tqdm(shard_paths, desc="Loading checkpoint shards")
    else:
        shard_iter = shard_paths
    for path in shard_iter:
        state_dict = load_state_dict(path)
        if not is_sharded and hasattr(model, "_fix_state_dict_keys_on_load"):
            model._fix_state_dict_keys_on_load(state_dict)
        load_model_dict_into_meta(
            model,
            state_dict,
            dtype=torch_dtype,
            device_map=None,
        )
        del state_dict
        gc.collect()

    # 5) Dtype and return
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    elapsed = time.perf_counter() - start
    diffusers_logging.get_logger(__name__).info("Loaded in %.2fs", elapsed)
    return model
