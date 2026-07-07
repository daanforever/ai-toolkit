"""Tests for runtime batch_size resize without full dataset recreation."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from testing.fixture_paths import FIXTURE_IMAGES_DIR
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import (
    get_dataloader_from_datasets,
    get_dataloader_datasets,
    resize_dataloader_batch_size,
)
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.unified_bucket_manager import UnifiedBucketManager


class FakeSD:
    """Minimal stub for bucket dataloader tests."""

    def __init__(self):
        class MC:
            latent_space_version = None
            arch = "sd1"
            is_pixart_sigma = False

        self.model_config = MC()
        self.use_raw_control_images = False
        self.is_xl = False
        self.is_vega = False
        self.is_ssd = False
        self.is_v3 = False
        self.is_auraflow = False
        self.is_flux = False
        self.te_padding_side = "right"

    def encode_control_in_text_embeddings(self, *args, **kwargs):
        return None

    def get_bucket_divisibility(self):
        return 32


def _bucket_dataloader(batch_size=1):
    dataset_config = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=512,
        default_caption="default",
        buckets=True,
        bucket_tolerance=64,
        shrink_video_to_frames=True,
        num_frames=1,
    )
    return get_dataloader_from_datasets([dataset_config], batch_size=batch_size, sd=FakeSD())


def test_update_batch_size_preserves_datasets():
    dataloader = _bucket_dataloader(batch_size=1)
    datasets_before = list(get_dataloader_datasets(dataloader))
    dataset_ids_before = [id(ds) for ds in datasets_before]
    epoch_before = datasets_before[0].epoch_num

    resize_dataloader_batch_size(dataloader, 2, epoch_num=epoch_before)

    datasets_after = list(get_dataloader_datasets(dataloader))
    assert [id(ds) for ds in datasets_after] == dataset_ids_before
    assert dataloader.dataset.bucket_manager.batch_size == 2
    assert all(ds.batch_size == 2 for ds in datasets_after)


def test_update_batch_size_changes_batch_indices():
    dataloader = _bucket_dataloader(batch_size=1)
    bm = dataloader.dataset.bucket_manager

    resize_dataloader_batch_size(dataloader, 2)

    assert bm.batch_size == 2
    assert all(len(batch) <= 2 for batch in bm.batch_indices)


def test_resize_syncs_epoch_num():
    dataloader = _bucket_dataloader(batch_size=1)
    target_epoch = 7
    resize_dataloader_batch_size(dataloader, 2, epoch_num=target_epoch)
    for ds in get_dataloader_datasets(dataloader):
        assert ds._epoch_num == target_epoch


def test_non_bucket_resize_reuses_concat_dataset():
    dataset_config = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=512,
        default_caption="default",
        buckets=False,
        num_frames=1,
    )
    dataloader = get_dataloader_from_datasets([dataset_config], batch_size=1, sd=FakeSD())
    dataset_before = dataloader.dataset

    new_loader = resize_dataloader_batch_size(dataloader, 2, epoch_num=3)

    assert new_loader.dataset is dataset_before
    assert new_loader.batch_size == 2


def test_unified_bucket_manager_update_batch_size_noop():
    config = SimpleNamespace(
        resolution=512,
        bucket_tolerance=64,
        scale=1.0,
        random_scale=False,
        random_crop=False,
        augments=None,
        transforms=None,
    )
    datasets = [MagicMock(dataset_config=config)]
    bm = UnifiedBucketManager(datasets, batch_size=2)
    bm.batch_indices = [[(0, 0), (0, 1)]]
    bm.update_batch_size(2)
    assert bm.batch_size == 2
    assert bm.batch_indices == [[(0, 0), (0, 1)]]


def _minimal_file_item(**kwargs):
    defaults = {
        "is_latent_cached": False,
        "extra_values": [],
        "audio_data": None,
        "loss_multiplier": 1.0,
        "network_weight": 1.0,
        "is_reg": False,
        "control_tensor": None,
        "control_tensor_list": None,
        "clip_image_tensor": None,
        "mask_tensor": None,
        "unaugmented_tensor": None,
        "unconditional_tensor": None,
        "clip_image_embeds": None,
        "clip_image_embeds_unconditional": None,
        "prompt_embeds": None,
        "audio_tensor": None,
        "inpaint_tensor": None,
        "_cached_first_frame_latent": None,
        "_cached_audio_latent": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_dto_is_latents_cached_requires_all_items():
    tensor = torch.randn(3, 64, 64)
    cached_item = _minimal_file_item(
        is_latent_cached=True,
        get_latent=lambda: tensor,
        tensor=tensor,
    )
    uncached_item = _minimal_file_item(
        is_latent_cached=False,
        tensor=tensor,
    )

    batch = DataLoaderBatchDTO(file_items=[cached_item, uncached_item])
    assert batch.tensor is not None
    assert batch.latents is None


def test_dto_all_cached_uses_latents_path():
    tensor = torch.randn(3, 64, 64)
    items = [
        _minimal_file_item(is_latent_cached=True, get_latent=lambda: tensor, tensor=tensor)
        for _ in range(2)
    ]
    batch = DataLoaderBatchDTO(file_items=items)
    assert batch.latents is not None
    assert batch.tensor is None


def test_dto_unconditional_tensor_uses_unconditional_field():
    tensor = torch.randn(3, 8, 8)
    base = torch.ones(3, 8, 8)
    items = [
        _minimal_file_item(unconditional_tensor=base, tensor=tensor),
        _minimal_file_item(unconditional_tensor=None, tensor=tensor),
    ]
    batch = DataLoaderBatchDTO(file_items=items)
    assert batch.unconditional_tensor is not None
    assert batch.unconditional_tensor.shape[0] == 2


def test_backward_done_branches_in_train_single_accumulation():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    import inspect

    source = inspect.getsource(SDTrainer.train_single_accumulation)
    assert "backward_done = False" in source
    assert 'backward_done = guidance_type != "targeted_flow"' in source
    assert "backward_done = True" in source
