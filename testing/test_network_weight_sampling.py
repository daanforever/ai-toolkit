"""Tests for excluding training items with network_weight == 0 from sampling."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from testing.fixture_paths import FIXTURE_IMAGES_DIR
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import (
    _active_concat_indices,
    get_dataloader_datasets,
    get_dataloader_from_datasets,
    rebuild_dataloader_network_weights,
)
from toolkit.unified_bucket_manager import UnifiedBucketManager


class FakeSD:
    """Minimal stub for dataloader tests."""

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


def _image_config(network_weight: float, *, buckets: bool = False) -> DatasetConfig:
    return DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=512,
        default_caption="default",
        buckets=buckets,
        bucket_tolerance=64,
        shrink_video_to_frames=True,
        num_frames=1,
        network_weight=network_weight,
    )


@pytest.fixture
def fake_sd():
    return FakeSD()


def test_unified_buckets_exclude_zero_weight():
    ref_config = SimpleNamespace(
        network_weight=1.0,
        resolution=512,
        current_resolution=512,
        bucket_tolerance=64,
        scale=1.0,
        random_scale=False,
        random_crop=False,
        augments=None,
        transforms=None,
    )
    inactive_config = SimpleNamespace(
        network_weight=0.0,
        resolution=512,
        current_resolution=512,
        bucket_tolerance=64,
        scale=1.0,
        random_scale=False,
        random_crop=False,
        augments=None,
        transforms=None,
    )
    bucket = MagicMock(file_list_idx=[0, 1])
    active_ds = MagicMock(
        dataset_config=ref_config,
        dataset_path="active",
        buckets={"512x512": bucket},
    )
    inactive_ds = MagicMock(
        dataset_config=inactive_config,
        dataset_path="inactive",
        buckets={"512x512": bucket},
    )

    bm = UnifiedBucketManager([active_ds, inactive_ds], batch_size=2)
    bm.build_unified_buckets()

    elements = bm.unified_buckets["512x512"]
    assert len(elements) == 2
    assert all(dataset_idx == 0 for dataset_idx, _ in elements)


def test_non_bucket_sampler_excludes_zero_weight(fake_sd):
    active = _image_config(1.0, buckets=False)
    inactive = _image_config(0.0, buckets=False)
    dataloader = get_dataloader_from_datasets(
        [active, inactive], batch_size=1, sd=fake_sd
    )

    concat = dataloader.dataset
    indices = _active_concat_indices(concat)
    active_len = len(get_dataloader_datasets(dataloader)[0].file_list)
    assert indices == list(range(active_len))
    assert dataloader.sampler is not None

    seen_weights = set()
    for batch in dataloader:
        for item in batch.file_items:
            seen_weights.add(item.network_weight)
    assert seen_weights == {1.0}


def test_all_zero_network_weight_raises(fake_sd):
    inactive = _image_config(0.0, buckets=False)
    with pytest.raises(ValueError, match="network_weight == 0"):
        get_dataloader_from_datasets([inactive], batch_size=1, sd=fake_sd)


def test_runtime_weight_zero_rebuilds_sampling(fake_sd):
    active = _image_config(1.0, buckets=False)
    inactive = _image_config(1.0, buckets=False)
    dataloader = get_dataloader_from_datasets(
        [active, inactive], batch_size=1, sd=fake_sd
    )
    datasets = get_dataloader_datasets(dataloader)
    assert len(_active_concat_indices(dataloader.dataset)) > 0

    datasets[1].dataset_config.network_weight = 0.0
    for file_item in datasets[1].file_list:
        file_item.network_weight = 0.0

    dataloader = rebuild_dataloader_network_weights(dataloader)
    indices = _active_concat_indices(dataloader.dataset)
    active_len = len(datasets[0].file_list)
    assert indices == list(range(active_len))

    seen_weights = set()
    for batch in dataloader:
        for item in batch.file_items:
            seen_weights.add(item.network_weight)
    assert seen_weights == {1.0}


def test_bucket_runtime_weight_zero_rebuilds_sampling(fake_sd):
    active = _image_config(1.0, buckets=True)
    inactive = _image_config(1.0, buckets=True)
    dataloader = get_dataloader_from_datasets(
        [active, inactive], batch_size=1, sd=fake_sd
    )
    datasets = get_dataloader_datasets(dataloader)
    bm = dataloader.dataset.bucket_manager
    assert any(idx == 1 for batch in bm.batch_indices for idx, _ in batch)

    datasets[1].dataset_config.network_weight = 0.0
    for file_item in datasets[1].file_list:
        file_item.network_weight = 0.0

    dataloader = rebuild_dataloader_network_weights(dataloader)
    bm = dataloader.dataset.bucket_manager
    assert all(idx == 0 for batch in bm.batch_indices for idx, _ in batch)
