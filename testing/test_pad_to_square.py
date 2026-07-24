"""Unit tests for pad-to-square letterbox geometry, validity masks, and cache keys."""

import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import (
    AiToolkitDataset,
    get_dataloader_from_datasets,
    resize_dataloader_batch_size,
)
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.dataloader_mixins import spatial_resize_crop_pil


class FakeSD:
    def __init__(self, arch="zimage_diffsynth", div=32):
        class MC:
            latent_space_version = None
            is_pixart_sigma = False

        self.arch = arch
        self.model_config = MC()
        self.model_config.arch = arch
        self.use_raw_control_images = False
        self.is_xl = False
        self.is_vega = False
        self.is_ssd = False
        self.is_v3 = False
        self.is_auraflow = False
        self.is_flux = False
        self.te_padding_side = "right"
        self._div = div
        self.device = "cpu"
        self.device_torch = torch.device("cpu")
        self.torch_dtype = torch.float32

    def get_bucket_divisibility(self):
        return self._div

    def encode_control_in_text_embeddings(self, *args, **kwargs):
        return False

    def set_device_state_preset(self, *args, **kwargs):
        pass

    def restore_device_state(self):
        pass

    def encode_images(self, imgs):
        # imgs: B,C,H,W → fake latents B,4,H/8,W/8
        b, _, h, w = imgs.shape
        return torch.zeros(b, 4, max(1, h // 8), max(1, w // 8), dtype=imgs.dtype, device=imgs.device)


def _make_image_dir(sizes):
    """Create temp dir with RGB images of given (name, w, h) sizes. Returns path."""
    d = tempfile.mkdtemp(prefix="pad_sq_")
    for name, w, h in sizes:
        Image.new("RGB", (w, h), color=(200, 100, 50)).save(os.path.join(d, name))
    return d


def _file_item(path, ds_cfg, **geo):
    return FileItemDTO(
        path=path,
        dataset_config=ds_cfg,
        dataloader_transforms=None,
        size_database={},
        dataset_root=os.path.dirname(path),
        **geo,
    )


def test_auto_enable_pad_to_square_for_zimage_diffsynth():
    d = _make_image_dir([("a.png", 128, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        default_caption="x",
    )
    assert ds_cfg.pad_to_square is False
    ds = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    assert ds_cfg.pad_to_square is True
    assert ds_cfg.buckets is True
    # Capability on, but inactive at batch_size==1
    assert ds.is_pad_to_square_active() is False


def test_pad_to_square_not_auto_for_other_arch():
    d = _make_image_dir([("a.png", 128, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        default_caption="x",
    )
    _ = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(arch="sd1"), train_config=None)
    assert ds_cfg.pad_to_square is False


def test_batch_size_1_uses_ar_buckets_not_letterbox():
    d = _make_image_dir(
        [
            ("land.png", 128, 64),
            ("port.png", 64, 128),
        ]
    )
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    assert dataset.is_pad_to_square_active() is False
    # Mixed AR → more than one bucket key (not forced 64x64 letterbox)
    assert "64x64" not in dataset.buckets or len(dataset.buckets) >= 1
    by_name = {os.path.basename(fi.path): fi for fi in dataset.file_list}
    land = by_name["land.png"]
    assert land.pad_to_square_active is False
    assert land.pad_x == 0 and land.pad_y == 0
    # Landscape AR bucket is wider than tall (not letterboxed square content+pad)
    assert land.crop_width != land.crop_height or land.scale_to_width == land.crop_width
    assert land.has_mask_image is False


def test_single_square_bucket_landscape_portrait_square():
    d = _make_image_dir(
        [
            ("land.png", 128, 64),
            ("port.png", 64, 128),
            ("sq.png", 80, 80),
        ]
    )
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    assert list(dataset.buckets.keys()) == ["64x64"]
    assert len(dataset.buckets["64x64"].file_list_idx) == 3

    by_name = {os.path.basename(fi.path): fi for fi in dataset.file_list}
    land = by_name["land.png"]
    assert land.pad_to_square_active is True
    assert land.scale_to_width == 64
    assert land.scale_to_height == 32
    assert land.pad_x == 0
    assert land.pad_y == 16
    assert land.content_width == 64
    assert land.content_height == 32

    port = by_name["port.png"]
    assert port.scale_to_width == 32
    assert port.scale_to_height == 64
    assert port.pad_x == 16
    assert port.pad_y == 0

    sq = by_name["sq.png"]
    assert sq.scale_to_width == 64
    assert sq.scale_to_height == 64
    assert sq.pad_x == 0 and sq.pad_y == 0


def test_spatial_resize_letterbox_pil_output():
    d = _make_image_dir([("land.png", 100, 50)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    fi = dataset.file_list[0]
    img = Image.new("RGB", (100, 50), color=(255, 0, 0))
    out = spatial_resize_crop_pil(fi, img)
    assert out.size == (64, 64)
    # Corners should be pad fill (0), center content red
    px = out.getpixel((0, 0))
    assert px == (0, 0, 0)
    cx, cy = 32, 32
    center = out.getpixel((cx, cy))
    assert center[0] > 200


def test_validity_mask_and_user_mask_product():
    d = _make_image_dir([("land.png", 128, 64)])
    mask_dir = tempfile.mkdtemp(prefix="pad_mask_")
    # User mask: all ones (white)
    Image.new("L", (128, 64), color=255).save(os.path.join(mask_dir, "land.png"))

    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
        mask_path=mask_dir,
        mask_min_value=0.25,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    fi = dataset.file_list[0]
    fi.load_mask_image()
    assert fi.image_valid_mask_tensor is not None
    assert fi.mask_tensor is not None
    valid = fi.image_valid_mask_tensor
    loss_m = fi.mask_tensor
    # Pad region: validity 0, loss mask 0 (even with mask_min_value)
    assert float(valid[0, 0, 0]) == 0.0
    assert float(loss_m[0, 0, 0]) == 0.0
    # Content region: validity 1
    cy = 16 + 8  # pad_y=16, mid content
    assert float(valid[0, cy, 32]) == 1.0
    # User white * validity → content uses mapped min..1; pad stays 0
    assert float(loss_m[0, cy, 32]) >= 0.25


def test_validity_only_without_user_mask():
    d = _make_image_dir([("land.png", 128, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    fi = dataset.file_list[0]
    assert fi.has_mask_image is True
    fi.load_mask_image()
    assert fi.image_valid_mask_tensor is not None
    assert torch.equal(fi.mask_tensor, fi.image_valid_mask_tensor)


def test_batch_missing_validity_is_all_valid_not_zeros():
    d = _make_image_dir([("a.png", 64, 64), ("b.png", 64, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    items = list(dataset.file_list[:2])
    items[0].load_mask_image()
    # Second item: no validity tensor (simulate missing)
    items[1].image_valid_mask_tensor = None
    items[1].mask_tensor = items[0].mask_tensor.clone()
    # Need tensors for batch construction
    for it in items:
        it.tensor = torch.zeros(3, 64, 64)
        it.is_latent_cached = False
    batch = DataLoaderBatchDTO(file_items=items)
    assert batch.image_valid_mask_tensor is not None
    # Missing item filled with ones
    assert float(batch.image_valid_mask_tensor[1].min()) == 1.0


def test_latent_cache_key_includes_pad_fields():
    d = _make_image_dir([("land.png", 128, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=2, sd=FakeSD(), train_config=None)
    fi = dataset.file_list[0]
    info = fi.get_latent_info_dict()
    assert info.get("pad_to_square") is True
    assert "pad_x" in info and "pad_y" in info
    assert "content_width" in info and "content_height" in info

    # Inactive pad (batch_size 1 geometry) → no pad keys
    ds1 = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    fi_b1 = ds1.file_list[0]
    info_b1 = fi_b1.get_latent_info_dict()
    assert "pad_to_square" not in info_b1


def test_resolution_divisibility_check():
    d = _make_image_dir([("a.png", 64, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=48,  # not divisible by 32
        buckets=True,
        bucket_tolerance=64,
        default_caption="x",
    )
    try:
        AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(div=32), train_config=None)
        raised = False
    except ValueError as e:
        raised = True
        assert "divisible" in str(e).lower() or "pad_to_square" in str(e)
    assert raised


def test_resize_batch_size_toggles_letterbox_geometry():
    d = _make_image_dir(
        [
            ("land.png", 128, 64),
            ("port.png", 64, 128),
        ]
    )
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
    )
    sd = FakeSD()
    dataloader = get_dataloader_from_datasets([ds_cfg], batch_size=1, sd=sd)
    ds = dataloader.dataset.datasets[0]
    assert ds.is_pad_to_square_active() is False
    land = next(fi for fi in ds.file_list if fi.path.endswith("land.png"))
    assert land.pad_to_square_active is False
    ar_crop = (land.crop_width, land.crop_height)

    resize_dataloader_batch_size(dataloader, 2)
    assert ds.batch_size == 2
    assert ds.is_pad_to_square_active() is True
    assert list(ds.buckets.keys()) == ["64x64"]
    land = next(fi for fi in ds.file_list if fi.path.endswith("land.png"))
    assert land.pad_to_square_active is True
    assert land.pad_y == 16
    assert (land.crop_width, land.crop_height) == (64, 64)
    assert (land.crop_width, land.crop_height) != ar_crop or ar_crop == (64, 64)

    resize_dataloader_batch_size(dataloader, 1)
    assert ds.is_pad_to_square_active() is False
    land = next(fi for fi in ds.file_list if fi.path.endswith("land.png"))
    assert land.pad_to_square_active is False
    assert land.pad_x == 0 and land.pad_y == 0


def test_resize_creates_pad_latent_cache_when_starting_at_batch_size_1():
    d = _make_image_dir([("land.png", 128, 64), ("sq.png", 64, 64)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=64,
        buckets=True,
        bucket_tolerance=32,
        default_caption="x",
        pad_to_square=True,
        cache_latents_to_disk=True,
    )
    sd = FakeSD()
    dataloader = get_dataloader_from_datasets([ds_cfg], batch_size=1, sd=sd)
    ds = dataloader.dataset.datasets[0]
    # At B=1, AR cache may exist; pad keys must not be in hash
    for fi in ds.file_list:
        info = fi.get_latent_info_dict()
        assert "pad_to_square" not in info
        assert fi.is_latent_cached is True

    cache_dir = os.path.join(d, "_latent_cache")
    before = set(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else set()

    resize_dataloader_batch_size(dataloader, 2)

    assert ds.is_pad_to_square_active() is True
    after = set(os.listdir(cache_dir))
    # New pad-geometry cache files created
    assert after - before
    for fi in ds.file_list:
        assert fi.pad_to_square_active is True
        assert fi.is_latent_cached is True
        info = fi.get_latent_info_dict()
        assert info.get("pad_to_square") is True
        path = fi.get_latent_path(recalculate=True)
        assert os.path.exists(path)
