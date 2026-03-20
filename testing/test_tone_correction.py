import os
import sys

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from testing.fixture_paths import FIXTURE_IMAGES_DIR
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset, ImageToModelSpaceTransform, RescaleTransform
from toolkit.dataloader_mixins import apply_tone_correction_tensor, spatial_resize_crop_pil


class FakeSD:
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

    def get_bucket_divisibility(self):
        return 32

    def encode_control_in_text_embeddings(self, *args, **kwargs):
        return False


def _tensor_01_after_crop(fi):
    """Same geometry as dataloader scan (exif + RGB + spatial_resize_crop_pil)."""
    img = Image.open(fi.path)
    img = exif_transpose(img).convert("RGB")
    img = spatial_resize_crop_pil(fi, img)
    return transforms.ToTensor()(img)


def _file_item_by_suffix(dataset, suffix: str):
    return next(x for x in dataset.file_list if x.path.replace("\\", "/").endswith(suffix))


def test_tone_targets_computed_and_means_closer():
    ds_cfg = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        tone_correction=True,
        default_caption="x",
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    assert ds_cfg.tone_target_mean is not None
    assert ds_cfg.tone_target_std is not None
    assert len(ds_cfg.tone_target_mean) == 3
    assert len(ds_cfg.tone_target_std) == 3

    tt = torch.tensor(ds_cfg.tone_target_mean, dtype=torch.float32)

    fi_a = _file_item_by_suffix(dataset, "fixture_a.png")
    fi_b = _file_item_by_suffix(dataset, "fixture_b.png")
    t0_a = _tensor_01_after_crop(fi_a)
    t0_b = _tensor_01_after_crop(fi_b)
    assert not torch.allclose(
        t0_a.mean(dim=(1, 2)), t0_b.mean(dim=(1, 2)), atol=0.02
    )
    t1_a = apply_tone_correction_tensor(t0_a.clone(), ds_cfg)
    t1_b = apply_tone_correction_tensor(t0_b.clone(), ds_cfg)
    assert torch.allclose(t1_a.mean(dim=(1, 2)), tt, atol=1e-4)
    assert torch.allclose(t1_b.mean(dim=(1, 2)), tt, atol=1e-4)

    for name in ("fixture_a.png", "fixture_b.png", "fixture_1.jpg", "fixture_2.png"):
        fi = _file_item_by_suffix(dataset, name)
        x0 = _tensor_01_after_crop(fi)
        x1 = apply_tone_correction_tensor(x0.clone(), ds_cfg)
        assert torch.allclose(x1.mean(dim=(1, 2)), tt, atol=1e-4), name


def test_image_to_model_space_transform_end_to_end():
    ds_cfg = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=32,
        buckets=True,
        bucket_tolerance=32,
        tone_correction=True,
        default_caption="x",
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    tail = transforms.Compose([RescaleTransform()])
    pipe = ImageToModelSpaceTransform(ds_cfg, tail)
    for suffix in ("fixture_a.png", "fixture_1.jpg"):
        fi = _file_item_by_suffix(dataset, suffix)
        img = Image.open(fi.path)
        img = exif_transpose(img).convert("RGB")
        img = spatial_resize_crop_pil(fi, img)
        out = pipe(img)
        assert out.shape[0] == 3
        assert out.min() >= -1.0 and out.max() <= 1.0, suffix


def test_apply_tone_correction_no_op_when_disabled():
    x = torch.rand(3, 8, 8)
    dc = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        tone_correction=False,
        default_caption="x",
    )
    dc.tone_target_mean = [0.2, 0.3, 0.4]
    dc.tone_target_std = [0.1, 0.1, 0.1]
    y = apply_tone_correction_tensor(x, dc)
    assert torch.allclose(y, x)


def test_apply_tone_correction_no_op_when_targets_missing():
    x = torch.rand(3, 8, 8)
    dc = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        tone_correction=True,
        default_caption="x",
    )
    assert dc.tone_target_mean is None and dc.tone_target_std is None
    y = apply_tone_correction_tensor(x, dc)
    assert torch.allclose(y, x)


def test_tone_correction_disabled_leaves_targets_none():
    ds_cfg = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        tone_correction=False,
        default_caption="x",
    )
    _ = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    assert ds_cfg.tone_target_mean is None
    assert ds_cfg.tone_target_std is None


def test_apply_tone_correction_constant_channel_shift_branch():
    ds_cfg = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        tone_correction=True,
        default_caption="x",
    )
    _ = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    tt = torch.tensor(ds_cfg.tone_target_mean, dtype=torch.float32)
    x = torch.zeros(3, 4, 4)
    x[0].fill_(0.2)
    x[1].fill_(0.55)
    x[2].fill_(0.9)
    y = apply_tone_correction_tensor(x.clone(), ds_cfg)
    assert torch.allclose(y.mean(dim=(1, 2)), tt, atol=1e-3)


def test_apply_tone_correction_clamps_to_unit_interval():
    dc = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        tone_correction=True,
        default_caption="x",
    )
    dc.tone_target_mean = [0.5, 0.5, 0.5]
    dc.tone_target_std = [1.0, 1.0, 1.0]
    x = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    y = apply_tone_correction_tensor(x, dc)
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert (y >= 1.0 - 1e-6).any()


def test_apply_tone_correction_scaled_path_mean_and_std_match_targets():
    dc = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        tone_correction=True,
        default_caption="x",
    )
    dc.tone_target_mean = [0.4, 0.5, 0.6]
    dc.tone_target_std = [0.2, 0.2, 0.2]
    base = torch.linspace(-1.0, 1.0, steps=16 * 16, dtype=torch.float32).reshape(16, 16)
    base = (base - base.mean()) / base.std()
    x = torch.stack([base * 0.1 + 0.4, base * 0.1 + 0.5, base * 0.1 + 0.6])
    y = apply_tone_correction_tensor(x, dc)
    tm = torch.tensor(dc.tone_target_mean, dtype=torch.float32)
    ts = torch.tensor(dc.tone_target_std, dtype=torch.float32)
    assert torch.allclose(y.mean(dim=(1, 2)), tm, atol=1e-5)
    assert torch.allclose(y.std(dim=(1, 2)), ts, atol=1e-4)


def test_apply_tone_correction_preserves_float64():
    ds_cfg = DatasetConfig(
        dataset_path=str(FIXTURE_IMAGES_DIR),
        resolution=64,
        buckets=True,
        bucket_tolerance=64,
        tone_correction=True,
        default_caption="x",
    )
    _ = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(), train_config=None)
    x = torch.rand(3, 8, 8, dtype=torch.float64)
    y = apply_tone_correction_tensor(x, ds_cfg)
    assert y.dtype == torch.float64
