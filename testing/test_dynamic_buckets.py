"""Unit tests for dynamic bucket aspect snapping in toolkit/buckets.py."""

import math
import os
import sys
import tempfile

from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch

from toolkit.buckets import get_bucket_for_image_size
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset, PairedImageDataset


class FakeSD:
    def __init__(self, arch="sd1", div=32):
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
        b, _, h, w = imgs.shape
        return torch.zeros(b, 4, max(1, h // 8), max(1, w // 8), dtype=imgs.dtype, device=imgs.device)


def _expected_long_short(width, height, resolution, divisibility):
    long_side = max(width, height)
    short_side = min(width, height)
    effective = min(resolution if resolution is not None else long_side, long_side)
    target_long = max(divisibility, (effective // divisibility) * divisibility)
    target_short = max(divisibility, (((short_side * target_long) // long_side) // divisibility) * divisibility)
    return target_long, target_short


# (width, height, resolution, divisibility, expected_w, expected_h)
EXACT_CASES = [
    # int()-truncation triggers (old code gave 320x480 / 320x448)
    (483, 725, 512, 32, 320, 512),
    (725, 483, 512, 32, 512, 320),
    (429, 644, 512, 64, 320, 512),
    (644, 429, 512, 64, 512, 320),
    # parity inputs unchanged
    (2000, 3000, 1024, 32, 672, 1024),
    (3000, 2000, 1024, 32, 1024, 672),
    (1500, 1500, 1024, 32, 1024, 1024),
    # smaller than resolution snaps down, never up
    (400, 600, 1024, 32, 384, 576),
    (600, 400, 1024, 32, 576, 384),
    # exactly-divisible scaled short side (float-boundary guard)
    (400, 600, 1024, 32, 384, 576),
    # resolution=None non-clamped
    (640, 480, None, 32, 640, 480),
    # clamp case: short side floors to 0 -> clamped (old code gave 96x96)
    (100, 101, None, 32, 64, 96),
]

# subset where the short side is not clamped: aspect bound applies
ASPECT_CASES = [c for c in EXACT_CASES if (c[0], c[1], c[3]) != (100, 101, 32)]


def test_exact_bucket_dims():
    for width, height, resolution, div, exp_w, exp_h in EXACT_CASES:
        bucket = get_bucket_for_image_size(
            width, height, resolution=resolution, divisibility=div
        )
        assert (bucket["width"], bucket["height"]) == (exp_w, exp_h), (
            f"{width}x{height} res={resolution} div={div}: "
            f"got {bucket['width']}x{bucket['height']}, expected {exp_w}x{exp_h}"
        )


def test_divisibility_and_long_side_exactness():
    for width, height, resolution, div, exp_w, exp_h in EXACT_CASES:
        bucket = get_bucket_for_image_size(
            width, height, resolution=resolution, divisibility=div
        )
        assert bucket["width"] % div == 0
        assert bucket["height"] % div == 0
        assert bucket["width"] >= div and bucket["height"] >= div
        exp_long, exp_short = _expected_long_short(width, height, resolution, div)
        assert max(bucket["width"], bucket["height"]) == exp_long
        assert min(bucket["width"], bucket["height"]) == exp_short
        if width >= height:
            assert bucket["width"] >= bucket["height"]
        else:
            assert bucket["height"] >= bucket["width"]


def test_short_side_is_exact_rational_floor():
    for width, height, resolution, div, _, _ in EXACT_CASES:
        if (width, height) == (100, 101):
            continue  # clamped case asserted exactly elsewhere
        long_side = max(width, height)
        short_side = min(width, height)
        effective = min(resolution if resolution is not None else long_side, long_side)
        target_long = max(div, (effective // div) * div)
        exact_scaled_short = (short_side * target_long) / long_side
        floored = ((short_side * target_long) // long_side)
        bucket = get_bucket_for_image_size(
            width, height, resolution=resolution, divisibility=div
        )
        out_short = min(bucket["width"], bucket["height"])
        assert out_short == (floored // div) * div
        assert 0 <= exact_scaled_short - out_short < div
        # long side has no floor loss beyond the grid snap
        assert max(bucket["width"], bucket["height"]) == target_long


def test_aspect_bound_for_non_clamped():
    for width, height, resolution, div, _, _ in ASPECT_CASES:
        bucket = get_bucket_for_image_size(
            width, height, resolution=resolution, divisibility=div
        )
        long_side = max(width, height)
        short_side = min(width, height)
        effective = min(resolution if resolution is not None else long_side, long_side)
        target_long = max(div, (effective // div) * div)
        scaled_short = short_side * target_long / long_side
        assert scaled_short >= div
        out_aspect = bucket["width"] / bucket["height"]
        src_aspect = width / height
        assert abs(out_aspect / src_aspect - 1) <= div / scaled_short + 1e-9


def test_clamp_case_documents_behavior_change():
    bucket = get_bucket_for_image_size(100, 101, resolution=None, divisibility=32)
    assert (bucket["width"], bucket["height"]) == (64, 96)


def _make_image_dir(sizes):
    d = tempfile.mkdtemp(prefix="dyn_bucket_")
    for name, w, h in sizes:
        Image.new("RGB", (w, h), color=(200, 100, 50)).save(os.path.join(d, name))
    return d


def test_setup_buckets_isotropic_scale_short_axis_crop_only():
    w, h, resolution, div = 483, 725, 512, 32
    d = _make_image_dir([("img.png", w, h)])
    ds_cfg = DatasetConfig(
        dataset_path=d,
        resolution=resolution,
        buckets=True,
        bucket_tolerance=div,
        default_caption="x",
    )
    dataset = AiToolkitDataset(ds_cfg, batch_size=1, sd=FakeSD(arch="sd1", div=div), train_config=None)
    fi = dataset.file_list[0]
    assert (fi.crop_width, fi.crop_height) == (320, 512)
    # isotropic cover scale: both axes share one factor (up to ceil to int pixels)
    f = max(fi.crop_width / w, fi.crop_height / h)
    assert fi.scale_to_width == math.ceil(w * f)
    assert fi.scale_to_height == math.ceil(h * f)
    # crop only on the short axis: portrait -> long (h) slack <= 1px, short slack <= div
    assert 0 <= fi.scale_to_height - fi.crop_height <= 1
    assert 0 <= fi.scale_to_width - fi.crop_width <= div
    assert fi.scale_to_width - fi.crop_width > 0 or fi.scale_to_height - fi.crop_height >= 0


def test_paired_image_branch_output_matches_bucket():
    w, h, size = 483, 725, 512
    pos_d = tempfile.mkdtemp(prefix="paired_pos_")
    neg_d = tempfile.mkdtemp(prefix="paired_neg_")
    Image.new("RGB", (w, h), color=(10, 200, 30)).save(os.path.join(pos_d, "pair.png"))
    Image.new("RGB", (w, h), color=(200, 30, 10)).save(os.path.join(neg_d, "pair.png"))
    bucket = get_bucket_for_image_size(width=w, height=h, resolution=size)
    dataset = PairedImageDataset(
        {"pos_folder": pos_d, "neg_folder": neg_d, "size": size}
    )
    assert len(dataset) == 1
    tensor, _, _ = dataset[0]
    # side-by-side halves, each half equals the bucket dims
    assert tensor.shape[1] == bucket["height"]
    assert tensor.shape[2] == 2 * bucket["width"]
