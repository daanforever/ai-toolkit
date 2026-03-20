import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from testing.fixture_paths import FIXTURE_IMAGES_DIR
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.image_utils import save_tensors, show_tensors

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

_DEFAULT_DATASET = str(FIXTURE_IMAGES_DIR)

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset_folder',
    nargs='?',
    default=_DEFAULT_DATASET,
    help=f"Image folder (default: {_DEFAULT_DATASET})",
)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=1)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument(
    '--show',
    action='store_true',
    help='Open CV windows via show_tensors (needs display); default is print stats only',
)


args = parser.parse_args()

if args.output_path is not None:
    args.output_path = os.path.abspath(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

dataset_folder = args.dataset_folder
resolution = 512
bucket_tolerance = 64
batch_size = 1


class FakeSD:
    """Minimal stub: no HF/CLIP — enough for image bucket dataloader smoke test."""

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

dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    # clip_image_path=dataset_folder,
    # square_crop=True,
    resolution=resolution,
    # caption_ext='json',
    default_caption='default',
    # clip_image_path='/mnt/Datasets2/regs/yetibear_xl_v14/random_aspect/',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
    shrink_video_to_frames=True,
    num_frames=args.num_frames,
    # poi='person',
    # shuffle_augmentations=True,
    # augmentations=[
    #     {
    #         'method': 'Posterize',
    #         'num_bits': [(0, 4), (0, 4), (0, 4)],
    #         'p': 1.0
    #     },
    #
    # ]
)

dataloader: DataLoader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size, sd=FakeSD())


# run through an epoch and check sizes
idx = 0
for epoch in range(args.epochs):
    for batch in tqdm(dataloader):
        batch: 'DataLoaderBatchDTO'
        img_batch = batch.tensor
        frames = 1
        if len(img_batch.shape) == 5:
            frames = img_batch.shape[1]
            batch_size, frames, channels, height, width = img_batch.shape
        else:
            batch_size, channels, height, width = img_batch.shape

        # img_batch = color_block_imgs(img_batch, neg1_1=True)

        # chunks = torch.chunk(img_batch, batch_size, dim=0)
        # # put them so they are size by side
        # big_img = torch.cat(chunks, dim=3)
        # big_img = big_img.squeeze(0)
        #
        # control_chunks = torch.chunk(batch.clip_image_tensor, batch_size, dim=0)
        # big_control_img = torch.cat(control_chunks, dim=3)
        # big_control_img = big_control_img.squeeze(0) * 2 - 1
        #
        #
        # # resize control image
        # big_control_img = torchvision.transforms.Resize((width, height))(big_control_img)
        #
        # big_img = torch.cat([big_img, big_control_img], dim=2)
        #
        # min_val = big_img.min()
        # max_val = big_img.max()
        #
        # big_img = (big_img / 2 + 0.5).clamp(0, 1)

        big_img = img_batch
        # big_img = big_img.clamp(-1, 1)
        if args.output_path is not None:
            if len(img_batch.shape) == 5:
                # video
                save_tensors(big_img, os.path.join(args.output_path, f'{idx}.webp'), fps=16)
            else:
                save_tensors(big_img, os.path.join(args.output_path, f'{idx}.png'))
        elif args.show:
            show_tensors(big_img)
            time.sleep(0.2)
        else:
            print(
                f"batch {idx} tensor shape={tuple(img_batch.shape)} "
                f"min={img_batch.min().item():.4f} max={img_batch.max().item():.4f}"
            )
        idx += 1
    # if not last epoch
    if epoch < args.epochs - 1:
        trigger_dataloader_setup_epoch(dataloader)

print('done')
