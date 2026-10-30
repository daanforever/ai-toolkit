"""Runtime num_repeats: per-copy file items, shuffle mixing, and rebuild without a rescan."""

import random
import sys
from pathlib import Path

import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from testing.test_network_weight_sampling import FakeSD
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import (
    get_dataloader_datasets,
    get_dataloader_from_datasets,
    rebuild_dataloader_network_weights,
)


def _write_images(folder: Path, sizes):
    folder.mkdir(parents=True, exist_ok=True)
    for name, size in sizes:
        Image.new("RGB", size, (20, 40, 60)).save(folder / name)


def _config(folder: Path, **kwargs) -> DatasetConfig:
    params = dict(
        dataset_path=str(folder),
        resolution=512,
        default_caption="default",
        buckets=True,
        bucket_tolerance=64,
        shrink_video_to_frames=True,
        num_frames=1,
        network_weight=1.0,
        num_repeats=1,
    )
    params.update(kwargs)
    return DatasetConfig(**params)


def _epoch_paths(dataloader):
    datasets = get_dataloader_datasets(dataloader)
    paths = []
    for batch in dataloader.dataset.bucket_manager.batch_indices:
        for dataset_idx, file_idx in batch:
            paths.append(datasets[dataset_idx].file_list[file_idx].path)
    return paths


def _max_run(paths):
    if not paths:
        return 0
    best = 1
    run = 1
    for previous, current in zip(paths, paths[1:]):
        if current == previous:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


@pytest.fixture
def fake_sd():
    return FakeSD()


def test_repeat_copies_are_shuffled_with_other_pairs(fake_sd, tmp_path):
    folder = tmp_path / "images"
    _write_images(folder, [(f"img_{i}.png", (640, 640)) for i in range(3)])
    random.seed(0)
    dataloader = get_dataloader_from_datasets(
        [_config(folder, num_repeats=10)],
        batch_size=2,
        sd=fake_sd,
    )
    paths = _epoch_paths(dataloader)
    assert len(paths) == 30
    assert _max_run(paths) < 10
    for name in ("img_0.png", "img_1.png", "img_2.png"):
        occurrences = [path for path in paths if path.endswith(name)]
        assert len(occurrences) == 10


def test_repeat_copies_get_independent_crops(fake_sd, monkeypatch, tmp_path):
    folder = tmp_path / "images"
    _write_images(folder, [("wide.png", (1000, 700))])
    sequence = iter(range(0, 10000))

    def randint(low, high):
        if low >= high:
            return low
        return low + (next(sequence) % (high - low + 1))

    monkeypatch.setattr("toolkit.dataloader_mixins.random.randint", randint)
    dataloader = get_dataloader_from_datasets(
        [_config(folder, num_repeats=8, random_crop=True)],
        batch_size=1,
        sd=fake_sd,
    )
    dataset = get_dataloader_datasets(dataloader)[0]
    copies = [item for item in dataset.file_list if item.path.endswith("wide.png")]
    assert len(copies) == 8
    assert len({id(item) for item in copies}) == 8
    assert len({(item.crop_x, item.crop_y) for item in copies}) > 1


def test_runtime_repeats_rebuilds_copies_without_rescan(fake_sd, monkeypatch, tmp_path):
    folder = tmp_path / "images"
    _write_images(folder, [("a.png", (512, 512)), ("b.png", (512, 512))])
    config = _config(folder, num_repeats=1, buckets=False, flip_x=True)
    dataloader = get_dataloader_from_datasets([config], batch_size=1, sd=fake_sd)
    dataset = get_dataloader_datasets(dataloader)[0]
    base_count = len(dataset.base_file_items)
    base_ids = [id(item) for item in dataset.base_file_items]
    assert base_count == 2
    assert len(dataset.file_list) == base_count * 2

    def fail_walk(*_args, **_kwargs):
        raise AssertionError("dataset folder was scanned during num_repeats rebuild")

    monkeypatch.setattr("toolkit.data_loader.os.walk", fail_walk)
    assert dataset.rebuild_for_num_repeats(3) is True
    assert len(dataset.base_file_items) == base_count
    assert [id(item) for item in dataset.base_file_items] == base_ids
    assert len(dataset.file_list) == base_count * 3 * 2
    assert sum(1 for item in dataset.file_list if item.flip_x) == base_count * 3
    assert dataset.rebuild_for_num_repeats(3) is False

    dataloader = rebuild_dataloader_network_weights(dataloader)
    assert len(list(dataloader)) == base_count * 3 * 2


def test_apply_runtime_dataset_num_repeats_updates_loader(fake_sd, tmp_path):
    folder = tmp_path / "images"
    _write_images(folder, [("a.png", (512, 512))])
    config = _config(folder, num_repeats=1, buckets=False)
    dataloader = get_dataloader_from_datasets([config], batch_size=1, sd=fake_sd)
    dataset = get_dataloader_datasets(dataloader)[0]
    cleared = []
    trainer = type("Trainer", (), {})()
    trainer.is_ui_trainer = True
    trainer.data_loader = dataloader
    trainer.data_loader_reg = None
    trainer.dataset_configs = [dataset.dataset_config]
    trainer.epoch_num = 1
    trainer._read_runtime_json_list = lambda column: [4]
    trainer._clear_runtime_columns = lambda *cols: cleared.append(cols)
    trainer.get_runtime_dataset_num_repeats = (
        lambda: DiffusionTrainer.get_runtime_dataset_num_repeats(trainer)
    )
    trainer._runtime_dataset_groups = lambda: DiffusionTrainer._runtime_dataset_groups(trainer)

    DiffusionTrainer.apply_runtime_dataset_num_repeats(trainer)

    assert len(dataset.file_list) == len(dataset.base_file_items) * 4
    assert trainer.data_loader is not dataloader
    assert cleared == [("runtime_dataset_num_repeats",)]
    assert len(list(trainer.data_loader)) == 4


def test_get_runtime_dataset_num_repeats_rejects_invalid_lists():
    trainer = type("Trainer", (), {})()

    def read(_column):
        return [1, 1.5]

    trainer._read_runtime_json_list = read
    assert DiffusionTrainer.get_runtime_dataset_num_repeats(trainer) is None

    trainer._read_runtime_json_list = lambda _column: [2, 5]
    assert DiffusionTrainer.get_runtime_dataset_num_repeats(trainer) == [2, 5]
