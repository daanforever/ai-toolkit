"""CPU tests: unquantized zimage_diffsynth generate never calls network.merge_in."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from toolkit.models.base_model import BaseModel


def _fake_base_generate(self, image_configs, sampler=None):
    network = getattr(self, "network", None)
    if network is not None:
        unique = {getattr(cfg, "network_multiplier", 1.0) for cfg in image_configs}
        if len(unique) == 1 and getattr(network, "can_merge_in", False):
            network.merge_in(merge_weight=unique.pop())
    return []


def _make_net(*, can_merge_in=True):
    net = SimpleNamespace(can_merge_in=can_merge_in, is_merged_in=False)
    net.merge_in = MagicMock()
    return net


def _stub_model(*, with_sampling=True, can_merge_in=True):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    model = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    model.device_torch = torch.device("cpu")
    model._train_on_turbo = False
    model.network = _make_net(can_merge_in=can_merge_in)
    if with_sampling:
        model._sampling_transformer = object()
        model._sampling_network = _make_net(can_merge_in=can_merge_in)
    else:
        model._sampling_transformer = None
        model._sampling_network = None
    return model


def _configs():
    return [SimpleNamespace(network_multiplier=1.0)]


def _generate(model):
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )

    return ZImageDiffSynthModel.generate_images(model, _configs())


def test_unquantized_generate_with_sampling_net_never_merges(monkeypatch):
    monkeypatch.setattr(BaseModel, "generate_images", _fake_base_generate)
    model = _stub_model(with_sampling=True, can_merge_in=True)
    main = model.network
    samp = model._sampling_network
    _generate(model)
    assert main.merge_in.call_count == 0
    assert samp.merge_in.call_count == 0
    assert main.can_merge_in is False
    assert samp.can_merge_in is False
    assert main.is_merged_in is False
    assert samp.is_merged_in is False


def test_unquantized_generate_without_sampling_net_never_merges(monkeypatch):
    monkeypatch.setattr(BaseModel, "generate_images", _fake_base_generate)
    model = _stub_model(with_sampling=False, can_merge_in=True)
    main = model.network
    _generate(model)
    assert main.merge_in.call_count == 0
    assert main.can_merge_in is False
    assert main.is_merged_in is False


def test_quantized_generate_still_skips_merge(monkeypatch):
    monkeypatch.setattr(BaseModel, "generate_images", _fake_base_generate)
    model = _stub_model(with_sampling=True, can_merge_in=False)
    main = model.network
    samp = model._sampling_network
    _generate(model)
    assert main.merge_in.call_count == 0
    assert samp.merge_in.call_count == 0
    assert main.can_merge_in is False
    assert samp.can_merge_in is False


def test_hook_before_train_loop_disables_merge_in(monkeypatch):
    from extensions_built_in.diffusion_models.z_image_diffsynth.trainer import (
        ZImageDiffSynthTrainer,
    )
    from extensions_built_in.diffusion_models.z_image_diffsynth.test_turbo_teacher import (
        _base_cfg,
        _patch_diffusion_trainer_init,
    )
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    _patch_diffusion_trainer_init(
        monkeypatch,
        turbo_teacher_weight=False,
        sampling_name_or_path="/tmp/turbo",
    )
    monkeypatch.setattr(SDTrainer, "hook_before_train_loop", lambda self: None)
    monkeypatch.setattr(
        ZImageDiffSynthTrainer,
        "internal_hook_before_train_loop",
        lambda self: None,
    )

    main = SimpleNamespace(can_merge_in=True)
    samp = SimpleNamespace(can_merge_in=True)
    sd = SimpleNamespace(
        _sampling_transformer=object(),
        _sampling_network=samp,
        network=main,
        gradient_checkpointing=True,
        apply_turbo_teacher_mode=lambda enabled: None,
    )
    trainer = ZImageDiffSynthTrainer(0, None, _base_cfg())
    trainer.is_ui_trainer = False
    trainer._compile_dit_blocks = False
    trainer.sd = sd
    trainer.network = main
    trainer.hook_before_train_loop()
    assert main.can_merge_in is False
    assert samp.can_merge_in is False
