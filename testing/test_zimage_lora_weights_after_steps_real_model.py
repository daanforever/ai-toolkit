"""
Real zimage_diffsynth training: LoRA A/B stay bf16 and contain no exact zeros after 5 steps.

Concern: stray exact zeros in A/B during bf16 training (flush / stuck). Catch via abs().min(),
not abs().max() (max only proves some elements moved).

- step 0: A abs().min() > 0 (kaiming survived bf16); B all-zero (abs().min() == 0)
- after 5 steps: A and B both abs().min() > 0 (no exact-zero elements left)

Reuses the short real trainer job from test_flowmatch_snr_real_model.

Run (repo root, venv, CUDA + local Z-Image weights + temp/test_train images):
  venv\\Scripts\\python.exe -m pytest testing/test_zimage_lora_weights_after_steps_real_model.py -v -s
"""

from __future__ import annotations

import gc
import importlib
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.conftest import _skip_unless_real_zimage_stack  # noqa: E402
from testing.test_flowmatch_snr_real_model import (  # noqa: E402
    _resolve_model_paths,
    _user_flowmatch_job_config,
)
from toolkit.job import run_job  # noqa: E402

TRAIN_STEPS = 5


@dataclass
class LoraWeightSnapshot:
    downs: List[torch.Tensor]
    ups: List[torch.Tensor]
    dtypes: List[Tuple[torch.dtype, torch.dtype]]


@dataclass
class LoraWeightProbeState:
    before: Optional[LoraWeightSnapshot] = None
    after: Optional[LoraWeightSnapshot] = None
    optimizer_steps: int = 0
    module_count: int = 0
    errors: List[str] = field(default_factory=list)


def _lora_modules(network):
    modules = getattr(network, "unet_loras", None) or []
    return [m for m in modules if hasattr(m, "lora_down") and hasattr(m, "lora_up")]


def _snapshot_lora(network) -> LoraWeightSnapshot:
    downs: List[torch.Tensor] = []
    ups: List[torch.Tensor] = []
    dtypes: List[Tuple[torch.dtype, torch.dtype]] = []
    for mod in _lora_modules(network):
        down_w = mod.lora_down.weight
        up_w = mod.lora_up.weight
        dtypes.append((down_w.dtype, up_w.dtype))
        downs.append(down_w.detach().float().cpu().clone())
        ups.append(up_w.detach().float().cpu().clone())
    return LoraWeightSnapshot(downs=downs, ups=ups, dtypes=dtypes)


def _abs_min(tensors: List[torch.Tensor]) -> float:
    return min(float(t.abs().min()) for t in tensors)


def _exact_zero_count(tensors: List[torch.Tensor]) -> int:
    return int(sum(int((t == 0).sum().item()) for t in tensors))


def _install_lora_weight_probe(
    monkeypatch: pytest.MonkeyPatch,
    state: LoraWeightProbeState,
) -> None:
    sdtrainer_module = importlib.import_module("extensions_built_in.sd_trainer.SDTrainer")
    orig_hook = sdtrainer_module.SDTrainer.hook_train_loop

    def _wrapped_hook_train_loop(self, batch):
        network = getattr(self, "network", None) or getattr(getattr(self, "sd", None), "network", None)
        if network is None:
            state.errors.append(f"network missing at step_num={getattr(self, 'step_num', None)}")
            return orig_hook(self, batch)

        if state.before is None:
            snap = _snapshot_lora(network)
            state.before = snap
            state.module_count = len(snap.downs)
            if state.module_count == 0:
                state.errors.append("create LoRA for U-Net: 0 modules (target_lora_modules mismatch?)")

        out = orig_hook(self, batch)

        if not getattr(self, "is_grad_accumulation_step", False):
            state.optimizer_steps += 1
            if state.optimizer_steps == TRAIN_STEPS:
                state.after = _snapshot_lora(network)
        return out

    monkeypatch.setattr(sdtrainer_module.SDTrainer, "hook_train_loop", _wrapped_hook_train_loop)


def _drop_accelerate_global_singleton() -> None:
    try:
        import toolkit.accelerator as acc
    except Exception:
        return
    acc.global_accelerator = None


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_five_step_job(work_root: Path) -> LoraWeightProbeState:
    model_path, sampling_path = _resolve_model_paths()
    dataset_dir = REPO_ROOT / "temp" / "test_train"
    config = _user_flowmatch_job_config(
        work_root,
        dataset_dir,
        model_path,
        sampling_path,
        min_snr_gamma=7.0,
        steps=TRAIN_STEPS,
        batch_size=1,
        debug=False,
    )
    proc = config["config"]["process"][0]
    proc["network"]["linear"] = 8
    proc["network"]["linear_alpha"] = 8
    proc["network"]["rank_dropout"] = 0.0
    config["config"]["name"] = "test_zimage_lora_weights_after_steps"

    state = LoraWeightProbeState()
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_lora_weight_probe(monkeypatch, state)
        torch.manual_seed(42)
        run_job(config)
    finally:
        monkeypatch.undo()
        _drop_accelerate_global_singleton()
        _release_cuda()
    return state


def test_lora_ab_bf16_no_exact_zeros_after_five_steps():
    _skip_unless_real_zimage_stack()

    work_root = (
        REPO_ROOT
        / "temp"
        / "zimage_lora_weights_after_steps"
        / f"pytest_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    work_root.mkdir(parents=True, exist_ok=True)

    state = _run_five_step_job(work_root)

    assert not state.errors, "; ".join(state.errors)
    assert state.before is not None, "did not capture pre-step LoRA snapshot"
    assert state.after is not None, (
        f"did not capture post-step LoRA snapshot (optimizer_steps={state.optimizer_steps})"
    )
    assert state.optimizer_steps == TRAIN_STEPS
    assert state.module_count > 0
    assert len(state.before.downs) == len(state.after.downs) == state.module_count

    for i, (down_dtype, up_dtype) in enumerate(state.before.dtypes):
        assert down_dtype == torch.bfloat16, f"module {i} lora_down dtype {down_dtype}"
        assert up_dtype == torch.bfloat16, f"module {i} lora_up dtype {up_dtype}"
    for i, (down_dtype, up_dtype) in enumerate(state.after.dtypes):
        assert down_dtype == torch.bfloat16, f"post module {i} lora_down dtype {down_dtype}"
        assert up_dtype == torch.bfloat16, f"post module {i} lora_up dtype {up_dtype}"

    # Exact zeros are the failure mode (bf16 flush / stuck). Probe abs().min(), not max.
    a0_min = _abs_min(state.before.downs)
    b0_min = _abs_min(state.before.ups)
    a5_min = _abs_min(state.after.downs)
    b5_min = _abs_min(state.after.ups)
    a0_zeros = _exact_zero_count(state.before.downs)
    b0_zeros = _exact_zero_count(state.before.ups)
    a5_zeros = _exact_zero_count(state.after.downs)
    b5_zeros = _exact_zero_count(state.after.ups)

    assert a0_min > 0.0, (
        f"lora_down (A) has exact zeros at step 0 after bf16 cast "
        f"(abs_min={a0_min}, zeros={a0_zeros})"
    )
    assert b0_min == 0.0 and b0_zeros > 0, (
        "lora_up (B) should still be all-zero at step 0"
    )

    assert a5_min > 0.0, (
        f"lora_down (A) still has exact zeros after {TRAIN_STEPS} steps "
        f"(abs_min={a5_min}, zeros={a5_zeros})"
    )
    assert b5_min > 0.0, (
        f"lora_up (B) still has exact zeros after {TRAIN_STEPS} steps "
        f"(abs_min={b5_min}, zeros={b5_zeros})"
    )

    print(
        f"\n[lora weights] modules={state.module_count} "
        f"A0_min={a0_min:.6g} zeros={a0_zeros} "
        f"B0_min={b0_min:.6g} zeros={b0_zeros} "
        f"A5_min={a5_min:.6g} zeros={a5_zeros} "
        f"B5_min={b5_min:.6g} zeros={b5_zeros}",
        flush=True,
    )
