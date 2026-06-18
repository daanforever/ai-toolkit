"""Shared assertions for flow-match SNR weighting (smoke + pytest)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Union

import torch

from toolkit.train_tools import apply_snr_weight, get_all_snr


def assert_scheduler_uses_compute_snr_path(scheduler) -> None:
    assert hasattr(scheduler, "compute_snr"), "scheduler must define compute_snr"
    assert callable(getattr(scheduler, "compute_snr")), "scheduler.compute_snr must be callable"
    # Flow-match schedulers must not rely on DDPM alphas_cumprod for SNR lookup.
    get_all_snr(scheduler, "cpu")


def assert_all_snr_table(scheduler, device) -> torch.Tensor:
    all_snr = get_all_snr(scheduler, device)
    assert all_snr is not None and all_snr.dim() == 1, "get_all_snr must return a 1d tensor"
    assert all_snr.shape[0] == 1000, "get_all_snr must return 1000 timesteps"

    reference = scheduler.compute_snr().to(device=device, dtype=all_snr.dtype)
    assert torch.allclose(all_snr, reference), "get_all_snr must match scheduler.compute_snr()"

    for i in range(all_snr.shape[0] - 1):
        assert all_snr[i] > all_snr[i + 1], (
            f"SNR must decrease with timestep index; failed at {i}: "
            f"{all_snr[i].item()} <= {all_snr[i + 1].item()}"
        )

    assert all_snr[0].item() > 1.0, "SNR at lowest noise (t≈0) should be > 1"
    assert all_snr[-1].item() < 1e-3, "SNR at highest noise (t=1) should be near 0"
    return all_snr


def lookup_snr(
    all_snr: torch.Tensor,
    scheduler,
    timestep_values: Union[Sequence[float], torch.Tensor],
    device,
) -> torch.Tensor:
    """Mirror apply_snr_weight flow_match SNR: t = timestep / ntt, SNR = (1-t)^2 / t^2."""
    del all_snr
    timestep_tensor = torch.as_tensor(timestep_values, device=device, dtype=torch.float32)
    ntt = float(scheduler.config.num_train_timesteps)
    t = (timestep_tensor / ntt).clamp(min=1e-8, max=1.0)
    return ((1.0 - t) ** 2) / (t ** 2 + 1e-8)


def expected_flow_match_min_snr_weight(snr: Union[float, torch.Tensor], gamma: float) -> torch.Tensor:
    snr_tensor = torch.as_tensor(snr, dtype=torch.float32)
    gamma_tensor = torch.ones_like(snr_tensor) * gamma
    denom = (1.0 + torch.sqrt(snr_tensor)) ** 2
    return torch.minimum(gamma_tensor, snr_tensor) / denom


def format_snr_weight_check_lines(
    timestep_list: Sequence[float],
    snr: torch.Tensor,
    expected: torch.Tensor,
    weighted: torch.Tensor,
    gamma: float,
) -> str:
    w_ts, w_snr, w_exp, w_act = 12, 14, 16, 14
    lines = [
        f"min_snr_gamma={gamma:g}",
        (
            f"{'timestep':>{w_ts}}  {'snr':>{w_snr}}  "
            f"{'expected_weight':>{w_exp}}  {'actual_weight':>{w_act}}"
        ),
    ]
    for i, ts in enumerate(timestep_list):
        lines.append(
            f"{ts:>{w_ts}g}  {snr[i].item():>{w_snr}.6g}  "
            f"{expected[i].item():>{w_exp}.6g}  {weighted[i].item():>{w_act}.6g}"
        )
    return "\n" + "\n".join(lines)


def assert_apply_snr_flow_match_weights(
    scheduler,
    timesteps: Iterable[float],
    gamma: float,
    device,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    verbose: bool = True,
) -> torch.Tensor:
    timestep_list = [float(t) for t in timesteps]
    all_snr = get_all_snr(scheduler, device)
    loss = torch.ones(len(timestep_list), device=device, dtype=torch.float32)
    timestep_tensor = torch.tensor(timestep_list, device=device, dtype=torch.float32)
    weighted = apply_snr_weight(
        loss,
        timestep_tensor,
        scheduler,
        gamma,
        prediction_type="flow_match",
    )
    assert weighted.shape == loss.shape, "apply_snr_weight must preserve loss shape"

    snr = lookup_snr(all_snr, scheduler, timestep_list, device)
    expected = expected_flow_match_min_snr_weight(snr, gamma)
    if verbose:
        print(format_snr_weight_check_lines(timestep_list, snr, expected, weighted, gamma), flush=True)
    for i, ts in enumerate(timestep_list):
        actual = weighted[i].item()
        exp = expected[i].item()
        snr_val = snr[i].item()
        if abs(exp) > 0:
            assert abs(actual - exp) <= atol + rtol * abs(exp), (
                f"timestep {ts}: snr={snr_val}, expected_weight={exp}, actual_weight={actual}"
            )
        else:
            assert abs(actual) <= atol, (
                f"timestep {ts}: snr={snr_val}, expected_weight~0, actual_weight={actual}"
            )
    return weighted


def non_integer_schedule_timesteps(scheduler, max_count: int = 3) -> list[float]:
    """Pick non-integer timesteps from scheduler.timesteps for interpolation checks."""
    picked: list[float] = []
    for value in scheduler.timesteps.tolist():
        if not math.isfinite(value):
            continue
        if abs(value - round(value)) > 1e-6:
            picked.append(float(value))
        if len(picked) >= max_count:
            break
    return picked
