"""Golden Adafactor outputs and optimizer host-sync profile.

Capture before changing the parameter loop, then compare after:

    venv\\Scripts\\python.exe testing/adafactor_host_sync_baseline.py capture
    venv\\Scripts\\python.exe testing/adafactor_host_sync_baseline.py compare
    venv\\Scripts\\python.exe testing/adafactor_host_sync_baseline.py profile
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from optimum.quanto import QBytesTensor

from toolkit.optimizers.adafactor import Adafactor
from toolkit.optimizers.optimizer_metrics import OptimizerStepMetrics

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "testing" / "fixtures" / "adafactor_host_sync_golden.pt"
PROFILE = ROOT / "testing" / "fixtures" / "adafactor_host_sync_profile.json"

SCALAR_KEYS = (
    "RMS",
    "grad_rms",
    "effective_lr",
    "effective_wd",
    "precond_gain",
    "momentum_gain",
    "beta2t",
    "dir_consistency",
)
GETTER_NAMES = (
    "get_mean_effective_lr",
    "get_mean_effective_wd",
    "get_mean_precond_gain",
    "get_mean_momentum_gain",
    "get_mean_beta2",
    "get_mean_learning_rate",
    "get_mean_dir_consistency",
)
STEP_GETTERS = (
    "get_mean_rms",
    "get_min_rms",
    "get_mean_grad_rms",
    "get_mean_grad_rms_max",
    "get_mean_update_rms",
    "get_mean_update_rms_max",
    "get_mean_dynamic_gain",
    "get_mean_step_efficiency",
)


def _devices():
    found = [torch.device("cpu")]
    if torch.cuda.is_available():
        found.append(torch.device("cuda"))
    return found


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _reference_tensor(param):
    if isinstance(param, QBytesTensor):
        return param.dequantize().detach()
    return param.detach()


def _weight_tensor(param):
    return _reference_tensor(param).cpu()


def _snapshot(opt: Adafactor, metrics: OptimizerStepMetrics):
    params = []
    groups = []
    for group in opt.param_groups:
        group_params = []
        for param in group["params"]:
            state = opt.state.get(param, {})
            scalars = {key: _as_float(state.get(key)) if key in state else None for key in SCALAR_KEYS}
            group_params.append({"weight": _weight_tensor(param), "scalars": scalars})
        params.append(group_params)
        groups.append(
            {
                "rms_max": _as_float(group["rms_max"]) if "rms_max" in group else None,
                "rms_min": _as_float(group["rms_min"]) if "rms_min" in group else None,
            }
        )
    getters = {name: float(getattr(opt, name)()) for name in GETTER_NAMES}
    getters["get_effective_lr"] = [float(v) for v in opt.get_effective_lr()]
    getters["get_max_rms"] = {
        str(key): float(value) for key, value in metrics.get_max_rms().items()
    }
    step = {name: float(getattr(metrics, name)()) for name in STEP_GETTERS}
    return {"params": params, "groups": groups, "getters": getters, "step": step}


def _make_opt(params, **kwargs):
    defaults = dict(
        lr=1e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        beta1=None,
        weight_decay=0.05,
        weight_decay_mode="absolute",
        factored=False,
        stochastic_accumulation=False,
        stochastic_rounding=False,
        eps=(1e-30, 1e-3),
    )
    defaults.update(kwargs)
    return Adafactor(params, **defaults)


def _run_case(case):
    device = case["device"]
    torch.manual_seed(case["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(case["seed"])
    params = []
    for shape, dtype in case["shapes"]:
        tensor = torch.randn(*shape, device=device, dtype=dtype)
        if case.get("nonfinite_param"):
            tensor = tensor.clone()
            tensor.view(-1)[0] = float("nan")
        params.append(torch.nn.Parameter(tensor))
    if case.get("quantized"):
        from optimum.quanto import qint8, quantize_weight

        weight = params[0].detach().float().cpu()
        scale = (weight.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
        params = [quantize_weight(weight, qint8, axis=-1, scale=scale, optimized=False)]
        params[0].requires_grad_(True)
    opt = _make_opt(params, **case["opt"])
    metrics = OptimizerStepMetrics()
    for step_index in range(3):
        for param in opt.param_groups[0]["params"]:
            like = _reference_tensor(param)
            grad = torch.empty(like.shape, device=like.device, dtype=like.dtype)
            grad.normal_()
            if case.get("nan_grad") and step_index == 0:
                grad = grad.clone()
                grad.view(-1)[0] = float("nan")
            param.grad = grad
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics.before_step(opt)
        opt.step()
        metrics.after_step(opt)
        opt.zero_grad(set_to_none=True)
    for param in opt.param_groups[0]["params"]:
        state = opt.state[param]
        for key in ("RMS", "effective_lr", "effective_wd", "precond_gain", "momentum_gain"):
            if key in state:
                assert isinstance(state[key], float), key
                assert math.isfinite(state[key]), key
    return _snapshot(opt, metrics)


def _cases():
    found = []
    seed = 1000
    for device in _devices():
        for scale_parameter, lr in ((False, 1e-3), (True, 1.0), (True, 1e-8)):
            for weight_decay_mode in ("absolute", "constant", "param_rms", "update_rms"):
                for factored in (False, True):
                    for beta1 in (0.0, 0.9):
                        for dtype in (torch.float32, torch.bfloat16):
                            found.append(
                                {
                                    "name": (
                                        f"{device.type}-scale{int(scale_parameter)}-lr{lr}"
                                        f"-{weight_decay_mode}-fact{int(factored)}"
                                        f"-b1{beta1}-{dtype}"
                                    ),
                                    "device": device,
                                    "seed": seed,
                                    "shapes": [((8, 16), dtype), ((4,), dtype)],
                                    "opt": dict(
                                        lr=lr,
                                        scale_parameter=scale_parameter,
                                        weight_decay_mode=weight_decay_mode,
                                        factored=factored,
                                        beta1=beta1,
                                        weight_decay=0.05,
                                    ),
                                }
                            )
                            seed += 1
        found.append(
            {
                "name": f"{device.type}-nan-grad",
                "device": device,
                "seed": seed,
                "shapes": [((8,), torch.float32)],
                "nan_grad": True,
                "opt": dict(beta1=0.9, factored=True, weight_decay=0.01, weight_decay_mode="absolute"),
            }
        )
        seed += 1
        found.append(
            {
                "name": f"{device.type}-nonfinite-param",
                "device": device,
                "seed": seed,
                "shapes": [((8,), torch.float32)],
                "nonfinite_param": True,
                "opt": dict(
                    beta1=None,
                    factored=False,
                    scale_parameter=True,
                    lr=1e-3,
                    weight_decay=0.05,
                    weight_decay_mode="param_rms",
                ),
            }
        )
        seed += 1
    try:
        import optimum.quanto  # noqa: F401
    except ImportError:
        pass
    else:
        found.append(
            {
                "name": "cpu-quantized",
                "device": torch.device("cpu"),
                "seed": seed,
                "shapes": [((2, 4), torch.float32)],
                "quantized": True,
                "opt": dict(lr=1.0, beta1=None, factored=False, weight_decay=0.0),
            }
        )
    return found


def _case_record(case, snapshot):
    stored = dict(case)
    stored["device"] = str(case["device"])
    stored["shapes"] = [(list(shape), str(dtype)) for shape, dtype in case["shapes"]]
    stored["snapshot"] = snapshot
    return stored


def capture():
    records = []
    for case in _cases():
        snapshot = _run_case(case)
        records.append(_case_record(case, snapshot))
        print(f"captured {case['name']}")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, FIXTURE)
    print(f"wrote {FIXTURE} ({len(records)} cases)")


def _binding_cap(case) -> bool:
    opt = case["opt"]
    return bool(opt.get("scale_parameter")) and float(opt.get("lr", 1e-3)) == 1.0


def _weight_tol(case):
    if _binding_cap(case):
        return 1e-5, 1e-6
    return 0.0, 0.0


def _state_close(case, key, got, exp):
    if key in ("effective_lr", "precond_gain", "momentum_gain"):
        return math.isclose(got, exp, rel_tol=0.0, abs_tol=0.0)
    if _binding_cap(case) and key in ("RMS", "effective_wd", "rms_max", "rms_min"):
        return math.isclose(got, exp, rel_tol=1e-5, abs_tol=1e-6)
    # param_rms stores the device clamp. The previous value was the float64
    # clamp before that factor was rounded into the parameter dtype.
    if key == "effective_wd" and case["opt"].get("weight_decay_mode") == "param_rms":
        return math.isclose(got, exp, rel_tol=1e-6, abs_tol=1e-8)
    return math.isclose(got, exp, rel_tol=0.0, abs_tol=0.0)


def _getter_tol(name: str):
    # Group means of Python floats are reduced in float64. The previous
    # reduction ran in the parameter dtype, so bf16 metrics move by about one ulp.
    if "bfloat16" in name:
        return 2e-2, 1e-4
    return 1e-6, 1e-8


def compare():
    golden = torch.load(FIXTURE, weights_only=False)
    by_name = {record["name"]: record for record in golden}
    failures = []
    for case in _cases():
        if case["name"] not in by_name:
            failures.append(f"{case['name']}: missing golden")
            continue
        current = _run_case(case)
        expected = by_name[case["name"]]["snapshot"]
        rtol, atol = _weight_tol(case)
        grtol, gatol = _getter_tol(case["name"])
        for group_index, (got_group, exp_group) in enumerate(zip(current["params"], expected["params"])):
            for param_index, (got, exp) in enumerate(zip(got_group, exp_group)):
                if not torch.allclose(got["weight"], exp["weight"], rtol=rtol, atol=atol, equal_nan=True):
                    diff = (got["weight"] - exp["weight"]).abs().max().item()
                    failures.append(
                        f"{case['name']} weight[{group_index}][{param_index}] max_abs={diff}"
                    )
                for key, exp_value in exp["scalars"].items():
                    got_value = got["scalars"][key]
                    if exp_value is None or got_value is None:
                        if exp_value != got_value:
                            failures.append(
                                f"{case['name']} {key} presence {got_value} vs {exp_value}"
                            )
                        continue
                    if not _state_close(case, key, got_value, exp_value):
                        failures.append(
                            f"{case['name']} state {key} {got_value} vs {exp_value}"
                        )
        for group_index, (got, exp) in enumerate(zip(current["groups"], expected["groups"])):
            for key in ("rms_max", "rms_min"):
                if got[key] is None or exp[key] is None:
                    if got[key] != exp[key]:
                        failures.append(f"{case['name']} group {key} missing mismatch")
                    continue
                if not _state_close(case, key, got[key], exp[key]):
                    failures.append(f"{case['name']} group {key} {got[key]} vs {exp[key]}")
        for key, exp_value in expected["getters"].items():
            got_value = current["getters"][key]
            if isinstance(exp_value, dict):
                if got_value.keys() != exp_value.keys():
                    failures.append(f"{case['name']} getter {key} keys")
                    continue
                pairs = [(got_value[name], exp_value[name]) for name in exp_value]
            elif isinstance(exp_value, list):
                if len(got_value) != len(exp_value):
                    failures.append(f"{case['name']} getter {key} length")
                    continue
                pairs = list(zip(got_value, exp_value))
            else:
                pairs = [(got_value, exp_value)]
            gain_getter = key in (
                "get_mean_effective_lr",
                "get_effective_lr",
                "get_mean_precond_gain",
                "get_mean_momentum_gain",
            )
            for got_item, exp_item in pairs:
                if gain_getter or not _binding_cap(case):
                    item_rtol, item_atol = grtol, gatol
                else:
                    item_rtol, item_atol = max(grtol, 1e-5), max(gatol, 1e-6)
                if isinstance(got_item, float) and isinstance(exp_item, float) and math.isnan(got_item) and math.isnan(exp_item):
                    continue
                if not math.isclose(float(got_item), float(exp_item), rel_tol=item_rtol, abs_tol=item_atol):
                    failures.append(
                        f"{case['name']} getter {key} {got_item} vs {exp_item}"
                    )
        for key, exp_value in expected["step"].items():
            got_value = current["step"][key]
            if isinstance(got_value, float) and isinstance(exp_value, float) and math.isnan(got_value) and math.isnan(exp_value):
                continue
            step_rtol, step_atol = (1e-5, 1e-6) if _binding_cap(case) else (0.0, 0.0)
            if not math.isclose(got_value, exp_value, rel_tol=step_rtol, abs_tol=step_atol):
                failures.append(f"{case['name']} step {key} {got_value} vs {exp_value}")
        print(f"compared {case['name']}")
    if failures:
        print(f"{len(failures)} mismatches")
        for line in failures[:40]:
            print(line)
        return 1
    print(f"matched {len(by_name)} cases")
    return 0


def _count_items(fn):
    item_n = 0
    original = torch.Tensor.item

    def _item(self):
        nonlocal item_n
        item_n += 1
        return original(self)

    torch.Tensor.item = _item
    try:
        fn()
    finally:
        torch.Tensor.item = original
    return item_n


def _profile_once(device: torch.device, n_params: int):
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    params = [
        torch.nn.Parameter(torch.randn(32, 32, device=device))
        for _ in range(n_params)
    ]
    opt = _make_opt(
        params,
        lr=1e-4,
        beta1=0.0,
        factored=True,
        scale_parameter=False,
        weight_decay=0.01,
        weight_decay_mode="absolute",
        warmup_init=False,
    )
    metrics = OptimizerStepMetrics(rms_max_decay_rate=0.99)

    def assign_grads():
        for param in params:
            param.grad = torch.randn_like(param)

    def one_step():
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics.before_step(opt)
        opt.step()
        metrics.after_step(opt)
        # Touch the log_every getters once, as the training loop does.
        opt.get_mean_effective_lr()
        opt.get_mean_effective_wd()
        opt.get_mean_precond_gain()
        opt.get_mean_momentum_gain()
        metrics.get_mean_rms()
        metrics.get_mean_grad_rms()
        metrics.get_mean_update_rms()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

    assign_grads()
    one_step()
    durations = []
    for _ in range(3):
        assign_grads()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        metrics.before_step(opt)
        opt.step()
        metrics.after_step(opt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        opt.zero_grad(set_to_none=True)

    assign_grads()
    item_calls = _count_items(one_step)

    def per_parameter_items(count):
        torch.manual_seed(1)
        local_params = [
            torch.nn.Parameter(torch.randn(16, 16, device=device))
            for _ in range(count)
        ]
        local_opt = _make_opt(
            local_params,
            lr=1e-4,
            beta1=0.0,
            factored=True,
            scale_parameter=False,
            weight_decay=0.01,
            weight_decay_mode="absolute",
        )
        local_metrics = OptimizerStepMetrics()
        for param in local_params:
            param.grad = torch.randn_like(param)
        local_metrics.before_step(local_opt)
        local_opt.step()
        local_metrics.after_step(local_opt)

        def measured():
            for param in local_params:
                param.grad = torch.randn_like(param)
            local_metrics.before_step(local_opt)
            local_opt.step()
            local_metrics.after_step(local_opt)

        return _count_items(measured)

    one_items = per_parameter_items(1)
    four_items = per_parameter_items(4)
    return {
        "device": str(device),
        "n_params": n_params,
        "step_seconds": durations,
        "mean_step_seconds": sum(durations) / len(durations),
        "python_item_calls_including_getters": item_calls,
        "per_parameter_item_calls": (four_items - one_items) / 3,
    }


def profile():
    reports = []
    for device in _devices():
        reports.append(_profile_once(device, 64 if device.type == "cuda" else 32))
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if PROFILE.exists():
        existing = json.loads(PROFILE.read_text(encoding="utf-8"))
    label = "after" if existing.get("before") else "before"
    existing[label] = reports
    PROFILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"wrote {label} profile to {PROFILE}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "capture"
    if command == "capture":
        capture()
    elif command == "compare":
        sys.exit(compare())
    elif command == "profile":
        profile()
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
