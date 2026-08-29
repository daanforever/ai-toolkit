"""Load, overlay, and strip experiment recipe for LoRA probe jobs."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def load_recipe(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"recipe at {path} must be a mapping")
    return data


def ema_horizon(beta2: float, time_constants: float = 1.0) -> int:
    """Steps for ~time_constants EMA horizons: ceil(tc / (1 - beta2))."""
    beta2 = float(beta2)
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
    tc = float(time_constants)
    if tc <= 0:
        raise ValueError(f"time_constants must be > 0, got {tc}")
    # Avoid 1-0.9 float noise (ceil(1/0.0999...) == 11).
    denom = 1.0 - beta2
    return int(math.ceil(tc / denom - 1e-12))


def parse_experiments(process0: dict) -> dict:
    if "experiments" not in process0:
        raise ValueError("process[0] missing required 'experiments' key")
    exp = copy.deepcopy(process0["experiments"])
    if not isinstance(exp, dict):
        raise ValueError("'experiments' must be a mapping")

    if exp.get("step_timeout_s") is None:
        exp["step_timeout_s"] = 30.0
    else:
        exp["step_timeout_s"] = float(exp["step_timeout_s"])
    if exp.get("load_budget_s") is None:
        exp["load_budget_s"] = 180.0
    else:
        exp["load_budget_s"] = float(exp["load_budget_s"])
    if exp.get("sample_budget_s") is None:
        exp["sample_budget_s"] = 0.0
    else:
        exp["sample_budget_s"] = float(exp["sample_budget_s"])
    if exp.get("training_seed") is None:
        exp["training_seed"] = 4
    else:
        exp["training_seed"] = int(exp["training_seed"])

    cases = exp.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("experiments.cases must be a non-empty list")
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"experiments.cases[{i}] must be a mapping")
        if "id" not in case:
            raise ValueError(f"experiments.cases[{i}] missing required 'id'")
        case.setdefault("enabled", True)
        if case.get("calibrate") is not None:
            cal = case["calibrate"]
            if not isinstance(cal, dict):
                raise ValueError(f"case {case['id']}: calibrate must be a mapping")
            grid = cal.get("grid")
            if not isinstance(grid, list) or len(grid) == 0:
                raise ValueError(f"case {case['id']}: calibrate.grid must be non-empty")
    return exp


def _trial_path(training_folder: Any, *parts: str) -> str:
    base = str(training_folder).rstrip("/\\")
    return "/".join((base,) + parts)


def overlay_run(
    recipe: dict,
    *,
    case: dict,
    lr: float,
    beta2: float,
    steps: int,
    training_folder: str | Path,
    is_warm: bool,
) -> dict:
    """Build a job config for a warm or resume-fork probe."""
    config = copy.deepcopy(recipe)
    process0 = config["config"]["process"][0]
    exp = process0.get("experiments") or {}

    config["config"]["name"] = "probe"

    training_folder = str(training_folder)
    process0["log_dir"] = _trial_path(training_folder, "tb")
    process0["sqlite_db_path"] = _trial_path(training_folder, "aitk.db")
    process0["training_folder"] = training_folder
    process0["training_seed"] = int(exp.get("training_seed", 4))

    train = process0["train"]
    train["lr"] = float(lr)
    train["steps"] = int(steps)
    train["disable_sampling"] = bool(case.get("disable_sampling", True))
    train["skip_first_sample"] = True
    train["force_first_sample"] = False

    op = train.setdefault("optimizer_params", {})
    op["beta2"] = float(beta2)
    op["warmup_init"] = bool(case.get("warmup_init", False))
    if "warmup_steps" in case:
        op["warmup_steps"] = int(case["warmup_steps"])
    if "weight_decay" in case:
        op["weight_decay"] = float(case["weight_decay"])
    if "stochastic_rounding" in case:
        op["stochastic_rounding"] = bool(case["stochastic_rounding"])
    if "stochastic_accumulation" in case:
        op["stochastic_accumulation"] = bool(case["stochastic_accumulation"])

    train["timestep_type"] = "turbo_prior"
    train["content_or_style"] = "balanced"
    train["turbo_teacher_weight"] = True

    save = process0["save"]
    save["dtype"] = "fp32"
    save["save_every"] = int(steps)
    save["max_step_saves_to_keep"] = 2

    sample = process0["sample"]
    sample["sample_every"] = int(steps)
    sample["sample_steps"] = 8
    sample["guidance_scale"] = 0

    linear = int(case.get("linear", 4))
    process0["network"]["linear"] = linear
    process0["network"]["linear_alpha"] = linear

    if "resolution" in case:
        process0["datasets"][0]["resolution"] = [int(case["resolution"])]

    if case.get("dataset") == "one_ref":
        process0["datasets"][0]["folder_path"] = _trial_path(training_folder, "ref")

    process0["model"]["model_kwargs"]["use_diffsynth_training_loop"] = False
    process0["performance_log_every"] = 0
    logging = process0.setdefault("logging", {})
    logging["use_ui_logger"] = False
    log_every = logging.get("log_every")
    try:
        log_every_n = int(log_every) if log_every is not None else 1
    except (TypeError, ValueError):
        log_every_n = 1
    logging["log_every"] = max(log_every_n, 10)
    train["dtype"] = "bf16"
    train["gradient_checkpointing"] = True
    process0["model"]["compile"] = False

    # Warm starts from scratch; fork resumes and should not force a first sample.
    if is_warm:
        train["skip_first_sample"] = True
        train["force_first_sample"] = False

    return config


def strip_experiments(config: dict) -> dict:
    out = copy.deepcopy(config)
    process0 = out["config"]["process"][0]
    process0.pop("experiments", None)
    return out


def write_overlay_yaml(config: dict, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return dest
