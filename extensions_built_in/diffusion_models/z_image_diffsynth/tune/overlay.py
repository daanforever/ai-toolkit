"""Load, overlay, and strip tune recipe for LoRA LR probe jobs."""

from __future__ import annotations

import copy
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


def parse_tune(process0: dict) -> dict:
    if "tune" not in process0:
        raise ValueError("process[0] missing required 'tune' key")
    tune = copy.deepcopy(process0["tune"])
    if not isinstance(tune, dict):
        raise ValueError("'tune' must be a mapping")

    promote = tune.get("promote_top_k")
    if not isinstance(promote, dict):
        raise ValueError("tune.promote_top_k must be a mapping with keys for stages a and b")
    for sid in ("a", "b"):
        if sid not in promote:
            raise ValueError(f"tune.promote_top_k missing required key '{sid}'")

    if "safe_range" not in tune or tune["safe_range"] is None:
        tune["safe_range"] = 100
    else:
        tune["safe_range"] = int(tune["safe_range"])

    if tune.get("step_timeout_s") is None:
        tune["step_timeout_s"] = 2.0
    else:
        tune["step_timeout_s"] = float(tune["step_timeout_s"])
    if tune.get("load_budget_s") is None:
        tune["load_budget_s"] = 180.0
    else:
        tune["load_budget_s"] = float(tune["load_budget_s"])
    if tune.get("sample_budget_s") is None:
        tune["sample_budget_s"] = 60.0
    else:
        tune["sample_budget_s"] = float(tune["sample_budget_s"])

    stages = tune.get("stages") or []
    for sid in stages:
        stage = tune.get(sid)
        if not isinstance(stage, dict):
            continue
        if "checkpoints" not in stage:
            continue
        checkpoints = stage["checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) == 0:
            raise ValueError(f"tune.{sid}.checkpoints must be a non-empty list")
        for i in range(1, len(checkpoints)):
            if not (checkpoints[i] > checkpoints[i - 1]):
                raise ValueError(
                    f"tune.{sid}.checkpoints must be strictly increasing; got {checkpoints}"
                )
    return tune


def _trial_path(training_folder: Any, *parts: str) -> str:
    base = str(training_folder).rstrip("/\\")
    return "/".join((base,) + parts)


def overlay_probe(
    recipe: dict,
    *,
    lr,
    steps,
    stage_id,
    training_folder,
    is_first_segment: bool,
) -> dict:
    config = copy.deepcopy(recipe)
    process0 = config["config"]["process"][0]
    tune = process0["tune"]
    if stage_id not in tune or not isinstance(tune[stage_id], dict):
        raise ValueError(f"tune missing stage '{stage_id}'")
    stage = tune[stage_id]

    config["config"]["name"] = "probe"

    process0["log_dir"] = _trial_path(training_folder, "tb")
    process0["sqlite_db_path"] = _trial_path(training_folder, "aitk.db")
    process0["training_folder"] = str(training_folder)

    train = process0["train"]
    train["lr"] = lr
    train["steps"] = steps
    train["optimizer_params"]["warmup_steps"] = stage["warmup_steps"]
    if is_first_segment:
        train["skip_first_sample"] = False
        train["force_first_sample"] = True
    else:
        train["skip_first_sample"] = True
        train["force_first_sample"] = False
    train["timestep_type"] = "turbo_prior"
    train["content_or_style"] = "balanced"
    # Fixed teacher mode (not swept; stage-1 recommends train.lr only).
    w = tune.get("turbo_teacher_weight", True)
    if w is None:
        train["turbo_teacher_weight"] = True
    elif isinstance(w, bool):
        train["turbo_teacher_weight"] = w
    else:
        raise ValueError(
            "turbo_teacher_weight must be boolean true/false; "
            "legacy float weights are not supported"
        )

    save = process0["save"]
    save["save_every"] = steps
    save["max_step_saves_to_keep"] = 2

    sample = process0["sample"]
    sample["sample_every"] = steps
    if "sample_width" in stage:
        sample["width"] = stage["sample_width"]
    if "sample_height" in stage:
        sample["height"] = stage["sample_height"]
    sample["sample_steps"] = 8
    sample["guidance_scale"] = 0

    linear = stage["linear"]
    process0["network"]["linear"] = linear
    process0["network"]["linear_alpha"] = linear

    if "resolution" in stage:
        process0["datasets"][0]["resolution"] = [stage["resolution"]]

    if stage.get("dataset") == "one_ref":
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
    process0["model"]["compile"] = True

    return config


def strip_tune(config: dict) -> dict:
    out = copy.deepcopy(config)
    process0 = out["config"]["process"][0]
    process0.pop("tune", None)
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
