"""LoRA LR expert driver: stage A/B/C funnel over tune.lrs."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .overlay import (
    default_config_path,
    load_recipe,
    overlay_probe,
    parse_tune,
    strip_tune,
)
from .probe import child_wall_timeout_s, median_step_s, run_probe
from .rubric import health_from_tb, visual_score

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_STEP_RE = re.compile(r"__(\d{9})_")
_TARGET_LR = 1e-4
_EXPAND_CAP = 1e-2


def _toolkit_root() -> Path:
    # tune/__main__.py → z_image_diffsynth → diffusion_models → extensions_built_in → repo
    return Path(__file__).resolve().parents[4]


def _format_lr(lr: float) -> str:
    return f"{float(lr):.10f}".rstrip("0").rstrip(".")


def _lr_dir_name(lr: float) -> str:
    return f"lr_{_format_lr(lr)}"


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _pick_ref(cache: Path) -> tuple[Path, Path, str] | None:
    """Return (image, txt, caption) from test_train cache, or None if empty/missing."""
    if not cache.is_dir():
        return None
    for img in _list_images(cache):
        txt = img.with_suffix(".txt")
        if txt.is_file():
            caption = txt.read_text(encoding="utf-8").strip()
            return img, txt, caption
    return None


def _resolve_dataset_folder(recipe: dict, toolkit_root: Path) -> Path:
    raw = recipe["config"]["process"][0]["datasets"][0]["folder_path"]
    fp = Path(raw)
    if not fp.is_absolute():
        fp = toolkit_root / fp
    return fp.resolve()


def _parse_step(name: str) -> int | None:
    m = _STEP_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


def _checkpoint_sample(samples_dir: Path) -> Path | None:
    """Highest-step image under samples/ (jpg/png/webp)."""
    if not samples_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in samples_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
            continue
        step = _parse_step(p.name)
        if step is None:
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return None if best is None else best[1]


def _step0_sample(samples_dir: Path) -> Path | None:
    if not samples_dir.is_dir():
        return None
    for p in samples_dir.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in _IMAGE_EXTS
            and "__000000000_" in p.name
        ):
            return p
    return None


def _copy_master(step0: Path, dest: Path) -> None:
    from PIL import Image

    Image.open(step0).convert("RGB").save(dest)


def _is_oom(stderr: str, stdout: str) -> bool:
    blob = f"{stderr}\n{stdout}".lower()
    return "out of memory" in blob or "cuda oom" in blob


def _thresholds(tune: dict) -> dict:
    return {
        "lpips_dead": tune["lpips_dead"],
        "lpips_boom": tune["lpips_boom"],
        "clip_model": tune["clip_model"],
        "clip_pretrained": tune["clip_pretrained"],
    }


def _rank_key(lr: float, score: float, last_instability: float | None) -> tuple:
    inst = float("inf") if last_instability is None else float(last_instability)
    return (-float(score), inst, abs(float(lr) - _TARGET_LR))


def _promote(
    survivors: list[float],
    last_by_lr: dict[float, dict[str, Any]],
    k: int,
) -> list[float]:
    ranked = sorted(
        survivors,
        key=lambda lr: _rank_key(
            lr,
            last_by_lr[lr]["score"],
            last_by_lr[lr]["last_instability"],
        ),
    )
    return ranked[:k]


def _expand_lrs(tried: set[float], *, cap: float = _EXPAND_CAP) -> list[float]:
    """√10 geometric steps above max(tried), capped at cap; skip already-tried."""
    ratio = math.sqrt(10.0)
    cand = max(tried) * ratio
    out: list[float] = []
    while cand <= cap + 1e-12:
        val = cap if cand >= cap - 1e-12 else float(cand)
        if not any(math.isclose(val, t, rel_tol=1e-6, abs_tol=0.0) for t in tried):
            out.append(val)
        if val >= cap - 1e-12:
            break
        cand *= ratio
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _health_dict(health: Any | None) -> dict[str, Any] | None:
    if health is None:
        return None
    return {
        "ok": health.ok,
        "reason": health.reason,
        "last_instability": health.last_instability,
        "loss_tag": health.loss_tag,
        "loss_first_mean": health.loss_first_mean,
        "loss_last_mean": health.loss_last_mean,
        "last_lr": health.last_lr,
        "last_effective_lr": health.last_effective_lr,
    }


def _visual_dict(visual: Any | None) -> dict[str, Any] | None:
    if visual is None:
        return None
    return {
        "drop": visual.drop,
        "reason": visual.reason,
        "score": visual.score,
        "clip_i_s_r": visual.clip_i_s_r,
        "clip_i_s_m": visual.clip_i_s_m,
        "clip_i_m_r": visual.clip_i_m_r,
        "clip_t": visual.clip_t,
        "lpips_s_r": visual.lpips_s_r,
        "lpips_s_m": visual.lpips_s_m,
        "lpips_m_r": visual.lpips_m_r,
    }


def main() -> int:
    toolkit_root = _toolkit_root()
    recipe = load_recipe(default_config_path())
    tune = parse_tune(recipe["config"]["process"][0])

    run_id = str(int(time.time() * 1000))
    run_dir = toolkit_root / "temp" / "tune" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    survivors: list[float] = [float(x) for x in tune["lrs"]]
    thr = _thresholds(tune)
    safe_range = int(tune["safe_range"])
    recipe_prompt = recipe["config"]["process"][0]["sample"]["samples"][0]["prompt"]

    ref_image: Path | None = None
    ref_caption: str | None = None

    def _flush_report() -> None:
        _write_json(run_dir / "report.json", {"run_id": run_id, "trials": report_rows})

    for stage_id in tune["stages"]:
        stage = tune[stage_id]
        checkpoints = list(stage["checkpoints"])
        warmup_steps = int(stage["warmup_steps"])

        dataset_images: list[Path] = []
        if stage_id != "a":
            ds_folder = _resolve_dataset_folder(recipe, toolkit_root)
            dataset_images = _list_images(ds_folder)
            if not dataset_images:
                report_rows.append(
                    {
                        "stage": stage_id,
                        "lr": None,
                        "checkpoint": None,
                        "health": None,
                        "visual": None,
                        "paths": {"dataset_folder": str(ds_folder)},
                        "score": None,
                        "drop_reasons": ["dataset_missing_or_empty"],
                    }
                )
                _flush_report()
                return 1

        if stage_id == "a":
            picked = _pick_ref(toolkit_root / "temp" / "test_train")
            if picked is None:
                report_rows.append(
                    {
                        "stage": "a",
                        "lr": None,
                        "checkpoint": None,
                        "health": None,
                        "visual": None,
                        "paths": {
                            "test_train": str(toolkit_root / "temp" / "test_train"),
                        },
                        "score": None,
                        "drop_reasons": ["test_train_missing_or_empty"],
                    }
                )
                _flush_report()
                return 1
            ref_image, _ref_txt, ref_caption = picked

        tried_lrs: set[float] = set(float(x) for x in survivors)
        stage_lrs: list[float] = list(survivors)
        expand_used = False
        next_survivors: list[float] = []
        last_by_lr: dict[float, dict[str, Any]] = {}

        while True:
            next_survivors = []
            last_by_lr = {}
            saw_actionable_dead = False
            saw_other_drop = False

            for lr in stage_lrs:
                training_folder = run_dir / f"stage_{stage_id}" / _lr_dir_name(lr)
                training_folder.mkdir(parents=True, exist_ok=True)
                master_path = training_folder / "master.png"
                dropped = False
                last_score: float | None = None
                last_instability: float | None = None

                for ckpt_i, steps in enumerate(checkpoints):
                    is_first_segment = ckpt_i == 0
                    drop_reasons: list[str] = []
                    health = None
                    visual = None
                    sample_path: Path | None = None
                    reference_path: Path | None = None

                    if stage_id == "a" and is_first_segment:
                        ref_dir = training_folder / "ref"
                        ref_dir.mkdir(parents=True, exist_ok=True)
                        assert ref_image is not None and ref_caption is not None
                        dest_img = ref_dir / ref_image.name
                        dest_txt = ref_dir / (ref_image.stem + ".txt")
                        shutil.copy2(ref_image, dest_img)
                        dest_txt.write_text(ref_caption + "\n", encoding="utf-8")
                        reference_path = dest_img

                    config = overlay_probe(
                        recipe,
                        lr=lr,
                        steps=steps,
                        stage_id=stage_id,
                        training_folder=str(training_folder),
                        is_first_segment=is_first_segment,
                    )

                    if stage_id == "a":
                        assert ref_image is not None and ref_caption is not None
                        config["config"]["process"][0]["sample"]["samples"][0][
                            "prompt"
                        ] = ref_caption
                        reference_path = training_folder / "ref" / ref_image.name

                    job = strip_tune(config)
                    print(
                        f"tune: probe start stage={stage_id} lr={_format_lr(lr)} "
                        f"ckpt={steps}",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    prev_steps = 0 if is_first_segment else int(checkpoints[ckpt_i - 1])
                    n_new = int(steps) - prev_steps
                    wall_limit_s = child_wall_timeout_s(
                        tune["load_budget_s"],
                        n_new,
                        tune["step_timeout_s"],
                        tune["sample_budget_s"],
                    )
                    probe = run_probe(
                        job,
                        python_exe=Path(sys.executable),
                        timeout_s=wall_limit_s,
                        step_timeout_s=tune["step_timeout_s"],
                    )
                    wall_s = time.perf_counter() - t0
                    step_s = median_step_s(probe.stdout)
                    step_s_txt = f"{step_s:.3f}" if step_s is not None else "n/a"
                    print(
                        f"tune: probe done stage={stage_id} lr={_format_lr(lr)} "
                        f"ckpt={steps} wall_s={wall_s:.1f} median_step_s={step_s_txt}",
                        flush=True,
                    )
                    save_root = Path(probe.save_root)
                    log_dir = Path(probe.log_dir)
                    samples_dir = save_root / "samples"

                    row_paths: dict[str, Any] = {
                        "training_folder": str(training_folder),
                        "save_root": str(save_root),
                        "log_dir": str(log_dir),
                        "sample": None,
                        "master": str(master_path) if master_path.is_file() else None,
                        "reference": str(reference_path) if reference_path else None,
                    }

                    if probe.exit_code != 0:
                        reason = (
                            "oom"
                            if _is_oom(probe.stderr, probe.stdout)
                            else "nonzero_exit"
                        )
                        drop_reasons.append(reason)
                        report_rows.append(
                            {
                                "stage": stage_id,
                                "lr": lr,
                                "checkpoint": steps,
                                "health": None,
                                "visual": None,
                                "paths": row_paths,
                                "score": None,
                                "drop_reasons": drop_reasons,
                                "exit_code": probe.exit_code,
                            }
                        )
                        dropped = True
                        saw_other_drop = True
                        break

                    health = health_from_tb(
                        log_dir,
                        warmup_steps=warmup_steps,
                        instability_max=float(tune["instability_max"]),
                    )
                    if not health.ok:
                        drop_reasons.append(health.reason or "health_fail")
                        report_rows.append(
                            {
                                "stage": stage_id,
                                "lr": lr,
                                "checkpoint": steps,
                                "health": _health_dict(health),
                                "visual": None,
                                "paths": row_paths,
                                "score": None,
                                "drop_reasons": drop_reasons,
                            }
                        )
                        dropped = True
                        saw_other_drop = True
                        break

                    sample_path = _checkpoint_sample(samples_dir)
                    if sample_path is None:
                        drop_reasons.append("no_sample")
                        report_rows.append(
                            {
                                "stage": stage_id,
                                "lr": lr,
                                "checkpoint": steps,
                                "health": _health_dict(health),
                                "visual": None,
                                "paths": row_paths,
                                "score": None,
                                "drop_reasons": drop_reasons,
                            }
                        )
                        dropped = True
                        saw_other_drop = True
                        break

                    row_paths["sample"] = str(sample_path)

                    if is_first_segment:
                        step0 = _step0_sample(samples_dir)
                        if step0 is None:
                            drop_reasons.append("no_master")
                            report_rows.append(
                                {
                                    "stage": stage_id,
                                    "lr": lr,
                                    "checkpoint": steps,
                                    "health": _health_dict(health),
                                    "visual": None,
                                    "paths": row_paths,
                                    "score": None,
                                    "drop_reasons": drop_reasons,
                                }
                            )
                            dropped = True
                            saw_other_drop = True
                            break
                        _copy_master(step0, master_path)
                        row_paths["master"] = str(master_path)

                    if stage_id == "a":
                        visual = visual_score(
                            sample=sample_path,
                            reference=reference_path,
                            master=master_path,
                            caption=ref_caption,
                            stage_id=stage_id,
                            dataset_images=[],
                            prompt=ref_caption,
                            thresholds=thr,
                        )
                    else:
                        visual = visual_score(
                            sample=sample_path,
                            reference=None,
                            master=master_path,
                            caption=None,
                            stage_id=stage_id,
                            dataset_images=dataset_images,
                            prompt=recipe_prompt,
                            thresholds=thr,
                        )

                    if visual.drop:
                        drop_reasons.append(visual.reason or "visual_gate")

                    report_rows.append(
                        {
                            "stage": stage_id,
                            "lr": lr,
                            "checkpoint": steps,
                            "health": _health_dict(health),
                            "visual": _visual_dict(visual),
                            "paths": row_paths,
                            "score": visual.score,
                            "drop_reasons": drop_reasons,
                        }
                    )

                    if visual.drop:
                        vreason = visual.reason or "visual_gate"
                        if vreason == "dead" and steps < safe_range:
                            continue
                        if vreason == "dead" and steps >= safe_range:
                            saw_actionable_dead = True
                        else:
                            saw_other_drop = True
                        dropped = True
                        break

                    last_score = float(visual.score)
                    last_instability = health.last_instability

                if not dropped and last_score is not None:
                    next_survivors.append(lr)
                    last_by_lr[lr] = {
                        "score": last_score,
                        "last_instability": last_instability,
                    }

            if next_survivors:
                break

            can_expand = (
                not expand_used
                and saw_actionable_dead
                and not saw_other_drop
            )
            if can_expand:
                new_lrs = _expand_lrs(tried_lrs)
                if new_lrs:
                    expand_used = True
                    for x in new_lrs:
                        tried_lrs.add(x)
                    stage_lrs = new_lrs
                    continue

            _flush_report()
            return 1

        if stage_id == "c":
            winner = _promote(next_survivors, last_by_lr, 1)[0]
            recommended = {"train.lr": winner}
            _write_json(run_dir / "recommended.json", recommended)
            _flush_report()
            print(f"train.lr: {_format_lr(winner)}")
            return 0

        k = int(tune["promote_top_k"][stage_id])
        survivors = _promote(next_survivors, last_by_lr, k)
        if not survivors:
            _flush_report()
            return 1

    _flush_report()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
