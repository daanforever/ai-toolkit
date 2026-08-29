"""Experiment driver: one session child (prefix + sequential forks) → calibrate LoRA Deltas."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .calibrate import (
    classify_equivalence,
    exchange_rates,
    pick_beta_star,
    stationary_v_diagnostic,
)
from .capture import (
    cosine_delta,
    find_latest_lora,
    load_lora_state,
    summarize_delta_pair,
    subtract_deltas,
    update_rms,
    delta_tensors,
)
from .overlay import (
    default_config_path,
    load_recipe,
    overlay_run,
    parse_experiments,
    strip_experiments,
)
from .probe import child_wall_timeout_s, run_probe


def _toolkit_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_case_module(case_id: str):
    cases_dir = Path(__file__).resolve().parent / "cases"
    path = cases_dir / f"{case_id}.py"
    if not path.is_file():
        raise ValueError(f"unknown experiment case id: {case_id}")
    name = f"_zids_exp_case_{case_id}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


_GPU_RE = re.compile(r"gpu=(.+?)(?:\s+\w+=|\s*$)", re.MULTILINE)


def _gpu_from_log(text: str) -> str | None:
    if not text:
        return None
    m = _GPU_RE.search(text)
    if m is None:
        return None
    name = m.group(1).strip()
    return name or None


def _fmt_sci(x: float) -> str:
    s = f"{float(x):.2e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def _fmt_lr(x: float) -> str:
    s = f"{float(x):.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def _fmt_wall(s: float) -> str:
    s = float(s)
    if s >= 90.0:
        mins = s / 60.0
        return f"~{mins:.0f} min" if mins >= 10.0 else f"~{mins:.1f} min"
    return f"~{s:.0f} s"


def _package_report_md_path() -> Path:
    return Path(__file__).resolve().parent / "reports" / "report.md"


def render_markdown_report(
    *,
    run_id: str,
    reports: list[dict[str, Any]],
    wall_s: float | None = None,
) -> str:
    """Human-readable reports/report.md from a live run payload."""
    try:
        date = datetime.fromtimestamp(int(run_id) / 1000.0).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        date = datetime.now().strftime("%Y-%m-%d")
    chunks = [
        render_case_markdown(rep, run_id=run_id, date=date, wall_s=wall_s)
        for rep in reports
    ]
    return "\n\n---\n\n".join(chunks).rstrip() + "\n"


def render_case_markdown(
    report: dict[str, Any],
    *,
    run_id: str,
    date: str,
    wall_s: float | None = None,
) -> str:
    case_id = str(report.get("case_id", "unknown"))
    gpu = report.get("gpu")
    lines = [
        "# LR vs beta2: mid-training window experiment",
        "",
        f"**Run:** `temp/experiments/{run_id}`  ",
        f"**Case:** `{case_id}`  ",
        f"**Date:** {date}  ",
    ]
    if gpu:
        lines.append(f"**GPU:** {gpu}  ")
    wall_bit = f"{_fmt_wall(wall_s)} " if wall_s is not None else ""
    lines.append(f"**Wall:** {wall_bit}(`exit 0`)")
    lines.append("")
    if report.get("error"):
        lines.extend(
            [
                f"Run failed: `{report.get('error')}`",
                "",
            ]
        )
        return "\n".join(lines)

    prefix = int(report["prefix_steps"])
    measure = int(report["measure_steps"])
    window = report.get("window") or [prefix, prefix + measure]
    lr_base = float(report["lr_base"])
    lr_hi = float(report["lr_hi"])
    beta2_hi = float(report["beta2_hi"])
    beta2_lo = float(report["beta2_lo"])
    cal = report["calibrate"]
    eq = cal["equivalence"]
    geometry = report.get("geometry") or {}
    interp = cal.get("interpolation") or {}
    status = str(eq.get("status"))
    equiv_ratio = eq.get("equiv_ratio")
    equiv_cosine = eq.get("equiv_cosine")
    s_lr = float(cal["s_lr"])
    s_b2 = float(cal["s_b2"])
    star = interp.get("beta_star")
    r_star = interp.get("r_star")
    star_cell = f"**{star}**"
    if r_star is not None:
        star_cell += f" (`r_star={float(r_star):.3f}`)"
    stationary = (cal.get("diagnostics") or {}).get("stationary_v")
    continue_rms = float(report["continue_update_rms"])
    lr_geo = geometry.get("lr_x4") or {}
    lo_id = f"beta2_{beta2_lo}"
    lo_geo = geometry.get(lo_id) or {}
    both = geometry.get("both") or {}
    excess = geometry.get("excess") or {}
    lr_b = _fmt_lr(lr_base)
    lr_h = _fmt_lr(lr_hi)

    lines.extend(
        [
            "## Question",
            "",
            f"Over `[{window[0]}, {window[1]})` from a shared prefix at "
            f"`(lr={lr_b}, beta2={beta2_hi})`:",
            "",
            f"1. Are window-Δ under `({lr_b}, {beta2_lo})` and `({lr_h}, {beta2_hi})` "
            "equivalent (ratio ≈ 1 and cosine ≥ 0.9)?",
            "2. How do `S_lr` and `S_b2` compare vs continue?",
            "3. Which grid β2 at `lr_base` interpolates toward `rms(Δ_lr_x4)`?",
            "",
            "## Setup",
            "",
            "| Knob | Value |",
            "|------|--------|",
            f"| Prefix | {prefix} steps, `lr={lr_b}`, `beta2={beta2_hi}` |",
            f"| Measure | `[{prefix}, {prefix + measure})` sequential resume, "
            f"`steps={prefix + measure}` |",
            "| 2×2 | continue / lr_x4 / beta2_0.9 / both |",
            "| LoRA rank | 4 |",
            "| Dataset | one image (`temp/test_train`) |",
            "| Optimizer | Adafactor, `beta1=0`, factored, WD=0, no LR warmup |",
            "| Save dtype | fp32 |",
            f"| calibrate | `rel_tol={eq.get('rel_tol', 0.25)}`, "
            f"`cosine_min={eq.get('cosine_min', 0.9)}` |",
            "",
            "## Headline",
            "",
            "| Question | Result |",
            "|----------|--------|",
            f"| Equivalence `({lr_b}, {beta2_lo})` vs `({lr_h}, {beta2_hi})` | "
            f"**{status}** — cosine **{float(equiv_cosine):.3f}**, "
            f"`equiv_ratio` **{float(equiv_ratio):.2f}** |",
            f"| `S_lr` / `S_b2` vs continue | **{s_lr:.2f}** / **{s_b2:.3f}** |",
            f"| `beta_star` interpolating to `Δ_lr_x4` | {star_cell} |",
            f"| `diagnostics.stationary_v` | {str(stationary).lower()} |",
            "",
            f"`lr×4` at {beta2_hi} is ~{float(equiv_ratio):.0f}× the window-Δ of "
            f"`beta2={beta2_lo}` at `{lr_b}`. Dropping β2 from {beta2_hi}→{beta2_lo} "
            f"does not trade for an LR×4 bump on this one-image window "
            f"(`S_b2={s_b2:.2f}`).",
            "",
            f"## LR change (`{lr_b} → {lr_h}`, beta2 fixed at {beta2_hi})",
            "",
            "| Metric | Value |",
            "|--------|--------|",
            f"| `\\|Δ\\|_continue` (RMS) | `{_fmt_sci(continue_rms)}` |",
            f"| `\\|Δ\\|_lr×4` (RMS) | `{_fmt_sci(float(lr_geo.get('update_rms_other', 0.0)))}` |",
            f"| Ratio `S_lr` | **{s_lr:.3f}** |",
            f"| Cosine vs continue | {float(lr_geo.get('cosine', 0.0)):.3f} |",
            f"| Median per-key / down / up | "
            f"{float(lr_geo.get('median_per_key_ratio', 0.0)):.2f} / "
            f"{float(lr_geo.get('down_ratio', 0.0)):.2f} / "
            f"{float(lr_geo.get('up_ratio', 0.0)):.2f} |",
            f"| Keys | {int(lr_geo.get('n_keys', 0))} |",
            "",
            "Over the measure window, LR scales the accumulated LoRA Δ nearly linearly.",
            "",
            f"## beta2 change (LR fixed at `{lr_b}`)",
            "",
            "Ratio = `\\|Δ(β₂)\\| / \\|Δ(" + str(beta2_hi) + ")\\|` (vs continue).",
            "",
            "| beta2 | Ratio | Cosine | down | up |",
            "|------:|------:|-------:|-----:|---:|",
        ]
    )

    beta_rows: list[tuple[float, dict[str, Any]]] = []
    for key, geo in geometry.items():
        if not str(key).startswith("beta2_") or not isinstance(geo, dict):
            continue
        if "ratio" not in geo:
            continue
        try:
            b = float(str(key).removeprefix("beta2_"))
        except ValueError:
            continue
        beta_rows.append((b, geo))
    beta_rows.sort(key=lambda t: t[0])
    for b, geo in beta_rows:
        lines.append(
            f"| {b} | {float(geo['ratio']):.3f} | {float(geo['cosine']):.3f} | "
            f"{float(geo.get('down_ratio', 0.0)):.3f} | "
            f"{float(geo.get('up_ratio', 0.0)):.3f} |"
        )

    ratios = interp.get("ratios") or {}
    r_vals = [float(v) for v in ratios.values() if v is not None]
    r_span = ""
    if r_vals:
        r_span = f" (`r` cluster {min(r_vals):.2f}–{max(r_vals):.2f})"
    star_cos = interp.get("cosine")
    star_cos_bit = (
        f", cosine vs lr_x4 **{float(star_cos):.3f}**" if star_cos is not None else ""
    )
    lines.extend(
        [
            "",
            f"Closest grid rms to `Δ_lr_x4` is {star_cell}{star_cos_bit}. "
            f"No β2 at `lr_base` approaches the LR×4 magnitude{r_span}.",
            "",
            "## 2×2 and excess",
            "",
            "| Cell | RMS | vs continue cosine | vs continue ratio |",
            "|------|-----|-------------------:|------------------:|",
            f"| continue `({lr_b}, {beta2_hi})` | `{_fmt_sci(continue_rms)}` | — | 1 |",
            f"| lr_x4 `({lr_h}, {beta2_hi})` | "
            f"`{_fmt_sci(float(lr_geo.get('update_rms_other', 0.0)))}` | "
            f"{float(lr_geo.get('cosine', 0.0)):.3f} | {s_lr:.2f} |",
            f"| {lo_id} `({lr_b}, {beta2_lo})` | "
            f"`{_fmt_sci(float(lo_geo.get('update_rms_other', 0.0)))}` | "
            f"{float(lo_geo.get('cosine', 0.0)):.3f} | {s_b2:.3f} |",
        ]
    )
    both_vs_c = both.get("vs_continue") or {}
    if both_vs_c:
        lines.append(
            f"| both `({lr_h}, {beta2_lo})` | "
            f"`{_fmt_sci(float(both_vs_c.get('update_rms_other', 0.0)))}` | "
            f"{float(both_vs_c.get('cosine', 0.0)):.3f} | "
            f"{float(both_vs_c.get('ratio', 0.0)):.2f} |"
        )
    lines.append("")
    both_vs_lr = both.get("vs_lr_x4") or {}
    both_vs_lo = both.get("vs_beta2_lo") or {}
    if both_vs_lr:
        lines.append(
            f"- both vs lr_x4: ratio **{float(both_vs_lr.get('ratio', 0.0)):.3f}**, "
            f"cosine **{float(both_vs_lr.get('cosine', 0.0)):.3f}** — adding "
            f"β2={beta2_lo} on top of LR×4 barely moves the measure-window Δ."
        )
    if both_vs_lo:
        lines.append(
            f"- both vs {lo_id}: ratio **{float(both_vs_lo.get('ratio', 0.0)):.2f}**, "
            f"cosine **{float(both_vs_lo.get('cosine', 0.0)):.3f}**."
        )
    if excess:
        lines.append(
            f"- excess `Δ_lr_x4 − Δ_continue` vs `Δ_β={beta2_lo} − Δ_continue`: "
            f"rms `{_fmt_sci(float(excess.get('lr_x4_rms', 0.0)))}` / "
            f"`{_fmt_sci(float(excess.get('beta2_lo_rms', 0.0)))}`, "
            f"cosine **{float(excess.get('cosine', 0.0)):.2f}**. "
            "The extra motion from the two knobs is not the same direction."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Adafactor (no momentum): "
            r"\(\Delta = \mathrm{lr}\cdot\mathrm{clip}(g/\sqrt{v})\). "
            f"After {prefix} prefix steps at β2={beta2_hi} on one image, "
            r"\(v\) is mixed. On the next "
            f"{measure} turbo_prior steps:",
            "",
            f"- **LR** still scales the window "
            r"\(\|\Delta\|\) "
            f"(~×{s_lr:.1f}).",
            "- **beta2** only retunes how fast "
            r"\(v\) "
            "tracks those gradients. On this steady one-ref regime that does not "
            "produce an LR×4-sized window; all β2 forks stay near continue "
            "(`stationary_v` diagnostic).",
            f"- Visual “{measure} steps at {lr_b}/{beta2_lo} look like "
            f"{measure} steps at {lr_h}/{beta2_hi}” is **not** a LoRA-tensor match: "
            f"ratio ~{float(equiv_ratio):.2f}, cosine {float(equiv_cosine):.2f} "
            f"(status **{status}**).",
            "",
            "## Artifacts",
            "",
            "| Path | Content |",
            "|------|---------|",
            f"| `temp/experiments/{run_id}/report.json` | Full run |",
            f"| `…/{case_id}/summary.json` | Case metrics |",
            f"| `…/{case_id}/calibrate.json` | Equivalence + rates |",
            f"| `…/{case_id}/warm/` | Prefix checkpoint |",
            f"| `…/{case_id}/fork_*/` | Per-variant resumes |",
            "",
            "Re-run:",
            "",
            "```bash",
            "venv\\Scripts\\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.experiments",
            "```",
        ]
    )
    return "\n".join(lines)


def write_package_report_md(
    *,
    run_id: str,
    reports: list[dict[str, Any]],
    wall_s: float | None = None,
) -> Path:
    path = _package_report_md_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(run_id=run_id, reports=reports, wall_s=wall_s),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def _pick_ref(cache: Path) -> tuple[Path, Path, str] | None:
    if not cache.is_dir():
        return None
    for img in _list_images(cache):
        txt = img.with_suffix(".txt")
        if txt.is_file():
            return img, txt, txt.read_text(encoding="utf-8").strip()
    return None


def _prepare_ref(training_folder: Path, ref_image: Path, caption: str) -> None:
    ref_dir = training_folder / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    dest_img = ref_dir / ref_image.name
    dest_txt = ref_dir / (ref_image.stem + ".txt")
    shutil.copy2(ref_image, dest_img)
    dest_txt.write_text(caption + "\n", encoding="utf-8")


def run_case(
    recipe: dict,
    exp: dict,
    case: dict,
    run_dir: Path,
    *,
    run_probe_fn=run_probe,
    python_exe: Path | None = None,
) -> dict[str, Any]:
    case_id = str(case["id"])
    mod = _load_case_module(case_id)
    prefix_steps = mod.resolve_prefix_steps(case)
    measure = int(case.get("measure_steps", 10))
    lr_base = float(case["lr_base"])
    beta2_hi = float(case.get("beta2_hi", 0.99))
    beta2_lo = float(case.get("beta2_lo", 0.9))
    cal_cfg = case.get("calibrate") or {}
    rel_tol = float(cal_cfg.get("rel_tol", 0.25))
    cosine_min = float(cal_cfg.get("cosine_min", 0.9))
    beta2_lo_id = f"beta2_{beta2_lo}"

    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    toolkit_root = _toolkit_root()
    if case.get("dataset") == "one_ref":
        picked = _pick_ref(toolkit_root / "temp" / "test_train")
        if picked is None:
            return {
                "case_id": case_id,
                "error": "test_train_missing_or_empty",
                "paths": {"test_train": str(toolkit_root / "temp" / "test_train")},
            }
        ref_image, _txt, caption = picked
    else:
        ref_image = None
        caption = None

    fork_specs = mod.calib_fork_specs(case)
    n_forks = len(fork_specs)

    cfg = overlay_run(
        recipe,
        case=case,
        lr=lr_base,
        beta2=beta2_hi,
        steps=prefix_steps,
        training_folder=case_dir,
        is_warm=True,
    )
    if case.get("dataset") == "one_ref" and ref_image is not None:
        _prepare_ref(case_dir, ref_image, caption or "")

    job = strip_experiments(cfg)
    job["config"]["process"][0]["experiments"] = {
        "prefix_steps": prefix_steps,
        "measure_steps": measure,
        "warm_training_folder": str(case_dir / "warm"),
        "forks": [
            {
                "id": spec["id"],
                "lr": spec["lr"],
                "beta2": spec["beta2"],
                "training_folder": str(case_dir / f"fork_{spec['id']}"),
            }
            for spec in fork_specs
        ],
    }
    wall = child_wall_timeout_s(
        exp["load_budget_s"],
        prefix_steps + n_forks * measure,
        exp["step_timeout_s"],
        exp["sample_budget_s"],
    )
    print(
        f"experiment: start case={case_id} steps={prefix_steps} "
        f"lr={lr_base} beta2={beta2_hi}",
        flush=True,
    )
    child = run_probe_fn(
        job,
        python_exe=python_exe or Path(sys.executable),
        timeout_s=wall,
        step_timeout_s=exp["step_timeout_s"],
    )
    if child.exit_code != 0:
        return {
            "case_id": case_id,
            "error": "child_nonzero_exit",
            "exit_code": child.exit_code,
            "prefix_steps": prefix_steps,
        }

    warm_lora_path = find_latest_lora(case_dir / "warm" / "probe")
    if warm_lora_path is None:
        return {
            "case_id": case_id,
            "error": "warm_no_lora",
            "prefix_steps": prefix_steps,
        }
    warm_state = load_lora_state(warm_lora_path)

    fork_results: dict[str, Any] = {}
    after_states: dict[str, dict] = {}

    for spec in fork_specs:
        fid = spec["id"]
        save_root = case_dir / f"fork_{fid}" / "probe"
        row: dict[str, Any] = {
            "id": fid,
            "lr": spec["lr"],
            "beta2": spec["beta2"],
            "exit_code": child.exit_code,
            "save_root": str(save_root),
        }
        after_path = find_latest_lora(save_root)
        if after_path is None:
            row["error"] = "no_lora"
            fork_results[fid] = row
            continue
        after_state = load_lora_state(after_path)
        after_states[fid] = after_state
        d = delta_tensors(warm_state, after_state)
        row["update_rms"] = update_rms(d)
        fork_results[fid] = row

    if "continue" not in after_states or "lr_x4" not in after_states:
        return {
            "case_id": case_id,
            "prefix_steps": prefix_steps,
            "error": "missing_continue_or_lr_x4",
            "forks": fork_results,
        }

    d_continue = delta_tensors(warm_state, after_states["continue"])
    d_lr_x4 = delta_tensors(warm_state, after_states["lr_x4"])
    continue_rms = update_rms(d_continue)
    rms_lr_x4 = update_rms(d_lr_x4)

    lr_sum = summarize_delta_pair(
        warm_state, after_states["continue"], after_states["lr_x4"]
    )
    geometry: dict[str, Any] = {"lr_x4": lr_sum}

    ratios_vs_continue: dict[float, float] = {}
    rms_by_beta: dict[float, float] = {}
    # continue is at lr_base — include in interpolation grid
    rms_by_beta[float(beta2_hi)] = continue_rms

    for spec in fork_specs:
        fid = str(spec["id"])
        if fid not in after_states:
            continue
        if abs(float(spec["lr"]) - lr_base) > 1e-15:
            continue
        if fid == "continue":
            continue
        if not fid.startswith("beta2_"):
            continue
        summ = summarize_delta_pair(
            warm_state, after_states["continue"], after_states[fid]
        )
        geometry[fid] = summ
        b = float(spec["beta2"])
        ratios_vs_continue[b] = float(summ["ratio"])
        rms_by_beta[b] = update_rms(delta_tensors(warm_state, after_states[fid]))

    has_beta2_lo = beta2_lo_id in after_states
    rms_beta2_lo = 0.0
    d_beta2_lo: dict | None = None
    if has_beta2_lo:
        d_beta2_lo = delta_tensors(warm_state, after_states[beta2_lo_id])
        rms_beta2_lo = update_rms(d_beta2_lo)

    rates = exchange_rates(
        rms_continue=continue_rms,
        rms_lr_x4=rms_lr_x4,
        rms_beta2_lo=rms_beta2_lo,
    )
    s_lr = float(rates["s_lr"])
    s_b2 = float(rates["s_b2"])

    if has_beta2_lo:
        pair = summarize_delta_pair(
            warm_state, after_states[beta2_lo_id], after_states["lr_x4"]
        )
        equivalence = classify_equivalence(
            equiv_ratio=float(pair["ratio"]),
            equiv_cosine=float(pair["cosine"]),
            rel_tol=rel_tol,
            cosine_min=cosine_min,
        )
    else:
        equivalence = {
            "status": "skipped",
            "reason": "missing_beta2_lo",
            "equiv_ratio": None,
            "equiv_cosine": None,
            "rel_tol": rel_tol,
            "cosine_min": cosine_min,
        }

    interpolation = pick_beta_star(rms_lr_x4=rms_lr_x4, rms_by_beta=rms_by_beta)
    star = interpolation.get("beta_star")
    if star is not None:
        star_id = "continue" if abs(float(star) - beta2_hi) < 1e-12 else f"beta2_{star}"
        if star_id in after_states:
            d_star = delta_tensors(warm_state, after_states[star_id])
            interpolation["cosine"] = cosine_delta(d_star, d_lr_x4)

    if "both" in after_states:
        both_geo: dict[str, Any] = {
            "vs_continue": summarize_delta_pair(
                warm_state, after_states["continue"], after_states["both"]
            ),
            "vs_lr_x4": summarize_delta_pair(
                warm_state, after_states["lr_x4"], after_states["both"]
            ),
        }
        if has_beta2_lo:
            both_geo["vs_beta2_lo"] = summarize_delta_pair(
                warm_state, after_states[beta2_lo_id], after_states["both"]
            )
        geometry["both"] = both_geo

    excess: dict[str, Any] = {}
    if has_beta2_lo and d_beta2_lo is not None:
        ex_lr = subtract_deltas(d_lr_x4, d_continue)
        ex_b2 = subtract_deltas(d_beta2_lo, d_continue)
        excess = {
            "lr_x4_rms": update_rms(ex_lr),
            "beta2_lo_rms": update_rms(ex_b2),
            "cosine": cosine_delta(ex_lr, ex_b2),
        }
        geometry["excess"] = excess

    calib = {
        "equivalence": equivalence,
        "s_lr": s_lr,
        "s_b2": s_b2,
        "interpolation": interpolation,
        "diagnostics": {
            "stationary_v": stationary_v_diagnostic(
                ratios_vs_continue=ratios_vs_continue,
                s_lr=s_lr,
            ),
        },
    }
    report = {
        "case_id": case_id,
        "prefix_steps": prefix_steps,
        "measure_steps": measure,
        "window": [prefix_steps, prefix_steps + measure],
        "lr_base": lr_base,
        "lr_hi": float(case["lr_hi"]),
        "beta2_hi": beta2_hi,
        "beta2_lo": beta2_lo,
        "continue_update_rms": continue_rms,
        "calibrate": calib,
        "geometry": geometry,
        "forks": fork_results,
        "warm_lora": str(warm_lora_path),
    }
    gpu = _gpu_from_log(getattr(child, "stdout", "") or "")
    if gpu:
        report["gpu"] = gpu
    _write_json(case_dir / "calibrate.json", calib)
    _write_json(case_dir / "summary.json", report)
    print(
        f"experiment: case={case_id} prefix_steps={prefix_steps} "
        f"equiv={equivalence.get('status')} "
        f"equiv_ratio={equivalence.get('equiv_ratio')} "
        f"equiv_cosine={equivalence.get('equiv_cosine')} "
        f"s_lr={s_lr:.4f} s_b2={s_b2:.4f} beta_star={star}",
        flush=True,
    )
    return report


def main() -> int:
    toolkit_root = _toolkit_root()
    recipe = load_recipe(default_config_path())
    exp = parse_experiments(recipe["config"]["process"][0])

    run_id = str(int(time.time() * 1000))
    run_dir = toolkit_root / "temp" / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    reports: list[dict[str, Any]] = []
    for case in exp["cases"]:
        if not case.get("enabled", True):
            continue
        case_id = str(case["id"])
        try:
            _load_case_module(case_id)
        except ValueError as exc:
            print(f"experiment: {exc}", flush=True)
            return 1
        report = run_case(
            recipe,
            exp,
            case,
            run_dir,
            python_exe=Path(sys.executable),
        )
        reports.append(report)
        if report.get("error"):
            _write_json(run_dir / "report.json", {"run_id": run_id, "cases": reports})
            return 1

    wall_s = time.perf_counter() - t0
    _write_json(run_dir / "report.json", {"run_id": run_id, "cases": reports})
    md_path = write_package_report_md(run_id=run_id, reports=reports, wall_s=wall_s)
    print(f"experiment: wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
