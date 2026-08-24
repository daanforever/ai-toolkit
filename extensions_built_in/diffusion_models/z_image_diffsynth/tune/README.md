# LoRA LR tuner (`tune`)

## Task

Pick hyperparameters that make a Z-Image DiffSynth **Turbo LoRA** train well. **Stage 1 is learning rate only** (`train.lr`). Later stages may add other real-job keys; the output stays a **sparse dict**, never a full YAML clone of the recipe.

The harness is the same idea as `simulate_turbo_prior.py` (venv re-exec, one `run_job` per process). The recipe is a **copy** of parent `../config.yaml` in this folder — do not edit the parent file. Toolkit ignores unknown YAML keys; sweep knobs live in process-level `tune:` and are **stripped** before `run_job`. No CLI for `--config` / `--lrs` / `--stages` / `--steps-*`.

**How it searches:** short probe jobs on a log-grid of LRs, not an in-run LR-range test and not “rules on old TensorBoard only”. Three fidelity stages (cheap → mid → production). Inside a stage, **checkpoints** (e.g. 10, 100, 1000): train to a threshold, score, **resume** the same `save_root` to the next. New folder when rank/res/dataset change (no resume across stages). Scratch is `{repo}/temp/tune/`, never the OS temp dir.

**How it scores:** TensorBoard **health first** (crash, NaN, exploding loss, instability). Then visual: stage A trains on **one image + one caption** (reference **R**); a **step-0 sample** is **master M** (untrained LoRA ≈ Turbo prior); later samples **S** are compared to both. Metrics: **OpenCLIP + LPIPS**. Too close to M with no gain vs R → LR too small (**dead**). Far from M without approaching R → **exploded**. Rank survivors by CLIP/LPIPS; promote top-k to the next stage. B/C use the recipe dataset; M is still used for dead/explode gates.

Stage-1 expert that recommends **`train.lr`**. Knobs live in process-level `tune:` inside [`config.yaml`](config.yaml). `tune:` is stripped before `run_job`.

## Current status (2026-08-23)

| Piece | State |
| --- | --- |
| Package | Implemented: `overlay.py`, `probe.py`, `rubric.py`, `__main__.py`, `config.yaml` |
| Unit tests | **20 passed** (CPU, no `run_job`) |
| Live GPU trial | **Failed with exit 1** — no `train.lr` recommendation |
| Stage A probes | All 5 LRs trained to checkpoint **10**; health OK |
| Visual gates | All 5 dropped as **`dead`** (sample ≈ master; no move toward R) |
| Stages B/C | Not reached |
| `./datasets/` | **Missing** — B/C would exit 1 even if A had survivors |

First live start died earlier: `HF_HUB_OFFLINE=1` and CLIP weights `laion2b_s34b_b79k` were not in the HF cache. Retry with hub online loaded LPIPS + CLIP and completed stage A.

Artifacts from that run:

- `temp/tune/1787501481046/report.json`
- per-LR folders `temp/tune/1787501481046/stage_a/lr_<value>/`
- no `recommended.json`

Typical dead-gate numbers at 10 steps: `lpips_s_m` ≈ 0.008–0.015 (threshold 0.04), `clip_i_s_r` ≈ `clip_i_m_r`.

That is **spec-correct** scoring (S≈M at 10 steps). Driver policy: **dead** before `tune.safe_range` (default 100) is recorded but does not drop; **dead** at `checkpoint >= safe_range` drops the LR (too small). All-actionable-dead in a stage triggers one √10 expand (cap `1e-2`). Exploded / health / OOM still drop immediately.

`parse_tune` **does not** require `warmup_steps < first checkpoint`. Default B (`warmup_steps: 30`, first ckpt `10`) and C (`100` / `100`) are valid. Empty post-warmup TB series skips the 8× loss-ratio check (not `no_tb`).

## Layout

```
tune/
  config.yaml     # copy of parent recipe + tune:
  overlay.py      # load, parse_tune, overlay_probe, strip_tune
  probe.py        # one run_job in a fresh subprocess
  rubric.py       # health_from_tb + visual_score
  __main__.py     # A→B→C funnel
  __init__.py     # re-exports overlay public API
  tests/          # CPU pytest
```

Scratch is always `{repo}/temp/tune/<run_id>/`, never `tempfile.gettempdir()`.

## Funnel

1. **A** — all `tune.lrs`. One image+caption from `{repo}/temp/test_train/` copied to `{trial}/ref/`. Rank/`linear` 4, res 512, sample 256². Checkpoints 10, 100. New `training_folder` per LR.
2. Keep **top 3** by last-checkpoint visual score (tie: lower last `train/instability_score`, then closer to `1e-4`).
3. **B** — recipe `datasets[0].folder_path`, rank 128, sample 512². Checkpoints 10, 100, 1000. **New** `save_root`.
4. Keep **top 2**.
5. **C** — recipe dataset, rank 128, sample size from recipe (1024×768). Checkpoints 100, 1000. Winner → stdout + `recommended.json`.

Each checkpoint is a subprocess `run_job` with `train.steps = N`. Resume **within** a stage/LR reuses the same `training_folder` / `config.name: probe` so `save_root = {training_folder}/probe` (safetensors + `optimizer.pt`). New stage → new folder (no resume across stages).

Drop LR (skip later checkpoints) if: non-zero child exit / OOM, no TB events, no sample image, health fail, visual **exploded**, or visual **dead** at `checkpoint >= tune.safe_range` (default 100). Early **dead** (`steps < safe_range`) is recorded but does **not** drop or break — same `training_folder` continues to the next checkpoint. If a stage ends with zero survivors and every drop was actionable **dead** (not health/OOM/exploded): **one** √10 expand above `max(tried LRs)` (cap `1e-2`; e.g. max `1e-3` → `~3e-3`, `1e-2`). Empty expand or second all-dead → `report.json`, **exit 1**.

Every probe overlays Turbo-t: `train.timestep_type: turbo_prior`, `sample.sample_steps: 8`, `sample.guidance_scale: 0`, `train.content_or_style: balanced`, `model.model_kwargs.use_diffsynth_training_loop: false`.

## How to run

From repo root, project venv:

```bash
# Windows
venv\Scripts\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.tune

# Unbuffered log
set PYTHONUNBUFFERED=1
venv\Scripts\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.tune
```

No `--config` / `--lrs` / `--stages` / `--steps-*`. Change [`config.yaml`](config.yaml) (or the copied `tune:` block) instead.

`python -m …z_image_diffsynth.tune` loads parent `z_image_diffsynth/__init__.py` (DiT/trainer). Probe **children** run `probe.py` by file path so they do not double-import via `-m`. Parent still pays the package import.

### Tests

```bash
venv\Scripts\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/tune/tests -q
```

Do not import `testing/conftest.py`. Tests load modules by file path to avoid the parent DiT import.

### Success output

Stdout:

```
train.lr: 0.0003
```

Files:

```
{repo}/temp/tune/<run_id>/recommended.json   # sparse: {"train.lr": <float>}
{repo}/temp/tune/<run_id>/report.json        # every trial: stage, lr, ckpt, health, CLIP/LPIPS, paths
{repo}/temp/tune/<run_id>/stage_<id>/lr_<value>/   # training_folder
  tb/           # TensorBoard (overlaid log_dir)
  aitk.db
  ref/          # stage A only
  master.png    # step-0 sample, once per (stage, lr)
  probe/        # toolkit save_root
    samples/    # {ms}__{step:09d}_{i}.jpg
```

## Prerequisites

| Need | Where | Notes |
| --- | --- | --- |
| CUDA GPU | `device: cuda` in recipe | Live trial used RTX 5080 |
| Venv | `venv/Scripts/python.exe` | `open_clip_torch`, `lpips` already in `requirements.txt` |
| Stage A cache | `{repo}/temp/test_train/` | Image + matching `.txt` (e.g. `000.png` / `000.txt`). Fail if missing/empty. |
| Stages B/C data | `{repo}/datasets/` (recipe `folder_path: "./datasets/"`) | **Must exist and contain images** before B. Currently missing. |
| Base / Turbo weights | `model.name_or_path` / `sampling_name_or_path` in `config.yaml` | Local snapshot paths in this tree |
| CLIP | open_clip `ViT-B-32` / `laion2b_s34b_b79k` | First `visual_score` download. **`HF_HUB_OFFLINE` must be unset** unless weights are already cached under `HF_HOME`. |
| LPIPS | `lpips` alex | Loaded from site-packages; first run may still hit the hub |

CLIP Hugging Face id is roughly `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`. If the machine is offline, pre-seed that repo in the HF hub cache, then keep `HF_HUB_OFFLINE` if you want.

## Visual / health rules (as implemented)

**Health fail:** no TB events/tags; NaN/Inf in `loss/*` (or bare `loss`), `train/grad_rms`, `train/update_rms`; post-warmup last-20% loss mean > 8× first-20%; last `train/instability_score` > `tune.instability_max` (default 1.0). Missing instability tag is not a fail. Empty post-warmup series: skip ratio.

**Dead:** `LPIPS(S,M) < lpips_dead` (0.04) **and** `CLIP-I(S,R) ≤ CLIP-I(M,R) + 0.01`.  
**Exploded:** `LPIPS(S,M) > lpips_boom` (0.45) **and** `CLIP-I(S,R) < CLIP-I(M,R)`.  
B/C: R is mean CLIP embedding of dataset images; LPIPS vs M only.

Stage A score (survivors):  
`0.45 CLIP-I(S,R) + 0.20 CLIP-T(S,caption) + 0.25 max(0, CLIP-I(S,R)-CLIP-I(M,R)) + 0.10 max(0, LPIPS(M,R)-LPIPS(S,R))`  
B/C: `0.6 CLIP-I + 0.4 CLIP-T`.

## Knobs (`tune:` in `config.yaml`)

```yaml
tune:
  lrs: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  stages: [a, b, c]
  promote_top_k: {a: 3, b: 2}   # no key c
  clip_model: ViT-B-32
  clip_pretrained: laion2b_s34b_b79k
  instability_max: 1.0
  lpips_dead: 0.04
  safe_range: 100          # dead actionable at checkpoint >= this (default 100)
  lpips_boom: 0.45
  a: { checkpoints: [10, 100], warmup_steps: 8, linear: 4, ... }
  b: { checkpoints: [10, 100, 1000], warmup_steps: 30, linear: 128, ... }
  c: { checkpoints: [100, 1000], warmup_steps: 100, linear: 128 }
```

`checkpoints` must be strictly increasing. `safe_range` is coerced to `int` in `parse_tune` (default 100 if omitted). Edit this file to change the sweep; do not add CLI flags.

Parent trainer recipe fields (model paths, optimizer, Turbo schedule) come from the copied YAML. Do not edit the parent [`../config.yaml`](../config.yaml) for tuner-only changes.

## Next actions

1. **Dataset for B/C** — put train images + captions in `{repo}/datasets/` (or change `datasets[0].folder_path` in `tune/config.yaml` to an existing folder). Without this, a successful A still dies at B.
2. **CLIP cache / hub** — either leave hub online for the first `visual_score`, or cache `laion2b_s34b_b79k` under `HF_HOME`. Do not run with `HF_HUB_OFFLINE=1` until that cache exists.
3. **Dead / `safe_range` (implemented)** — `dead` before `safe_range` does not drop; at `checkpoint >= safe_range`, dead means LR too small (drop). If a stage is all such dead: one √10 expand up to `1e-2`, then exit 1 if still all dead. Do not remove checkpoint 10 from YAML. `visual_score` formula unchanged.
4. **Re-run** the module after (1)+(2); expect hours of GPU (5×A + expand + B/C probes, up to 1000 steps).
5. **Optional:** `python -m` loads parent DiT in the driver process. Harmless for VRAM if probes are subprocesses; do not “fix” by editing `z_image_diffsynth/__init__.py` unless a dedicated WP says so.

## Public API (tests / imports)

Overlay: `default_config_path`, `load_recipe`, `parse_tune`, `overlay_probe`, `strip_tune`, `write_overlay_yaml`.  
Probe: `run_probe(config, *, python_exe=None) -> ProbeResult`.  
Rubric: `health_from_tb(log_dir, *, warmup_steps, instability_max)`, `visual_score(...)`.  
Driver: `main() -> int`.

Prefer file-path import of these modules if you must not load the parent package.
