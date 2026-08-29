# LR vs beta2 LoRA experiments

Mid-training window protocol: one child loads DiT once (`load_model` / one `run_job`), trains a prefix at `beta2=0.99`, then sequential in-process measure forks over `[prefix, prefix+measure)`. Forks restore LoRA + optimizer from a RAM snapshot on the live trainer. Compare LoRA weight window-Δ from fp32 safetensors.

## Run

From repo root, project venv:

```bash
venv\Scripts\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.experiments
```

Needs CUDA, `temp/test_train/` (image + `.txt` caption), and model paths in [`config.yaml`](config.yaml).

Scratch: `temp/experiments/<run_id>/`. GPU forks run **sequentially** in that one child (not parallel). On success, the driver rewrites [`reports/report.md`](reports/report.md) from live metrics.

## Protocol

1. Overlay once onto `case_dir` with `steps=prefix_steps` (100) at `lr=1e-4`, `beta2=0.99` (`beta2_hi`), `save.dtype: fp32`. Slim session payload: `warm_training_folder` and `forks[].training_folder`.
2. One child: `load_model` once. Prefix train, RAM snapshot of LoRA + optimizer, save warm to `warm/probe/`.
3. Sequential 10-step measure windows `[prefix, prefix+measure)` on the same trainer (restore snapshot, `set_lr` / `set_beta2`, reseed). Saves under `fork_<id>/probe/`:
   - `continue` — `(1e-4, 0.99)`
   - `lr_x4` — `(4e-4, 0.99)`
   - `beta2_0.9` — `(1e-4, 0.9)`
   - `both` — `(4e-4, 0.9)`
   - plus extra `beta2_*` at `lr_base` from `calibrate.grid`
4. Window Δ = `W_after − W_prefix`. Hypothesis: `(1e-4, 0.9) ≈ (4e-4, 0.99)` (equivalence via ratio + cosine). Exchange rates `S_lr` / `S_b2` vs continue.

## Add a case

1. Add `cases/<id>.py` with `CASE_ID`, `resolve_prefix_steps`, `calib_fork_specs`.
2. Append an entry under `experiments.cases` in `config.yaml` with `id: <id>`.

## Tests

```bash
venv\Scripts\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/experiments/tests -q
```
