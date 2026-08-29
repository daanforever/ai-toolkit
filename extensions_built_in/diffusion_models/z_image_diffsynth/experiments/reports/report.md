# LR vs beta2: mid-training window experiment

**Run:** `temp/experiments/1787940071299`  
**Case:** `lr_vs_beta2`  
**Date:** 2026-08-28  
**GPU:** NVIDIA GeForce RTX 5080  
**Wall:** ~8.1 min (`exit 0`)

## Question

Over `[100, 110)` from a shared prefix at `(lr=1e-4, beta2=0.99)`:

1. Are window-Δ under `(1e-4, 0.9)` and `(4e-4, 0.99)` equivalent (ratio ≈ 1 and cosine ≥ 0.9)?
2. How do `S_lr` and `S_b2` compare vs continue?
3. Which grid β2 at `lr_base` interpolates toward `rms(Δ_lr_x4)`?

## Setup

| Knob | Value |
|------|--------|
| Prefix | 100 steps, `lr=1e-4`, `beta2=0.99` |
| Measure | `[100, 110)` sequential resume, `steps=110` |
| 2×2 | continue / lr_x4 / beta2_0.9 / both |
| LoRA rank | 4 |
| Dataset | one image (`temp/test_train`) |
| Optimizer | Adafactor, `beta1=0`, factored, WD=0, no LR warmup |
| Save dtype | fp32 |
| calibrate | `rel_tol=0.25`, `cosine_min=0.9` |

## Headline

| Question | Result |
|----------|--------|
| Equivalence `(1e-4, 0.9)` vs `(4e-4, 0.99)` | **divergent** — cosine **0.895**, `equiv_ratio` **4.52** |
| `S_lr` / `S_b2` vs continue | **4.20** / **0.929** |
| `beta_star` interpolating to `Δ_lr_x4` | **0.99** (`r_star=0.238`) |
| `diagnostics.stationary_v` | true |

`lr×4` at 0.99 is ~5× the window-Δ of `beta2=0.9` at `1e-4`. Dropping β2 from 0.99→0.9 does not trade for an LR×4 bump on this one-image window (`S_b2=0.93`).

## LR change (`1e-4 → 4e-4`, beta2 fixed at 0.99)

| Metric | Value |
|--------|--------|
| `\|Δ\|_continue` (RMS) | `1.49e-4` |
| `\|Δ\|_lr×4` (RMS) | `6.24e-4` |
| Ratio `S_lr` | **4.199** |
| Cosine vs continue | 0.843 |
| Median per-key / down / up | 4.03 / 4.31 / 4.11 |
| Keys | 480 |

Over the measure window, LR scales the accumulated LoRA Δ nearly linearly.

## beta2 change (LR fixed at `1e-4`)

Ratio = `\|Δ(β₂)\| / \|Δ(0.99)\|` (vs continue).

| beta2 | Ratio | Cosine | down | up |
|------:|------:|-------:|-----:|---:|
| 0.7 | 0.985 | 0.790 | 0.977 | 0.992 |
| 0.8 | 0.941 | 0.934 | 0.957 | 0.928 |
| 0.85 | 0.948 | 0.826 | 0.935 | 0.958 |
| 0.88 | 0.974 | 0.836 | 0.982 | 0.968 |
| 0.9 | 0.929 | 0.866 | 0.929 | 0.929 |
| 0.92 | 0.904 | 0.924 | 0.910 | 0.900 |
| 0.95 | 0.927 | 0.946 | 0.935 | 0.920 |

Closest grid rms to `Δ_lr_x4` is **0.99** (`r_star=0.238`), cosine vs lr_x4 **0.843**. No β2 at `lr_base` approaches the LR×4 magnitude (`r` cluster 0.22–0.24).

## 2×2 and excess

| Cell | RMS | vs continue cosine | vs continue ratio |
|------|-----|-------------------:|------------------:|
| continue `(1e-4, 0.99)` | `1.49e-4` | — | 1 |
| lr_x4 `(4e-4, 0.99)` | `6.24e-4` | 0.843 | 4.20 |
| beta2_0.9 `(1e-4, 0.9)` | `1.38e-4` | 0.866 | 0.929 |
| both `(4e-4, 0.9)` | `5.64e-4` | 0.923 | 3.80 |

- both vs lr_x4: ratio **0.905**, cosine **0.812** — adding β2=0.9 on top of LR×4 barely moves the measure-window Δ.
- both vs beta2_0.9: ratio **4.09**, cosine **0.837**.
- excess `Δ_lr_x4 − Δ_continue` vs `Δ_β=0.9 − Δ_continue`: rms `5.05e-4` / `7.50e-5`, cosine **0.09**. The extra motion from the two knobs is not the same direction.

## Interpretation

Adafactor (no momentum): \(\Delta = \mathrm{lr}\cdot\mathrm{clip}(g/\sqrt{v})\). After 100 prefix steps at β2=0.99 on one image, \(v\) is mixed. On the next 10 turbo_prior steps:

- **LR** still scales the window \(\|\Delta\|\) (~×4.2).
- **beta2** only retunes how fast \(v\) tracks those gradients. On this steady one-ref regime that does not produce an LR×4-sized window; all β2 forks stay near continue (`stationary_v` diagnostic).
- Visual “10 steps at 1e-4/0.9 look like 10 steps at 4e-4/0.99” is **not** a LoRA-tensor match: ratio ~4.52, cosine 0.89 (status **divergent**).

## Artifacts

| Path | Content |
|------|---------|
| `temp/experiments/1787940071299/report.json` | Full run |
| `…/lr_vs_beta2/summary.json` | Case metrics |
| `…/lr_vs_beta2/calibrate.json` | Equivalence + rates |
| `…/lr_vs_beta2/warm/` | Prefix checkpoint |
| `…/lr_vs_beta2/fork_*/` | Per-variant resumes |

Re-run:

```bash
venv\Scripts\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.experiments
```
