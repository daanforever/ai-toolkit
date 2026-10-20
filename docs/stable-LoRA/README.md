# Stable-LoRA

Before each real optimizer step, `ZImageDiffSynthTrainer` multiplies the LoRA down matrix A by `(1 - stable_lora_shrink/100)` until `rms(A) <= rms(B)` for that pair. B is not modified. The check and the write live in `extensions_built_in/diffusion_models/z_image_diffsynth/stable_lora.py` and run from `SDTrainer.hook_before_optimizer_step`.

Every plan based on this file must start with a first item that requires reading this file (`docs/stable-LoRA/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on Stable-LoRA in this repo. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections. Do not change the numbering.

Structure (top to bottom):

| Section                      | Contains                                                                                                              | Does not contain                          |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | What Stable-LoRA does in this repo, where the shrink runs, the plan rule                                              | Details covered below                     |
| `## 1. Done`                 | Implemented behavior as a contract. Subsections: `Shrinkage`, `Config`, `Trainer hook`, `Precision`                   | Plans, limitations, test steps            |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                        | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing`      | Behavior already described in `1. Done`   |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and describe it in `1. Done`.

## 1. Done

### Shrinkage

- A is `lora_down.weight` for toolkit `lora` / `locon` modules, and `lora_A[adapter].weight` for `peft`. B is `lora_up.weight` or `lora_B[adapter].weight`. Pairs come from `network.unet_loras`.
- `rms(X) = ||X||_F / sqrt(numel(X))`, with the norm computed in fp32. A pair is stable when `rms(A) <= rms(B)`.
- On a shrink step, each non-stable A is multiplied by `(1 - stable_lora_shrink/100)`. B is not written. The gradient is not scaled.
- The condition is evaluated on optimizer step 1 for the current network object, then every 10 optimizer steps (`STABILITY_CHECK_INTERVAL` in `stable_lora.py`). Between checks, pairs that are not yet stable keep shrinking.
- A pair that has become stable is not shrunk again in that process. The flag is not written to the checkpoint. The next process recomputes it on step 1.
- `full_rank` modules, and modules whose `lora_up` has no `weight`, are not pairs. Both increment `skipped_full_rank`. An A that does not `requires_grad` is not a pair (`skipped_frozen`). A module that is neither A/B form is ignored (`skipped_unrecognized`).
- `network_type == "peft_dora"` builds no pairs and does not write weights.
- Rebuilding happens when `id(self.network)` changes.

### Config

- Key: `network.network_kwargs.stable_lora_shrink`. Read from the trainer `network_config` first, then from `network.network_config`.
- Absent key means `1` (multiply A by `0.99`). `0` disables the feature: no pairs are built and no weights are written.
- Accepted values are real numbers in `[0, 100)`. `bool` is rejected. Anything else raises `ValueError` with text `stable_lora_shrink must be a number in [0, 100), got {value!r}`.

### Trainer hook

- `SDTrainer.hook_before_optimizer_step` is a no-op. `ZImageDiffSynthTrainer` overrides it and calls `super()` first.
- The call is inside `SDTrainer.hook_train_loop`, only when `is_grad_accumulation_step` is false, after `clip_grad_norm_`, and before `OptimizerStepMetrics.before_step` and `optimizer.step`. The metrics snapshot is therefore the post-shrink weights, and the recorded update is the optimizer delta.
- Startup log, once per network object, when the percent is not `0`: `[zimage_diffsynth] stable_lora_shrink={percent} pairs={n} check_every=10`. Nonzero skip counts are appended as `skipped_full_rank={k}`, `skipped_frozen={k}`, and `skipped_unrecognized={k}`.
- Skip logs: `[zimage_diffsynth] stable_lora_shrink={percent} skipped: peft_dora` and `[zimage_diffsynth] stable_lora_shrink={percent} skipped: no LoRA A/B pairs`.
- When a check marks new pairs stable: `[zimage_diffsynth] stable_lora stable={stable}/{total}`.
- `{percent}` is the number with trailing zeros stripped (`1`, `0.1`).

### Precision

- fp32 A: in-place `mul_` under `torch.no_grad()`.
- Any other dtype: multiply in fp32 and write back with `copy_stochastic` from `toolkit/optimizers/optimizer_utils.py`. That path requires CUDA. A non-fp32 A on CPU raises `RuntimeError` with text `stable_lora_shrink for non-fp32 weights requires CUDA (dtype={dtype}, device={device})`.

## 2. To do



## 3. Notes for continuing

### Limits

- The default `1` percent per step is larger than the paper range 0.05 to 0.5. It is on when the key is omitted, so existing Z-Image LoRA configs change. Set `0` to keep the old update.
- Because the check runs every 10 steps after the first, a pair can be shrunk up to 9 more times after `rms(A)` drops to `rms(B)`.
- Norms are computed with `torch._foreach_norm` on fp32 copies, in chunks of 8, so one check does not hold every A and B in fp32 at once. The host sync is one boolean transfer per check, not one sync per pair.
- Shrinkage compounds with optimizer `weight_decay`. Both are multiplicative on A. B is not decayed by this feature.
- `peft_dora` is out of scope. Toolkit `dora` that still exposes `lora_down` / `lora_up` is shrunk like `lora`.
- Paper notes live in `docs/stable-LoRA/Source.md`. This file is the contract.

### Known issue

- None.

### Testing

- Unit tests are `extensions_built_in/diffusion_models/z_image_diffsynth/test_stable_lora.py`. From the repo root, `venv\Scripts\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_stable_lora.py -q` reported `25 passed`. That run included the CUDA bf16 case. Without CUDA that case is skipped; fp32 shrink and the stop condition still run on CPU.
- The same command on `test_lora_param_groups.py`, `test_turbo_teacher.py`, `test_scheduler_dynamic_shifting.py`, `test_snr_weighting.py`, `test_turbo_prior.py`, and `test_diffsynth_training.py` reported `126 passed` and 2 failed. The failures are `test_calculate_loss_applies_timestep_weight_after_snr` and `test_zimage_calculate_loss_does_not_double_apply_timestep_weight`, both `AttributeError: '_Batch' object has no attribute 'get_network_weight_list'` at `SDTrainer.py:810`. That line is outside this change.
- A short train comparing `stable_lora_shrink: 1` against `0` and against `0.1`, including `peft` and `peft_dora`, was not run with the Z-Image weights.
