# Turbo-LoRA

MVP for `z_image_diffsynth`. The imitation pass runs after each concept backward and before `hook_before_optimizer_step`.

Every plan based on this file must start with a first item that requires reading this file (`docs/turbo-LoRA/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on Turbo-LoRA in this repo. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections. Do not change the numbering.

Structure (top to bottom):

| Section                      | Contains                                                                                                              | Does not contain                          |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | What the Turbo-LoRA MVP is, where the second pass runs, the plan rule                                                 | Details covered below                     |
| `## Contract`                | MVP behavior to implement. Subsections: `Passes`, `Optimizer`, `Config`, `Trainer hook`, `Compile`                     | Shipped behavior, task list, test steps   |
| `## 1. Done`                 | Behavior that is already in the tree. Empty until a `Contract` subsection is moved here                               | The unimplemented spec                    |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                        | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing`      | Behavior already described in `Contract` or `1. Done` |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and move the matching behavior from `Contract` into `1. Done`. When `Contract` is empty, delete the "Not implemented." sentence in the intro.

## Contract

## 1. Done

### Passes

- The concept pass is `train_single_accumulation`: existing flow-matching backward on the base DiT and the base LoRA, with `_train_on_turbo` false. This MVP leaves that loss as it is.
- The imitation pass runs after that function returns and before `hook_before_optimizer_step`.
- One `t` per microbatch, one index uniform over the `train.turbo_prior_steps` centers from `get_turbo_sigmas_and_timesteps(num_inference_steps=turbo_prior_steps, use_dynamic_shifting=False)`, independent of `content_or_style`. `t` is that center repeated to `(B,)` in scheduler units `0..1000`. This pass applies no jitter. For `turbo_prior_steps=8` the centers are `1000, 954.545, 900, 833.333, 750, 642.857, 500, 300`.
- `noise` is `torch.randn_like(batch.latents)`. `x_t = sd.add_noise(batch.latents, noise, t)`. `batch.prompt_embeds` is placed on the `latents` device before the forwards. The four forwards share that `x_t`, that `t`, and that `batch.prompt_embeds`.
- `base_off`: base network inactive, `torch.no_grad`, `get_noise_prediction`. `base_on`: `with self.network` under `torch.no_grad`, `get_noise_prediction`. Both keep `_train_on_turbo` false, so the sampling DiT stays on CPU.
- `turbo_off`: set `_train_on_turbo` true, sampling network inactive, `torch.no_grad`, `get_noise_prediction`. `turbo_on`: the flag stays true; `with sd._sampling_network` covers `get_noise_prediction` and `accelerator.backward`. The base network stays inactive, so the base DiT stays on CPU.
- `target = (turbo_off + base_on - base_off).detach()`. `loss = turbo_lora_weight * mse(turbo_on, target)`. Reduction is mean. Backward is `accelerator.backward(loss * microbatch_scale)` with the same `microbatch_scale` as that concept call. Only `turbo_on` is in the autograd graph.
- Rank dropout and module dropout stay off for these four forwards. Their previous values are restored after the pass.
- `get_noise_prediction` selects the DiT from `_train_on_turbo` and runs `_enable_dit_gradient_checkpointing` on that DiT.
- After `turbo_on`, before `hook_before_optimizer_step`: `_train_on_turbo` false, `_park_sampling_dit_and_lora`, `_flush_cuda` only when that park moved weights, `_move_main_network` to the train device. The same restore runs if the pass raises, and the exception propagates.
- LoRA tensors are not moved to CPU. `apply_turbo_teacher_mode` is not used. `trainer.network` remains the base network object. Both networks are inactive again after the pass.
- The imitation pass feeds conditional embeds only.

### Optimizer

- The residual gradient is added to the concept gradient. One `Adafactor.step` consumes the sum.
- `zero_grad` stays at the start of `hook_train_loop`. Nothing clears gradients between the concept backward and the `turbo_on` backward.
- fp32 LoRA sums those two backwards in `.grad`.
- Non-fp32 LoRA: the post-accumulate hook `stochastic_grad_accummulation` adds each backward into `_accum_grad` and deletes `.grad`. `Adafactor.step_hook` copies `_accum_grad` back to `.grad` inside `step()`. This pass calls `step_hook` only from that `step()`.
- `clip_grad_norm_` stays skipped when `train.optimizer` is `adafactor`. There is no clip between the two backwards. `clip_threshold` caps RMS of the preconditioned update. `turbo_lora_weight` sets the mix of the two gradients.
- One `step()` per real optimizer step. A second `step()` would advance the `decay_rate` step index, warmup, and weight decay, and would shrink stable-LoRA twice on the same batch.
- The weight write is `Adafactor.step`. Moments and stochastic rounding describe that write.
- `hook_before_optimizer_step` still runs once, after the `turbo_on` backward, on the base network object.

### Config

- `train.turbo_lora`: bool, default false when the key is absent or the value is `None`. Any other non-bool raises `ValueError` with text `turbo_lora must be boolean true/false; got {value!r}`.
- `train.turbo_lora_weight`: finite float `> 0`, default `1.0` when the key is absent. Read and checked only when `turbo_lora` is true. `bool` is rejected. Otherwise `ValueError` with text `turbo_lora_weight must be a finite number > 0, got {value!r}`.
- When `turbo_lora` is false the weight is ignored and the checks below do not run.
- When `turbo_lora` is true, `ZImageDiffSynthTrainer.__init__` after `super().__init__` raises `ValueError` if any of these hold:
  - `train.optimizer`, compared case-insensitively, is not `adafactor`. Text: `turbo_lora=true requires train.optimizer='adafactor' (got {opt!r}).`
  - `model.sampling_name_or_path` is missing. Text: `turbo_lora=true requires model.sampling_name_or_path (Z-Image-Turbo / _sampling_transformer).`
  - `train.turbo_teacher_weight` is true. Text: `turbo_lora=true is incompatible with train.turbo_teacher_weight=true.`
  - `model.compile` is true. Text: `turbo_lora=true is incompatible with model.compile=true.`
  - `train.cache_text_embeddings` is false. Text: `turbo_lora=true requires train.cache_text_embeddings=true.`
- A true flag with `sd._sampling_transformer is None` raises at the first imitation pass. Text: `turbo_lora=true requires a loaded sampling transformer (set model.sampling_name_or_path to Z-Image-Turbo).`
- A true flag with `sd._sampling_network is None` raises at that pass. Text: `turbo_lora=true requires a loaded sampling network (shared LoRA on the sampling DiT).`
- A true flag with `batch.prompt_embeds is None` raises at that pass. Text: `turbo_lora=true requires cached prompt embeds on the batch.`

### Trainer hook

- `ZImageDiffSynthTrainer.hook_train_loop` overrides `SDTrainer.hook_train_loop` and calls `super()`. `SDTrainer.hook_train_loop` calls `_after_train_single_accumulation` after each `train_single_accumulation` return. That hook runs the imitation pass when `turbo_lora` is true, before `hook_before_optimizer_step`. The parent keeps `zero_grad`, the microbatch loop, and the single `optimizer.step` behind `not is_grad_accumulation_step`.
- `loss_dict['loss']` is the concept loss. When the flag is true, `loss_dict['turbo_lora']` is the sample-weighted mean of the detached per-microbatch residual MSEs (`mse(turbo_on, target)` before `turbo_lora_weight`), using the same batch-size weights as `loss`. When the flag is false the key is absent.
- When the flag is true, once, before the loop: `[zimage_diffsynth] turbo_lora=true weight={weight} slots={turbo_prior_steps}`. `{weight}` has trailing zeros stripped.

### Compile

- The imitation pass contains no `torch.compile` path and does not call `compile_dit_blocks`.
- `model.compile: true` together with `turbo_lora: true` raises in `__init__` (see `Config`). The raise runs before `hook_after_sd_init_before_load` clears `model_config.compile`.

## 2. To do

## 3. Notes for continuing

### Limits

- The delayed second pass and `turbo_lora_steps` are outside this MVP.
- `compile_blocks.py` stays for jobs with `turbo_lora: false`.
- `docs/turbo-LoRA/Source.md` is the sketch. This file wins where they differ. The imitation target is the residual in `Passes`. Its `[0.85, 0.65, 0.45, 0.25, 0.15, 0.07, 0.03]` list is not an imitation `t`.
- Concept `t` stays on the job's `timestep_type`. Imitation `t` is one turbo-grid center.
- `clip_threshold` caps update RMS. It leaves the mix to `turbo_lora_weight`.
- Stable-LoRA still shrinks A on that single step. The imitation backward scales the gradient only.
- A step that leaves both DiTs on CUDA breaks the residency contract above.
- There is no UI control for `turbo_lora` in this MVP. The runtime `turbo_teacher_weight` checkbox stays on its existing path, which `turbo_lora: true` already rejects.
- The pass has no 8-step rollout, no unconditional embed, and no distribution-matching term.
- One shared LoRA state is saved. This MVP adds no second adapter and no separate Turbo state dict.

### Known issue

- None.

### Testing

- Pytest passed: `venv\Scripts\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_turbo_lora.py -q`.
- GPU train (`turbo_lora/test_gpu.py`) loads real Z-Image and Z-Image-Turbo and runs one Adafactor step with `turbo_lora: true`. It skips unless `ZIMAGE_TURBO_LORA_GPU=1`, CUDA is available, and both snapshot directories exist, so a normal pytest run does not load the models. Run: `venv\Scripts\python.exe -m extensions_built_in.diffusion_models.z_image_diffsynth.turbo_lora.test_gpu`.
- Cases: `base_off` hits the base DiT with the base network inactive and the sampling DiT on CPU; `base_on` hits the base DiT under `torch.no_grad` with the base network active and the sampling DiT on CPU; `turbo_off` hits the sampling DiT with the sampling network inactive and the base DiT on CPU; `turbo_on` hits the sampling DiT with the sampling network active, the base network inactive, and the base DiT on CPU; `_train_on_turbo` is true only for `turbo_off` and `turbo_on`; order is `base_off`, `base_on`, park base, flush, host turbo, `turbo_off`, `turbo_on` backward, park turbo, flush, host base; `target` is detached `turbo_off + base_on - base_off`; `base_on` writes no `.grad`; `turbo_on` backward writes `.grad` on the shared LoRA parameter; LoRA device is the same before and after the swaps; rank dropout and module dropout are restored; `apply_turbo_teacher_mode` is not called; `trainer.network` is still the base network at `hook_before_optimizer_step`; concept backward plus `turbo_on` backward then one `Adafactor.step` increase the step counter by 1 and both gradients are inside that step; `loss` excludes the residual MSE; `turbo_lora` is the sample-weighted mean of those MSEs; a raise still restores base residency and `_train_on_turbo` false; each `ValueError` text in `Config`.
