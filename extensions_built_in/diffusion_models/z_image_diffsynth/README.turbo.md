# Training LoRA for Z-Image-Turbo

This note is for users who want a LoRA that **behaves well at official Turbo inference**: 8 denoising steps, static time shift 3, no classifier-free guidance (CFG).

Train-time `t` comes from the official Turbo grid (`timestep_type: turbo_prior`). Boolean `train.turbo_teacher_weight` selects **which DiT the LoRA trains on**:

| Mode | Behavior |
| --- | --- |
| `false` (default) | Flow-match on **base Z-Image** + LoRA, turbo_prior t-grid; Turbo sampling DiT stays on **CPU** while training |
| `true` | Train LoRA on the **Turbo DiT only** (one DiT on GPU); **no** MSE teacher regularizer |

When `true`, this is **not** “train base, infer Turbo”: it is **direct LoRA on distilled Turbo**. Runtime toggle `false→true` parks base on CPU and pins Turbo on CUDA; `true→false` reverses that via `apply_turbo_teacher_mode`.

Use process type `z_image_diffsynth_trainer` and `model.arch: zimage_diffsynth`. Adapter flags and letterbox / compile behaviour are documented in `README.md`. Example YAML: `config/examples/train_lora_zimage_diffsynth_turbo_prior.yaml`.

## Checkpoints

| Role | Checkpoint | Config key |
| --- | --- | --- |
| Base transformer | Tongyi-MAI **Z-Image** | `model.name_or_path` |
| Sample / Turbo transformer | Tongyi-MAI **Z-Image-Turbo** | `model.sampling_name_or_path` |

Both paths should be local snapshot directories (or equivalent Hugging Face IDs your machine can resolve). Sampling during training uses the Turbo transformer when `sampling_name_or_path` is set. With `turbo_teacher_weight: true`, train forwards also use that Turbo DiT.

## Raise contract

With `timestep_type: turbo_prior`, the trainer / sampler **raises** (does not silently coerce) on:

| Setting | Forbidden value | Required |
| --- | --- | --- |
| `model.model_kwargs.use_diffsynth_training_loop` | `true` | `false` |
| `model.model_kwargs.use_diffsynth_prompt_encoding` | explicit `false` | `true` (or omit → defaulted on) |
| `model.model_kwargs.use_dynamic_shifting` | `true` | `false` |
| `train.content_or_style` | `gaussian` or `gaussian_bimodal` | `balanced`, `content`, or `style` (any non-gaussian mode) |
| `train.turbo_slot_weighting` | any value other than `dsigma` | omit the key (always dsigma) |

When `train.turbo_teacher_weight` is `true`, the trainer also **raises** if:

| Condition | Required |
| --- | --- |
| `train.timestep_type` ≠ `turbo_prior` | `turbo_prior` |
| `model.model_kwargs.use_diffsynth_training_loop` is `true` | `false` |
| Sampling DiT missing (`model.sampling_name_or_path` unset / unloaded) | Z-Image-Turbo sampling transformer loaded |

There is **no A/B `turbo_slot_weighting` dropdown**. Omit the key (or set `dsigma` only). Slot multinomial uses dsigma; `content_or_style: content` **reverses** those weights (first-heavy); `style` keeps dsigma (last-heavy); `balanced` reflects dsigma onto the slot nearest to `t=750` (slot 4 on 8-step Turbo).

## Required settings

These must be set this way or the job is not a Turbo-t prior run.

| Location | Parameter | Value | Why |
| --- | --- | --- | --- |
| `train` | `noise_scheduler` | `flowmatch` | Z-Image family is flow matching. |
| `train` | `prediction_type` | `flowmatch` | Match the scheduler / loss. |
| `train` | `timestep_type` | `turbo_prior` | Sample `t` from the official 8-NFE Turbo grid, not a dense 1000-step linear/shift/gaussian schedule. |
| `train` | `turbo_teacher_weight` | `false` / `true` | Boolean train-on-Turbo switch (default `false`). See modes table above. |
| `model` | `arch` | `zimage_diffsynth` | This adapter. |
| `model.model_kwargs` | `use_diffsynth_training_loop` | `false` | Turbo-t prior goes through the toolkit timestep sampler. `true` raises. |
| `model.model_kwargs` | `use_diffsynth_prompt_encoding` | `true` | Explicit `false` raises; omit defaults to `true`. |
| `model.model_kwargs` | `use_dynamic_shifting` | `false` | Official Turbo inference is **static** shift 3. `true` raises. |
| `sample` | `sample_steps` | `8` | Same NFE as Turbo. Do not judge the LoRA at 10 or 30 steps. |
| `sample` | `guidance_scale` | `0` | Official Turbo does not use CFG. |

## Important settings

Strongly recommended so training `t` and preview sampling stay on the Turbo trajectory.

| Location | Parameter | Recommended value | Notes |
| --- | --- | --- | --- |
| `train` | `content_or_style` | `balanced` | `gaussian` / `gaussian_bimodal` **raise**. Under `turbo_prior` this is **slot** bias (not a dense 1000-step cubic): `balanced` = dsigma reflected onto nearest `t=750` (slot 4 on 8-step, ~30% peak); `style` = dsigma last-heavy (`t≈300` ~30%); `content` = reversed dsigma first-heavy (`t≈1000` ~30%, composition). |
| `train` | `timestep_weighting` | `none` | Do not re-weight a dense 1000-step SNR/gaussian schedule; `t` is already on eight Turbo slots. |
| `train` | `min_snr_gamma` | `0` | Disable min-SNR reweighting for this prior. |
| `train` | `turbo_prior_steps` | `8` | Number of Turbo slots (default 8). Keep in lockstep with `sample.sample_steps`. |
| `train` | `turbo_t_jitter` | `0.5` | Start of Voronoi jitter anneal (default 0.5). `0` pins exact centers. |
| `train` | `turbo_t_jitter_end` | `0` | End of jitter anneal over training steps (default `0`). Effective `j = lerp(start, end, step / (steps-1))`. |
| `model` | `name_or_path` | Z-Image snapshot | Base weights (train DiT when `turbo_teacher_weight: false`). |
| `model` | `sampling_name_or_path` | Z-Image-Turbo snapshot | In-training samples; train DiT when `turbo_teacher_weight: true`. |

### Train-on-Turbo residency

Exclusive DiT residency is pinned by `apply_turbo_teacher_mode`:

| `turbo_teacher_weight` | During train forward |
| --- | --- |
| `false` | Base + train LoRA on CUDA; Turbo on CPU |
| `true` | Turbo + sampling LoRA on CUDA; base on CPU |

Never co-reside both DiTs on GPU. Live Runtime checkbox flips the mode without restarting the job.

**Sim gate** (`simulate_turbo_prior`): two sequential GPU passes (`false` then `true`). Hard-fails (no GREEN) if residency is wrong for the mode, if CUDA peak ≥ 85% of device total, or if LoRA / sample PNG / t-slot coverage fails.

**Grid (static shift 3, 8 steps):** train-time centers are about

`1000, 955, 900, 833, 750, 643, 500, 300`

Jitter spreads each sample in its Voronoi cell. Anneal from `turbo_t_jitter: 0.5` → `turbo_t_jitter_end: 0` so early steps explore cells and late steps pin centers. The last slot does not jitter toward `t → 0`. After jitter, `t` is clamped to `[0, num_train_timesteps]` so slot 0 cannot exceed 1000. A mass of `t` near 120 means you are still on the old gaussian prior — check `timestep_type`.

## Recommended settings

Not required for correctness, but typical for a stable Turbo LoRA job on this adapter.

| Location | Parameter | Suggestion |
| --- | --- | --- |
| `network` | `type` | `lora` |
| `network` | `linear` / `linear_alpha` | Rank = alpha **16** for the normative recipe (8–32 also fine). |
| `network.network_kwargs.ignore_if_contains` | — | Exclude `context_refiner`, `noise_refiner`, `all_final_layer`. |
| `model` | `quantize` / `quantize_te` | `true` with `qtype` / `qtype_te` `qfloat8` to cut VRAM. |
| `model` | `dtype` | `bf16` for the transformer; LoRA weights often `fp32`. |
| `model.model_kwargs` | `loader` | `auto`, `diffusers`, or `diffsynth` — see `README.md`. |
| `train` | `batch_size` | `1` avoids letterbox; `> 1` enables pad-to-square for this arch. |
| `train` | `steps` | ~2000–4000 for a real run. |
| `train` | `gradient_checkpointing` | `true` |
| `train` | `train_unet` / `train_text_encoder` | Train DiT LoRA only (`true` / `false`). |
| `train` | `cache_text_embeddings` | `true` so the text encoder can be unloaded after caching. |
| `train` | `optimizer` | `adafactor` (toolkit) or `hfadafactor` (stock Hugging Face). |
| `save` | `save_format` | `safetensors` |
| `sample` | width / height | Match dataset resolution (normative: 1024). |

Do **not** expand this recipe to assistant LoRA, trajectory imitation, or full Decoupled DMD / DMDR.

## What not to copy from a dense SFT job

| Avoid | Instead |
| --- | --- |
| `timestep_type: linear` or `shift` plus `content_or_style: gaussian` | `timestep_type: turbo_prior` + `content_or_style: balanced` (ODE default). `content` is the composition experiment, not a replacement of that default. |
| `use_diffsynth_training_loop: true` | `false` (required; `true` raises) |
| `use_diffsynth_prompt_encoding: false` | `true` (explicit `false` raises) |
| `use_dynamic_shifting: true` | `false` (`true` raises) |
| `turbo_slot_weighting: uniform` (or any non-dsigma) | Omit the key (always dsigma) |
| `sample_steps` 9, 20, or 30 as the quality check | `8` |
| `guidance_scale` 1 or higher at sample time | `0` |
| Float `turbo_teacher_weight` (e.g. `0.25`) | Boolean `false` / `true` only |

## After training

Load the saved LoRA on **Z-Image-Turbo** and generate at **8 steps, CFG 0**, static shift 3 (the default Turbo sampler in this adapter when dynamic shifting is off). Quality at 20–30 steps is not the target metric for this recipe. With `turbo_teacher_weight: true`, the LoRA was fit directly on Turbo weights — expect behavior closer to that distilled backbone than to a base-trained adapter transferred at sample time.
