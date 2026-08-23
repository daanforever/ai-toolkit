# Training LoRA for Z-Image-Turbo

This note is for users who want a LoRA that **behaves well at official Turbo inference**: 8 denoising steps, static time shift 3, no classifier-free guidance (CFG).

Train the LoRA on **Z-Image** (dense SFT / base) weights. Sample and later use it on **Z-Image-Turbo**. The training loss stays ordinary flow-matching MSE on the base model. What changes is **which noise levels `t` you train on**.

Z-Image-Turbo only calls the transformer on **eight** shifted sigmas. If you train on the usual 1000-step dense schedule (especially `content_or_style: gaussian`, whose peak sits near `t ≈ 120`), the LoRA is barely defined on the Turbo grid. The last Turbo center is about `t ≈ 300`, so that gaussian peak is **outside** the trajectory Euler actually uses.

Use process type `z_image_diffsynth_trainer` and `model.arch: zimage_diffsynth`. Adapter flags and letterbox / compile behaviour are documented in `README.md`. This file covers the **Turbo-t prior** job only.

## Checkpoints

| Role | Checkpoint | Config key |
| --- | --- | --- |
| Train (base transformer) | Tongyi-MAI **Z-Image** | `model.name_or_path` |
| Sample / Turbo transformer | Tongyi-MAI **Z-Image-Turbo** | `model.sampling_name_or_path` |

Both paths should be local snapshot directories (or equivalent Hugging Face IDs your machine can resolve). Sampling during training uses the Turbo transformer when `sampling_name_or_path` is set.

## Required settings

These must be set this way or the job is not a Turbo-t prior run.

| Location | Parameter | Value | Why |
| --- | --- | --- | --- |
| `train` | `noise_scheduler` | `flowmatch` | Z-Image family is flow matching. |
| `train` | `prediction_type` | `flowmatch` | Match the scheduler / loss. |
| `train` | `timestep_type` | `turbo_prior` | Sample `t` from the official 8-NFE Turbo grid, not a dense 1000-step linear/shift/gaussian schedule. |
| `model` | `arch` | `zimage_diffsynth` | This adapter. |
| `model.model_kwargs` | `use_diffsynth_training_loop` | `false` | Turbo-t prior goes through the toolkit timestep sampler. If you leave the DiffSynth loop on, the trainer **warns and turns it off**. |
| `model.model_kwargs` | `use_dynamic_shifting` | `false` | Official Turbo inference is **static** shift 3. Dynamic (Flux-style) shift is for dense SFT, not Turbo. |
| `sample` | `sample_steps` | `8` | Same NFE as Turbo. Do not judge the LoRA at 10 or 30 steps. |
| `sample` | `guidance_scale` | `0` | Official Turbo does not use CFG. |

## Important settings

Strongly recommended so training `t` and preview sampling stay on the Turbo trajectory.

| Location | Parameter | Recommended value | Notes |
| --- | --- | --- | --- |
| `train` | `content_or_style` | `balanced` | With `turbo_prior`, gaussian modes are **ignored** (warning in the log). Set `balanced` so the YAML matches what actually runs. |
| `train` | `timestep_weighting` | `none` | Do not re-weight a dense 1000-step SNR/gaussian schedule; `t` is already on eight Turbo slots. |
| `train` | `min_snr_gamma` | `0` | Disable min-SNR reweighting for this prior. |
| `train` | `turbo_prior_steps` | `8` | Number of Turbo slots (default 8). Keep in lockstep with `sample.sample_steps`. |
| `train` | `turbo_t_jitter` | `0.5` | Voronoi jitter around each slot (default 0.5). `0` pins exact centers. The last slot does not jitter toward `t → 0`. |
| `model` | `name_or_path` | Z-Image snapshot | Base weights the LoRA is trained on. |
| `model` | `sampling_name_or_path` | Z-Image-Turbo snapshot | Transformer used for in-training samples. |

**Grid (static shift 3, 8 steps):** train-time centers are about

`1000, 955, 900, 833, 750, 643, 500, 300`

Jitter spreads each sample in its Voronoi cell. A small fraction of values slightly below 300 is normal at `turbo_t_jitter: 0.5`. A mass of `t` near 120 means you are still on the old gaussian prior — check `timestep_type`.

## Recommended settings

Not required for correctness, but typical for a stable Turbo LoRA job on this adapter.

| Location | Parameter | Suggestion |
| --- | --- | --- |
| `network` | `type` | `lora` |
| `network` | `linear` / `linear_alpha` | Rank 8–32 for a real run; rank 4 is only for short smoke tests. |
| `network.network_kwargs.ignore_if_contains` | — | Often exclude `context_refiner`, `noise_refiner`, `all_final_layer` (same as the adapter’s example job). |
| `model` | `quantize` / `quantize_te` | `true` with `qtype` / `qtype_te` `qfloat8` to cut VRAM. |
| `model` | `dtype` | `bf16` for the transformer; LoRA weights often `fp32`. |
| `model.model_kwargs` | `loader` | `auto`, `diffusers`, or `diffsynth` — see `README.md`. |
| `train` | `batch_size` | `1` avoids letterbox; `> 1` enables pad-to-square for this arch. |
| `train` | `gradient_checkpointing` | `true` |
| `train` | `train_unet` / `train_text_encoder` | Train DiT LoRA only (`true` / `false`). |
| `train` | `cache_text_embeddings` | `true` so the text encoder can be unloaded after caching. |
| `train` | `optimizer` | `adafactor` (toolkit) or `hfadafactor` (stock Hugging Face). |
| `save` | `save_format` | `safetensors` |
| `sample` | width / height | Match the resolution you care about at 8 NFE (previews in the example jobs use 256 for speed). |

Do **not** expand this recipe to assistant LoRA, trajectory imitation, or Decoupled DMD. Those are separate training modes.

## What not to copy from a dense SFT job

| Avoid | Instead |
| --- | --- |
| `timestep_type: linear` or `shift` plus `content_or_style: gaussian` | `timestep_type: turbo_prior` |
| `use_diffsynth_training_loop: true` | `false` (required for this prior) |
| `use_dynamic_shifting: true` | `false` |
| `sample_steps` 9, 20, or 30 as the quality check | `8` |
| `guidance_scale` 1 or higher at sample time | `0` |
| Training and sampling from the same Turbo checkpoint only | Train on **Z-Image**, sample on **Z-Image-Turbo** |

## After training

Load the saved LoRA on **Z-Image-Turbo** and generate at **8 steps, CFG 0**, static shift 3 (the default Turbo sampler in this adapter when dynamic shifting is off). Quality at 20–30 steps is not the target metric for this recipe.
