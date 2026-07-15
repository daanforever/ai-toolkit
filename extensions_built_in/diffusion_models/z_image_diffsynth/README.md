# Z-Image DiffSynth adapter (`zimage_diffsynth`)

## Flags: `use_diffsynth_training_loop`, `use_diffsynth_prompt_encoding`, and `use_dynamic_shifting`

- **Training loop:** `model.model_kwargs["use_diffsynth_training_loop"]` (default `true` if omitted).
- **Dynamic time shifting:** `model.model_kwargs["use_dynamic_shifting"]` (default `false` if omitted). When `true`, uses Flux-style resolution-dependent `mu` + exponential `time_shift` (as in diffusers `ZImagePipeline`). Requires **`use_diffsynth_training_loop: false`** and **`train.timestep_type: shift`** (or `flux_shift`). Ignored when DiffSynth training loop is enabled.
- **Prompt encoding:** `model.model_kwargs["use_diffsynth_prompt_encoding"]` — if omitted, inherits `use_diffsynth_training_loop`.
  - `true`: literal `<|im_start|>user…` chain (`encode_prompt_diffsynth_literal_t2i`, DiffSynth T2I).
  - `false`: toolkit chat template (`encode_prompt`).
  - Example: `use_diffsynth_training_loop: false` with `use_diffsynth_prompt_encoding: true` keeps toolkit loss / SNR / scheduler but DiffSynth literal prompts.

## `model.compile` (per-block DiT `torch.compile`)

- Set **`model.compile: true`** to JIT-compile DiffSynth DiT blocks with `torch.compile(..., dynamic=True)` (Z-Image paper §4.2 style).
- Order: **quantize → LoRA → compile** (compile runs in `hook_before_train_loop` after the main DiT is on GPU). Gradient checkpointing stays **outside** each compiled block.
- Works with the usual **`quantize: true`** (quanto float8) path.
- Diffusers sampling transformer (`sampling_loader: diffusers` / `_sampling_is_diffusers`) is **not** compiled by this path; only DiffSynth `ZImageDiT` ModuleLists (`layers`, optional refiners).
- Device moves (sample / offload): **same-device is a no-op** (keeps compiled blocks); real moves **unwrap → Parameter/buffer replace (not `Module.to`) → recompile on GPU** so quanto + compile do not hit `Couldn't swap ... weight` / weakref errors.
- On Windows or when inductor/Triton is unavailable, failures are soft: training continues in eager mode.
- Expect slower first steps while JIT warms up.

## `use_diffsynth_training_loop`

- **Source of truth:** `model.model_kwargs["use_diffsynth_training_loop"]` (default `true` if omitted).
- **`ZImageDiffSynthTrainer`** copies the same boolean to `trainer.use_diffsynth_training_loop` at init (for tests and logging). **`TrainConfig` is not extended.**
- **`true` (DiffSynth mode):**
  - Prompt encoding for train / preview / cache: literal `<|im_start|>user…` chain as in DiffSynth `encode_prompt_omni` with no `edit_image` (see `diffsynth_training.encode_prompt_diffsynth_literal_t2i`).
  - Flow MSE aggregation matches **FlowMatchSFTLoss** scaling for the usual case: per-sample `timestep_weight * mean_spatial((pred - target)² × mask)` (see `diffsynth_training.aggregate_flow_matching_mse_diffsynth`).
  - **`linear_timesteps` is forced to `true`** on the trainer so `get_weights_for_timesteps` uses DiffSynth `linear_timesteps_weights` (same family as `FlowMatchScheduler.training_weight`). If you need full manual control of toolkit timestep weighting, set **`use_diffsynth_training_loop: false`**.
- **`false` (toolkit loop):** generic `prompt_encoding.encode_prompt` (chat template), standard `SDTrainer` MSE path with your YAML `timestep_type` / `content_or_style` / SNR settings; trainer sets `noise_scheduler: flowmatch` string for the batch processor.

### Example: dynamic time shifting (toolkit loop)

```yaml
process:
  - type: z_image_diffsynth_trainer
    train:
      noise_scheduler: flowmatch
      prediction_type: flowmatch
      timestep_type: shift
    model:
      arch: zimage_diffsynth
      model_kwargs:
        use_diffsynth_training_loop: false
        use_dynamic_shifting: true
```

Default `use_dynamic_shifting: false` keeps current behaviour: DiffSynth static `shift=3` in DiffSynth loop, static shift in toolkit loop.

## Checklist vs DiffSynth `Z-Image.sh`

When aligning a job with [DiffSynth-Studio/examples](DiffSynth-Studio/examples/z_image/model_training/lora/Z-Image.sh), compare at least:

- Learning rate, epochs / dataset repeats, resolution / `max_pixels` (toolkit: bucket / resolution lists).
- `use_diffsynth_training_loop: true` and the forced `linear_timesteps` behaviour above.
- Latent caching: same VAE / latent layout as Z-Image training expects.
- Optimizer and batch size (toolkit `train` section vs script).

## Tests

- `test_scheduler_dynamic_shifting.py` — `use_dynamic_shifting` scheduler config and per-resolution timesteps.
- `test_diffsynth_training.py` — loss scale, timestep single-application, config reader.
- `test_snr_weighting.py` — SNR / `min_snr_gamma` for toolkit loop (`use_diffsynth_training_loop: false`).
- `test_smoke.py` step 4d — SNR API guard for default DiffSynth adapter (`compute_snr`; SNR disabled in default training).
- `test_smoke.py` step 7a — trainer flags and `linear_timesteps`.
- `test_compile_quantized_blocks.py` — per-block compile helper (CPU) + float8/quanto + checkpoint smoke (CUDA).
- `testing/test_gaussian_full.py` — regression with `use_diffsynth_training_loop: false` (toolkit gaussian path).
