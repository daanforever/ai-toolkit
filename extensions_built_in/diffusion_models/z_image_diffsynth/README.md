# Z-Image DiffSynth adapter (`zimage_diffsynth`)

## Flags: `use_diffsynth_training_loop` and `use_diffsynth_prompt_encoding`

- **Training loop:** `model.model_kwargs["use_diffsynth_training_loop"]` (default `true` if omitted).
- **Prompt encoding:** `model.model_kwargs["use_diffsynth_prompt_encoding"]` — if omitted, inherits `use_diffsynth_training_loop`.
  - `true`: literal `<|im_start|>user…` chain (`encode_prompt_diffsynth_literal_t2i`, DiffSynth T2I).
  - `false`: toolkit chat template (`encode_prompt`).
  - Example: `use_diffsynth_training_loop: false` with `use_diffsynth_prompt_encoding: true` keeps toolkit loss / SNR / scheduler but DiffSynth literal prompts.

## `use_diffsynth_training_loop`

- **Source of truth:** `model.model_kwargs["use_diffsynth_training_loop"]` (default `true` if omitted).
- **`ZImageDiffSynthTrainer`** copies the same boolean to `trainer.use_diffsynth_training_loop` at init (for tests and logging). **`TrainConfig` is not extended.**
- **`true` (DiffSynth mode):**
  - Prompt encoding for train / preview / cache: literal `<|im_start|>user…` chain as in DiffSynth `encode_prompt_omni` with no `edit_image` (see `diffsynth_training.encode_prompt_diffsynth_literal_t2i`).
  - Flow MSE aggregation matches **FlowMatchSFTLoss** scaling for the usual case: per-sample `timestep_weight * mean_spatial((pred - target)² × mask)` (see `diffsynth_training.aggregate_flow_matching_mse_diffsynth`).
  - **`linear_timesteps` is forced to `true`** on the trainer so `get_weights_for_timesteps` uses DiffSynth `linear_timesteps_weights` (same family as `FlowMatchScheduler.training_weight`). If you need full manual control of toolkit timestep weighting, set **`use_diffsynth_training_loop: false`**.
- **`false` (toolkit loop):** generic `prompt_encoding.encode_prompt` (chat template), standard `SDTrainer` MSE path with your YAML `timestep_type` / `content_or_style` / SNR settings; trainer sets `noise_scheduler: flowmatch` string for the batch processor.

## Checklist vs DiffSynth `Z-Image.sh`

When aligning a job with [DiffSynth-Studio/examples](DiffSynth-Studio/examples/z_image/model_training/lora/Z-Image.sh), compare at least:

- Learning rate, epochs / dataset repeats, resolution / `max_pixels` (toolkit: bucket / resolution lists).
- `use_diffsynth_training_loop: true` and the forced `linear_timesteps` behaviour above.
- Latent caching: same VAE / latent layout as Z-Image training expects.
- Optimizer and batch size (toolkit `train` section vs script).

## Tests

- `test_diffsynth_training.py` — loss scale, timestep single-application, config reader.
- `test_smoke.py` step 7a — trainer flags and `linear_timesteps`.
- `testing/test_gaussian_full.py` — regression with `use_diffsynth_training_loop: false` (toolkit gaussian path).
