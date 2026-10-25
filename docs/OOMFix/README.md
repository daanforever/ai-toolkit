# OOM fix

Not implemented.

LoRA step time dropped after `train.cuda_memory_fraction` (default 0.95) and, on Windows, `max_split_size_mb:24`. The fraction caps the CUDA caching allocator's reserved pool. At the cap the allocator synchronizes, frees cached blocks, and allocates again. A block at or above 24 MiB is reused only for a request within about 1 MiB of its size, so a bucket change pays that reclaim on the step.

Every plan based on this file must start with a first item that requires reading this file (`docs/OOMFix/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on the CUDA memory cap and LoRA step time in this repo. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections beyond the structure table. Do not change the numbering.
- `docs/OOMFix/Source.md` is the investigation record. This file wins where they differ.

Structure (top to bottom):

| Section                      | Contains                                                                                                         | Does not contain                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | Problem statement, the plan rule                                                                                 | Allocator internals, commit list, test steps |
| `## Contract`                | Behavior to implement. Subsections are added when the contract is written                                        | Shipped behavior, task list, test steps   |
| `## 1. Done`                 | Behavior that is already in the tree. Empty until a `Contract` subsection is moved here                          | The unimplemented spec                    |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                   | Done items, ideas without a concrete task |
| `## Ideas`                   | Candidate fixes, one subsection each. The chosen check and its measured result                                   | Shipped behavior, the task list, confirmed defects |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing` | Behavior already described in `Contract` or `1. Done` |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and move the matching behavior from `Contract` into `1. Done`. When `Contract` is empty, delete the "Not implemented." sentence in the intro.

## Contract

## 1. Done

## 2. To do

## Ideas

Candidates. Not a contract. First check is float8 on the frozen DiT. The repro default stays quantize off, so the Known issue run keeps its shape.

### float8 frozen DiT

- `model.quantize: true` and `qtype: float8` select torchao `Float8WeightOnlyConfig` in `toolkit/util/quantize.py`. `ZImageDiffSynthModel.load_model` rewrites `qfloat8` to `float8`.
- `load_components` calls `quantize_model` on the sampling DiT and the train DiT. `quantize_te` stays false. Block names come from `get_transformer_block_names`. This run quantized 30 blocks plus extras.
- Result, RTX 5080 15.89 GiB, `fraction=0.92`, `max_split_size_mb:24`, `OOM_REPRO_QUANTIZE=float8`, exit 0: 8/8 steps finished. `base.params` is 5.740 GiB (`n=595`). End of step alloc is 6.09 GiB and reserved is 6.85 GiB (6.84 on step 5). `later_median_s=18.969`, `first_wall_s=20.268`. `oversize_seg` stays 104.
- `1x16x216x256` finished (step 7, 59.555s). Cumulative reserved peak reads 14.6 GB from `Load components` onward. The cap on this card is `0.92 * 15.89` GiB = 14.62 GiB.
- Alloc high-water of the step is 13.6 GB, set on the sampling forward of step 7 (12.9 → 13.4 on the sampling H2D, 13.4 → 13.6 on that forward). The train forward does not move it. The 13.6 includes one float8 DiT (+5.8 GB on that move). The same step without quantize adds the weight gap, 11.464 − 5.740 GiB.
- Per-step counters. `d_retry` is 0 except step 2. Repeated `1x16x148x148` still does about 800 `d_malloc`.
- The step layout runs after `hook_train_loop` returns. `other_live_tensors` is 0.002–0.005 GiB. The largest entry is the batch latent `float32 [1, 16, H, W]`. The `bf16 [1, seq, 3840]` stack from Known issue is not in this log.

| Step | Latent | wall_s | d_retry | d_sync | d_malloc | d_free |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `1x16x148x148` | 20.268 | 0 | 3 | 1000 | 757 |
| 2 | `1x16x192x256` | 58.312 | 1 | 4 | 1434 | 1254 |
| 3 | `1x16x148x148` | 18.766 | 0 | 3 | 801 | 798 |
| 4 | `1x16x148x148` | 18.969 | 0 | 3 | 823 | 825 |
| 5 | `1x16x224x148` | 29.052 | 0 | 3 | 810 | 813 |
| 6 | `1x16x148x148` | 18.403 | 0 | 3 | 825 | 823 |
| 7 | `1x16x216x256` | 59.555 | 0 | 3 | 1583 | 1584 |
| 8 | `1x16x148x148` | 18.420 | 0 | 3 | 824 | 823 |

### resolution 1024

- `OOM_REPRO_QUANTIZE=off` and `OOM_REPRO_RESOLUTION=1024`. Exit 0, 8/8 steps. `base.params` is 11.464 GiB. End-of-step alloc is 11.79 GiB, reserved is 12.30 GiB. `later_median_s=15.669`.
- Census is taken at the return of a grad-enabled `get_noise_prediction`. `turbo=0` and `turbo=1` print the same saved-input stack. `n=29` of `bf16 [1, seq, 3840]`. No `[1, seq, 10240]` row at that point.

| Latent | shape | n | GiB |
| --- | --- | --- | --- |
| `1x16x128x84` | `[1, 2752, 3840]` | 29 | 0.571 |
| `1x16x88x128` | `[1, 2880, 3840]` | 29 | 0.597 |
| `1x16x108x128` | `[1, 3520, 3840]` | 29 | 0.730 |
| `1x16x128x128` | `[1, 4160, 3840]` | 29 | 0.863 |

### checkpoint activation offload

- `run_forward` takes `use_gradient_checkpointing_offload` and defaults it to false. The call in `ZImageDiffSynthModel` does not pass it.
- This applies only when that flag moves the saved block input off the GPU. Those inputs are the `bf16 [1, seq, 3840]` stack in Known issue. Cost is a copy of that input per block per step.

### layer offload

- `model.layer_offloading` streams frozen Linear weights from CPU through `toolkit/memory_management/manager.py`.
- `pack=True` in `compile_blocks.py` puts a frozen block into one `uint8` storage. The repro sets `compile: false`, so this path is not part of the float8 check.

### bucket grouping

- `UnifiedBucketManager.shuffle_and_build_batches` shuffles `batch_indices` across buckets, so consecutive steps change `seq`.
- A cached block at or above 24 MiB is reused only inside the window in the intro. The latent shapes in Known issue fall outside that window of each other. A run of steps on one bucket makes reclaim a boundary event.

### expandable_segments

- `run.py` states that Windows has no `expandable_segments`. That is unverified on torch `2.12.1+cu132`. The check is a process that starts with `expandable_segments:True`. That flag would replace `max_split_size_mb:24`.

### garbage_collection_threshold

- With a fraction set and the threshold `> 0`, the allocator frees unused cached blocks before the syncing reclaim. This repo does not set the threshold.
- That free does not restore cross-size reuse. It matters when `d_sync` on the repro line dominates `d_malloc`.

### refiner modules

- The comment in `ZImageDiffSynthModel.load_model` says `noise_refiner` ~10 GB and `context_refiner` ~4 GB. Known issue `base.params` with both refiners enabled is the whole frozen DiT. Disabling them is not that comment until `base.params` is measured with the repro flags flipped.
- `load_model` defaults `disable_noise_refiner` and `disable_context_refiner` to true. The repro sets both false.

## 3. Notes for continuing

### Limits

- Combat `disable_sampling: false` and `steps: 30000` are not applied, so the counters stay train steps.
- Combat `unload_text_encoder: false` does not leave the text encoder on GPU. `cache_text_embeddings: true` still takes the unload path.

### Known issue

- `datasets/chairs` at resolution 2048 aborts inside `hook_train_loop` with `RuntimeError: OOM during training step 3 times in a row, aborting training`. The inner exception is `torch.cuda.OutOfMemoryError` or a `RuntimeError` containing `CUDA out of memory`.
- Before the step, `base.params` is 11.464 GiB and CUDA alloc is 11.7 GB. The OOM snapshot's extra live tensors are DiT activations. `patch_size=2` makes the image sequence `(H/2)*(W/2)`.

| Step | Latent | dtype | shape | grad | n | GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `1x16x172x172` | `torch.bfloat16` | `[1, 7488, 3840]` | true | 30 | 1.607 |
| 0 | `1x16x172x172` | `torch.float32` | `[1, 7488, 3840]` | false | 3 | 0.321 |
| 0 | `1x16x172x172` | `torch.float32` | `[1, 7488, 30, 128]` | false | 2 | 0.214 |
| 1 | `1x16x224x148` | `torch.bfloat16` | `[1, 8352, 3840]` | true | 32 | 1.912 |
| 1 | `1x16x224x148` | `torch.bfloat16` | `[1, 8352, 10240]` | true | 2 | 0.319 |
| 3 | `1x16x216x256` | `torch.bfloat16` | `[1, 13888, 3840]` | true | 15 | 1.490 |
| 3 | `1x16x216x256` | `torch.bfloat16` | `[1, 13888, 10240]` | true | 2 | 0.530 |
- After the exception unwinds, `other_live_tensors` is 0.002 GiB. The leftover batch tensor is `torch.float32 [1, 16, H, W]`. The cached text embed left in the census is `torch.float32 [1, 8, 2560]`.
- This is normal per-block checkpointing, not a leak. `gradient_checkpoint_forward` calls `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` once per block and keeps that block's input. `n=30` of `bf16 [1, seq, 3840]` is those saved inputs (`30 * seq * 3840 * 2` bytes). The FFN activation `bf16 [1, seq, 10240]` (`hidden_dim = dim / 3 * 8`) is only the live block (`n=2`). On the longest latent the forward dies at `n=15`, so the stack is not retained past the failing block. `lora.grads` stays `n=0`.

### Testing

- `testing/oom/repro.py` is the LoRA repro of the combat train shape. Dataset `datasets/chairs`, rank 64 on attention `to_q`/`to_k`/`to_v`, `train.dtype` fp32, Adafactor `beta2` 0.996 and `beta1` 0 with stochastic accumulation and rounding off, gradient checkpointing, refiners on, `turbo_lora` true, sampling off, `cuda_memory_fraction` `0.92`, `logging.debug` true. It calls `run_job` and prints one counter line per step.
- `logging.debug: true` enables `memory_debug` in `toolkit/util/debug.py` and `log_oom_layout`. The train step is wrapped with `memory_debug(print_acc, "Train step {step}", kind="cuda", layout=self.memory_layout)`. That prints `[DEBUG Train step N] CUDA alloc/reserved/cache/diff` and `[DEBUG Train step N] CUDA layout`.
- The layout census lists owners from `memory_layout_owners` (`lora.params`, `lora.grads`, `optimizer.state`, `base.params`, `base.buffers`, `base.grads`, `vae`, `text_encoder`), then `other_live_tensors` grouped as `dtype shape grad= n= GiB`, then `untracked(autograd/C++)`. On OOM, `log_oom_layout` prints the same census as `[DEBUG OOM at train step N]` inside the `except`, while the failed step's tensors are still alive.
- Direct run: `venv\Scripts\python.exe testing/oom/repro.py`
- Pytest skips unless `OOM_REPRO=1`: `venv\Scripts\python.exe -m pytest testing/oom/test_repro.py -q -s`
- `OOM_CUDA_MEMORY_FRACTION` overrides the cap; `none` passes null. `OOM_REPRO_DTYPE` overrides `train.dtype`. `OOM_REPRO_MAX_SPLIT` defaults to `24` on Windows; `off` removes `max_split_size_mb`. `OOM_REPRO_STEPS` defaults to 8. `OOM_REPRO_RESOLUTION` defaults to 2048. `OOM_REPRO_QUANTIZE` defaults to off (`quantize: false`, `qtype: qfloat8`); `float8` sets `quantize: true` and `qtype: float8`. Base weights: `ZIMAGE_DIFFSYNTH_MODEL_PATH`, otherwise `DEFAULT_ZIMAGE_MODEL_PATH` in `extensions_built_in/diffusion_models/z_image_diffsynth/test_smoke.py`. Turbo weights: `ZIMAGE_DIFFSYNTH_SAMPLING_PATH`, otherwise `models--dimitribarbot--Z-Image-Turbo-BF16` snapshot `e11f18c054525642077b44c5edf00b99515a6182`.
- A step line is `[oom-repro] step=N wall_s=… latent=… alloc_gib=… reserved_gib=… d_retry=… d_sync=… d_malloc=… d_free=… oversize_seg=… oversize_alloc=…`. After the run, `later_median_s` is the median wall time of every step except the first. No speed threshold.
- A grad-enabled `get_noise_prediction` prints `[oom-repro] activations turbo=… latent=… alloc_gib=… untracked_gib=…` and `[oom-repro] act dtype shape grad= n= gib=` for each CUDA storage whose nbytes equal the tensor shape and whose group is at least 64 MiB.
