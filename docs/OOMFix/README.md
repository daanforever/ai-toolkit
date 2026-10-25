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
- Do not rename, reorder, or add top-level sections. Do not change the numbering.
- `docs/OOMFix/Source.md` is the investigation record. This file wins where they differ.

Structure (top to bottom):

| Section                      | Contains                                                                                                         | Does not contain                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | Problem statement, the plan rule                                                                                 | Allocator internals, commit list, test steps |
| `## Contract`                | Behavior to implement. Subsections are added when the contract is written                                        | Shipped behavior, task list, test steps   |
| `## 1. Done`                 | Behavior that is already in the tree. Empty until a `Contract` subsection is moved here                          | The unimplemented spec                    |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                   | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing` | Behavior already described in `Contract` or `1. Done` |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and move the matching behavior from `Contract` into `1. Done`. When `Contract` is empty, delete the "Not implemented." sentence in the intro.

## Contract

## 1. Done

## 2. To do

## 3. Notes for continuing

### Limits

### Known issue

### Testing

- `testing/oom/repro.py` is the LoRA repro. Four aspect ratios, rank 8, gradient checkpointing, refiners off, sampling off, bf16. It calls `run_job` and prints one counter line per step.
- Direct run: `venv\Scripts\python.exe testing/oom/repro.py`
- Pytest skips unless `OOM_REPRO=1`: `venv\Scripts\python.exe -m pytest testing/oom/test_repro.py -q -s`
- `OOM_CUDA_MEMORY_FRACTION` defaults to `0.95`; `none` passes null. `OOM_REPRO_MAX_SPLIT` defaults to `24` on Windows; `off` removes `max_split_size_mb`. `OOM_REPRO_STEPS` defaults to 8. `OOM_REPRO_RESOLUTION` defaults to 1024. `OOM_REPRO_DTYPE` defaults to `bf16`. Model path: `ZIMAGE_DIFFSYNTH_MODEL_PATH`, otherwise `DEFAULT_ZIMAGE_MODEL_PATH` in `extensions_built_in/diffusion_models/z_image_diffsynth/test_smoke.py`.
- A step line is `[oom-repro] step=N wall_s=… latent=… alloc_gib=… reserved_gib=… d_retry=… d_sync=… d_malloc=… d_free=… oversize_seg=… oversize_alloc=…`. After the run, `later_median_s` is the median wall time of every step except the first. No speed threshold.
