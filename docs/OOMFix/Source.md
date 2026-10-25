# CUDA memory cap and LoRA step-time regression

Source notes for a later fix. Facts below were read from this tree and from PyTorch's caching allocator. Nothing here is a chosen fix.

## Symptom

After `train.cuda_memory_fraction` landed, LoRA training step time dropped sharply. The cap was added to stop a different slowdown: on a 16 GiB Windows card the caching allocator's reserved pool grew past physical VRAM and Windows paged it into shared system memory. Allocated tensors were stable; the growth was the cache.

Both regimes are slow, for different reasons:

- Shared-VRAM spill: kernels touch pages that live in system RAM. Step time degrades after reserved memory crosses physical VRAM.
- Cap reclaim: the allocator is not allowed to keep that reserved pool. Allocations that used to hit the cache synchronize the GPU, `cudaFree` cached segments, and `cudaMalloc` them again.

A workload that already fitted in dedicated VRAM, with a stable cache, gets only the second cost.

## Commits

| Commit | Date | What changed |
| --- | --- | --- |
| `850922ed` | 2026-10-11 12:54 +0300 | `fix(train): cap cuda allocator to avoid shared vram`. Adds `train.cuda_memory_fraction`, default **0.9**. Applied once at the start of `BaseSDTrainProcess.run`. |
| `23597bf3` | 2026-10-11 15:06 +0300 | `fix(train): stop cuda cache fragmentation ooms`. Windows `max_split_size_mb:24`. Packed frozen DiT blocks. LoRA autocast cast removed. Sim yaml cap set to 0.9, rank 8, 40 steps, varied-shape dataset. |
| `b02d5e21` | 2026-10-17 21:24 +0300 | `fix(train): increase cuda memory fraction to improve VRAM management`. Default **0.9 → 0.95**. Same mechanism, higher ceiling. Commit text talks about OOM, not step time. |

`23597bf3` describes the failure the cap created: varying image buckets on a 16 GB GPU fragment the cache until a step cannot place a ~60 MiB activation under the 0.9 cap. Allocated memory between steps was flat.

## Current behavior in this tree

Default is on for every `TrainConfig`. A missing yaml key means 0.95. `None` is the only off switch (`cuda_memory_fraction: null`).

`toolkit/config_modules.py`, the `TrainConfig` default:

```python
# cap on PyTorch's share of total VRAM (0-1]. At the cap the allocator frees its
# cached blocks and retries before raising OOM, which keeps reserved memory from
# spilling into shared system VRAM on Windows. None = no cap.
self.cuda_memory_fraction: Optional[float] = kwargs.get('cuda_memory_fraction', 0.95)
```

Applied before model load, on the current CUDA device. `accelerator.device` can be an unindexed `"cuda"`, so the call does not pass a device index.

`jobs/process/BaseSDTrainProcess.py`, start of `run`:

```python
if self.train_config.cuda_memory_fraction is not None:
    # current CUDA device: accelerator.device can be an unindexed "cuda"
    torch.cuda.set_per_process_memory_fraction(self.train_config.cuda_memory_fraction)
    print_acc(f"CUDA memory fraction capped at {self.train_config.cuda_memory_fraction}")
```

Windows-only, in `run.py`, before any torch import. Skipped when `PYTORCH_CUDA_ALLOC_CONF` already contains `max_split_size_mb`.

`run.py`, before any torch import:

```python
# Windows has no expandable_segments: with varying batch shapes, long-lived activations
# split cached segments of short-lived ones and pin the remainders until a step OOMs at
# the train.cuda_memory_fraction cap. Blocks >= max_split_size_mb are never split, so they
# stay whole and releasable. Must stay above the allocator's 20 MiB small-request segments.
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if os.name == "nt" and "max_split_size_mb" not in _alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        filter(None, [_alloc_conf, "max_split_size_mb:24"])
    )
```

`extensions_built_in/diffusion_models/z_image_diffsynth/simulate_turbo_prior.yaml` still sets `cuda_memory_fraction: 0.9`. No other yaml in the tree sets the key. The cap is not z-image-specific; every training job through `BaseSDTrainProcess` gets it. `max_split_size_mb:24` applies to every Windows `run.py` launch.

Installed torch in this venv: `2.12.1+cu132`. Allocator behavior below was read from `c10/cuda/CUDACachingAllocator.cpp` on PyTorch `main` (the file fetched during the investigation). The fraction check, the retry ladder, and the oversize-block rules match that file. Re-read the same file at tag `v2.12.1` before patching if a line has moved.

## What the cap actually limits

`set_per_process_memory_fraction(fraction)` sets

`allowed_memory_maximum = fraction * device_prop.totalGlobalMem`

when `fraction < 1`. Fraction `1.0` clears the cap. The limit is a fraction of **total** device memory, not of free memory. Desktop and other processes are not subtracted. On a 16 GiB card, 0.95 is about 15.2 GiB of PyTorch reserved memory; 0.9 was about 14.4 GiB.

Inside the allocator, `total_allocated_memory` is the **reserved** pool: it increases on `cudaMalloc` of a segment and decreases on `cudaFree` of a segment. It is not the live-tensor total. The cap therefore bounds the cache, which is the thing the allocator is supposed to keep.

`alloc_block` refuses the allocation before `cudaMalloc` when `total_allocated_memory + size > allowed_memory_maximum`. The malloc path then:

1. Searches the free-block pool (`get_free_block`).
2. If the cap is set and `garbage_collection_threshold > 0`, runs `garbage_collect_cached_blocks`. That path does not force a device sync. This repo does not set the threshold. PyTorch's default is 0, so this branch does not run.
3. Tries `alloc_block`. Under the cap this returns `cudaErrorMemoryAllocation` immediately.
4. `release_available_cached_blocks`, then `alloc_block` again. This function returns immediately when `max_split_size` is unlimited. With `max_split_size_mb:24` it `cudaFree`s enough oversize cached blocks to cover the request.
5. `release_cached_blocks`, then `alloc_block` with `isRetry=true` (`num_alloc_retries`). This calls `synchronize_and_free_events` (`num_sync_all_streams`, `cudaEventSynchronize` on every outstanding allocator event) and then `cudaFree`s every non-split cached block in the large and small pools.
6. If that still fails, OOM. The error string includes the allowed size.

A successful step that merely sits on the cap still pays steps 4 or 5. That is a GPU pipeline drain plus driver alloc/free, then the real kernel. On WDDM, `cudaMalloc` / `cudaFree` of large blocks are heavy even when the event sync is skipped.

PyTorch documents `max_split_size_mb` as a last resort whose performance cost ranges from zero to substantial. The usual examples are 128 or 512, not 24.

## Why `max_split_size_mb:24` makes the cap a per-step cost

Allocator rules that matter here:

- Large and small pools are separate. Small requests are packed into 2 MiB buffers. The small/large threshold in the file header is 1 MiB.
- A request between 1 MiB and 10 MiB allocates and splits a 20 MiB segment when no free block fits (`kMinLargeAlloc` / `large_segment_size`). The comment in `run.py` keeps `max_split_size_mb` above that 20 MiB segment so those segments can still be split. 24 is just above that floor.
- A block `>= max_split_size` is never split (`should_split`).
- `get_free_block` will not hand an oversize block to a smaller request. An oversize block satisfies a request only when the request is within about 1 MiB of that block's size (header comment: "within 1MB of the oversize cached block size"; the check uses `max_non_split_rounding_size`).

So a cached 60 MiB activation is not reused for a 40 MiB activation. The 60 MiB block stays reserved, counts toward the 95% cap, and the 40 MiB request needs a new segment. If that segment would cross the cap, the allocator frees oversize cache (or syncs and frees the whole non-split cache) and allocates again. The next bucket size repeats the cycle.

Without `max_split_size`, a larger free block can be split and the remainder stays cached. That is the fast path the 11 Oct follow-up disabled, because those remainders were pinned by long-lived tensors until a step OOM'd at the cap. `expandable_segments` is the upstream answer to that fragmentation. `run.py` states it is unsupported on this Windows platform. Do not assume it can be turned on here without a separate check.

24 MiB is below typical diffusion activations. Once a tensor is that large it becomes its own unsplittable segment. Fixed-shape steps can still hit an exact-size cache entry. Shape changes, and any size that is not within ~1 MiB of a cached oversize block, cannot.

## Why LoRA on Z-Image hits this

- The frozen DiT stays resident. LoRA optimizer state is small. Almost all VRAM is weights plus activations, so the process lives near the cap on a 16 GiB card. The commits that added the cap were reproduced on that card.
- `ZImageDiffSynthModel.gradient_checkpointing` defaults to `True` (`model.py`). Checkpointing allocates and frees a large activation per block on the forward and again on the backward recompute. That pattern is cheap only when the caching allocator reuses blocks.
- Training buckets change activation sizes. `23597bf3` used a varied-shape jpg/txt folder to reproduce the OOM. The same shape mix is what makes oversize blocks unusable for the next step.
- The cap is applied before load, so the reserved pool is already on the leash while weights are materialised. After that, every training step inherits it.

`_flush_cuda` (`gc.collect`, `empty_cache`, `synchronize`) runs on sampling-transformer moves and parking, not on every train step. It discards the pool; the next step rebuilds it under the same cap. It is an extra flush at sample time, not the steady step-time mechanism.

## Same-day edits that are not the step-time cause

`23597bf3` also did these. They do not match a large, steady step slowdown:

- **Frozen-weight packing.** `compile_blocks.py` calls `safe_module_to_device(..., pack=True)` for each DiT block on a CUDA move. `toolkit/util/device.py` `_pack_frozen_params` copies frozen plain contiguous parameters into one `uint8` storage, 512-byte aligned, and views them back. Quantized and trainable parameters stay on the old path. This is once per load. 512-byte alignment is enough for GEMM. A packed block's storage stays alive until every one of those params has left the device. That increases how much of the cap is permanently reserved; it does not add a sync per step.
- **LoRA autocast cast.** `toolkit/network_mixins.py` skips `x.to(lora_down.weight.dtype)` while autocast is enabled on that device, and deletes the unscaled LoRA output before the residual add. The old cast was a full-size copy that autocast immediately recast. Removing it saves memory and time.
- **Swallowed exceptions in the Z-Image load path.** Refiner disable, VAE move, and `model_kwargs` parsing now raise. Behavior change on failure, not on the hot path.

## How to confirm on a live run

`torch.cuda.memory_stats()` after warmup. The regression is this pattern, not a one-time load spike:

- `reserved_bytes` pinned near `0.95 * totalGlobalMem` (or `0.9` when the yaml sets that).
- `num_alloc_retries`, `num_sync_all_streams`, `num_device_alloc`, `num_device_free` keep rising every step.
- `oversize_segments` / `oversize_allocations` are high because of `max_split_size_mb:24`.
- `allocated_bytes` stays flat while reserved sits on the cap. That matches the original commit note: live tensors were stable, the cache was the problem.

A/B that isolates the mechanism: `cuda_memory_fraction: null`, and a Windows launch without `max_split_size_mb:24` (the env var must be set to something that does not contain that key, or `run.py` will inject it). If step time returns, the cap plus the split limit is sufficient. Compare a fixed bucket and a mixed-bucket run; the mixed run should be worse under `max_split_size_mb:24`.

Watch `num_alloc_retries` separately from wall time. OOM is the same path after the retries fail. A run can be slow and still finish.

## Constraints on a later fix

- The cap exists because an uncapped cache on Windows 16 GiB spilled into shared system RAM and the step time collapsed after a few steps. Removing the cap with no other guard restores that failure mode.
- The 0.9 cap then OOM'd on a ~60 MiB activation while allocated bytes were flat, because fragmentation left no splittable hole under the cap. `max_split_size_mb:24` is the workaround for that, and it is what stops cross-size reuse.
- Raising 0.9 to 0.95 only moved the ceiling (~0.8 GiB on 16 GiB). It does not change the reclaim path. If the working set still sits on the cap, step time stays bad.
- `expandable_segments` is recorded in `run.py` as unsupported on this platform. A fix that depends on it needs a Windows test, not an assumption.
- `garbage_collection_threshold` is a less synchronous free of old cached blocks, and it runs only when a fraction is set and the threshold is `> 0`. It is not configured in this repo. It is not, by itself, a substitute for cross-size reuse.
- The default applies to every train job, not only Z-Image and not only jobs that were spilling. A fix that leaves 0.95 as the implicit default keeps the cost on workloads that had headroom.
- `simulate_turbo_prior.yaml` pins 0.9. Any default change leaves that run on the tighter cap until the yaml is edited.
- Do not "fix" this by adding per-step `empty_cache`. That is the same flush the cap already triggers, on purpose.

## Open checks

- Measure one fixed-shape LoRA step and one mixed-bucket step with the stats above, at 0.95, at `null`, and with `max_split_size_mb` unset. The two knobs need to be separated; they shipped on the same day.
- Record whether `reserved_bytes` actually reaches the cap on the user's GPU. If it does not, the fraction check is only a compare and the slowdown is the split limit (extra `cudaMalloc`s, then the cap is hit sooner because unsplittable blocks waste reserved memory).
- Check how much dedicated VRAM the desktop already holds. The fraction does not account for it. `cudaMalloc` can still fail below `allowed_memory_maximum` and fall into the same retry ladder.
- Re-read `CUDACachingAllocator.cpp` at `v2.12.1` before relying on a line number. The behavior cited here is from `main` as fetched on 2026-10-25.
