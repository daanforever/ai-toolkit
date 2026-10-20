# LoRA heatmap

A Heatmap Graph tab on `/jobs/:id` (`JobHeatmapGraph`). Refresh calls `GET /api/jobs/:id/heatmap`. The route picks the newest `.safetensors` in the job folder and runs `toolkit/lora_heatmap.py` with the project venv. The script prints one JSON object. The tab draws it. Nothing is fetched until Refresh.

Dev server, from `ui/`: `npx next dev --port 3000`.

Every plan based on this file must start with a first item that requires reading this file (`docs/heatmap/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on the LoRA heatmap. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections. Do not change the numbering.

Structure (top to bottom):

| Section                      | Contains                                                                                                              | Does not contain                          |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | What the feature is, data flow in 2-3 sentences, dev server command, the plan rule                                    | Details covered below                     |
| `## 1. Done`                 | Implemented behavior as a contract. Subsections: `Tab`, `Grid`, `Help`, `API: GET /api/jobs/:id/heatmap`, `Scorer`    | Plans, limitations, test steps            |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                        | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing`      | Behavior already described in `1. Done`   |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and describe it in `1. Done`.

## 1. Done

### Tab

- Page key `heatmap`, label `Heatmap Graph`, placed after `Loss Graph` in `ui/src/app/jobs/[jobID]/page.tsx`. `mainCss` is `pt-24`. The card heading is `Heatmap graph`.
- Opening the tab does not request `/heatmap`. Header status is `Press Refresh`. The body says `Press Refresh to read the latest saved LoRA.`
- `Refresh` calls `GET /api/jobs/${job.id}/heatmap`. The button is disabled while status is `loading`. Header status is `Loading...`. The body says `Loading heatmap...` and the previous grid is hidden. The previous payload stays until the request finishes.
- Changing `job.id` resets to idle and drops the payload.
- A failed request clears the payload. Header status is `Error`. The body shows `response.data.error` when that field is a string, otherwise the axios message, otherwise `Failed to load heatmap`.
- `available: false` keeps status success. Header status is `No saved LoRA yet`. The body says `No saved LoRA yet.`
- `available: true` without `rows`, `cols`, or the selected metric leaves the header at `Press Refresh` and shows `Heatmap response is missing {metric}.`

### Grid

- Default metric is `frobenius`. The toggles are `||ΔW||_F`, `||B|| / ||A||`, `stable rank`, `effective rank`, mapped to `frobenius`, `balance`, `stable_rank`, `effective_rank`. Switching metrics does not call Refresh.
- Drawn rows are the payload `cols` top to bottom. Drawn columns are the payload `rows` left to right. Column headers are vertical and grow with the layer id. The type-name column and the corner cell are `sticky left-0` with the card background. The header text is `{cols} × {rows}`. The grid scrolls inside `max-h-[36rem]`. Cell size is fixed (`1.5rem` by `h-5`).
- Color is linear from the min to the max of the finite numbers in the selected metric. `null` is an empty cell (`n/a`), not the low end of the scale. When min equals max, every finite cell uses the high color.
- The scale card shows that min and max, or `No values yet.` when the metric has no finite number. Under it: `{filename}`, `step {step}` when `step` is not `null`, and `mtime_ms` via `toLocaleString()`. `epoch` is not shown.
- Hover sets a fixed tooltip and the cell `title` to `{row} · {col}: {formatted}` or `{row} · {col}: n/a`.

### Help

- The help card is under the controls and is visible before Refresh. Heading `What this map shows` is always the three paragraphs in `SHARED_HELP`. The second heading is the selected metric label. Changing the toggle replaces only that metric's paragraphs and its `What to do:` list.
- Color compares cells inside one map. A single color means the values are close; the scale numbers say whether that value is small or large. Brightness is size, not a grade. An empty cell is a zero adapter half, usually at the start.
- `q`, `k`, `v` are where the layer looks. `out` writes that look back. `w1`, `w2`, `w3` are what the layer does next. Other row names are the names from the file.
- `||ΔW||_F` is how loud the correction is in that cell. It is not compared with the original weights, so the same number in two layers is not the same kind of change. A few cells much larger than the rest took most of the change: leave them if the samples look right; if the change shows up in the wrong place, train fewer steps or lower the learning rate. Scale numbers still tiny late: train longer or raise the learning rate. Scale numbers large and samples look like copies of the training images: stop earlier or lower the learning rate. The colors cannot say that.
- `||B|| / ||A||` is how large the second half (up, starts empty) has grown versus the first (down, starts random). Tiny or empty at the start is normal. On a long run the second half often ends larger than the first. Tiny across the map late: train longer, or raise the up learning rate above the down learning rate. A few cells much larger than the rest, with blown-out or unstable samples: lower the learning rate. A second half larger than the first across the map is a normal end to a long run. Empty at the start: keep training. Empty late: that piece's second half is still zero.
- `stable rank` near 1: one direction holds the strength. Decrease the rank to drop idle directions and make a smaller adapter. A low number is not a failed run. Most trained LoRAs use only a fraction of the rank. Near the rank: increase the rank only if samples still miss detail. A high number is not better by itself, and a higher rank does not always learn more. One snapshot does not show whether the number fell during training.
- `effective rank` counts quiet directions too, so it is usually higher than stable rank. Near 1: decrease the rank. Near the rank while stable rank is also high: increase the rank only if samples still miss detail. High while stable rank stays near 1: decrease the rank. The quiet directions are not a reason to add rank.

### API: `GET /api/jobs/:id/heatmap`

`runtime` is `nodejs`. `maxDuration` is 60. No query parameters.

Success body is the scorer JSON plus `available: true`. The scorer object is `{ file, rows, cols, metrics }`. `file` is `{ filename, path, mtime_ms, step, epoch }`. `step` and `epoch` come from safetensors metadata `training_info` (`{"step", "epoch"}`) and are `null` when that key is absent. `metrics` has `frobenius`, `balance`, `stable_rank`, `effective_rank`, each a `rows` by `cols` matrix of number or `null`.

The file is the `.safetensors` file directly in `{trainingFolder}/{job.name}` with the greatest `mtimeMs`. Name order is not used. Equal mtime keeps the first name `readdir` returned. A missing folder, a folder with no such file, or a directory whose name ends in `.safetensors` is `{ available: false }` with status 200.

The child is `resolveVenvPython(TOOLKIT_ROOT)` plus `toolkit/lora_heatmap.py` and the absolute file path. `cwd` is `TOOLKIT_ROOT`. `CUDA_VISIBLE_DEVICES` is `-1`. `PYTHONIOENCODING` is `utf-8`. `resolveVenvPython` uses `.venv`, then `venv`, then `python` on `PATH`. Windows binaries are `Scripts/python.exe`. Unix binaries are `bin/python`. The same helper launches training from `ui/cron/actions/startJob.ts`.

The child is killed after 60000 ms. Stdout must be one JSON object. Non-zero exit uses trimmed stderr.

Errors (JSON `{ error }`):

| Status | Message                                              | Cause                                                                 |
| ------ | ---------------------------------------------------- | --------------------------------------------------------------------- |
| 404    | `Job not found`                                      | no row in `Job`                                                       |
| 500    | `lora_heatmap.py not found at {path}`                | script missing under `TOOLKIT_ROOT`                                  |
| 500    | `heatmap timed out after 60000ms`                    | child still running at 60s                                            |
| 500    | stderr text                                          | child exit code is not 0                                              |
| 500    | `heatmap exited with code {code}`                    | non-zero exit and empty stderr                                        |
| 500    | `heatmap returned invalid JSON: {message}`           | stdout is not JSON                                                    |
| 500    | `heatmap returned a non-object JSON value`           | JSON is `null` or an array                                            |
| 500    | spawn `error.message`                                | python failed to start                                                |

### Scorer

- CLI: `python toolkit/lora_heatmap.py <file.safetensors>`. Exit 0 prints JSON and a newline on stdout. Exit 2 prints `usage: python toolkit/lora_heatmap.py <file.safetensors>` on stderr. Exit 1 prints the exception message on stderr. Matplotlib is not imported.
- Keys: `{prefix}.lora_down.weight` shaped `(rank, in)`, `{prefix}.lora_up.weight` shaped `(out, rank)`, optional scalar `{prefix}.alpha`. Other keys are ignored. Computation is fp32 on CPU. `scale` is `alpha / rank`. Missing alpha and alpha `0` use `scale` 1.
- `frobenius` is `scale * ||up @ down||_F` from the `rank x rank` Gram matrices. `balance` is `||up||_F / ||down||_F`. `stable_rank` is `||dW||_F^2 / ||dW||_2^2`. `effective_rank` is `exp(entropy)` of the singular values, with `p_i = sigma_i / sum(sigma)`. Singular values come from a reduced QR of `down.T` and `up`, then an SVD of the `rank x rank` core, when both `in` and `out` are at least `rank`. Otherwise they come from `svdvals(up @ down)`.
- `||down||_F` or `||up||_F` at or below `1e-12` sets `balance`, `stable_rank`, and `effective_rank` to `null`. `frobenius` stays a number (`0` when the product is zero). A negative Gram product within `1e-5` of `trace(down) + trace(up) + 1` is clamped to `0`. A larger negative value raises `negative Frobenius energy {fro2}`. Rank metrics use the singular values of `up @ down`, so `scale` cancels.
- The earliest `single_transformer_blocks`, `transformer_blocks`, `double_blocks`, `single_blocks`, `context_refiner`, `noise_refiner`, `layers`, `blocks`, `layer`, or `block` index becomes the row id (`layers_0`, `noise_refiner_1`). Longer names are tried before shorter ones, so `blocks` does not take `double_blocks`. The remainder, with `.` turned into `_`, is the column. An empty remainder is the column `weight`. No such index uses row `other`. Rows with an index sort by block name then index; `other` is last. Columns sort alphabetically. A module present in only some rows is `null` in the others.
- `LyCORIS/LoKr weights are not supported (found {key}). Expected lora_down/lora_up.`
- `PEFT lora_A/lora_B weights are not supported (found {key}). Expected lora_down/lora_up.` LoKr is reported first when both are present.
- `No lora_down.weight tensors found. Expected a kohya-style LoRA safetensors file.`
- `non-matrix LoRA weights down={shape} up={shape}. Conv LoRA is not supported.`
- Also exit 1: a missing file, a corrupt safetensors file, alpha that is not a scalar, rank mismatch, a module missing up or down, a duplicate tensor, two modules mapped to the same cell, `training_info` that is not JSON or not an object.

## 2. To do

## 3. Notes for continuing

### Limits

- Only files directly in the job folder are considered. Checkpoints in a subdirectory are ignored.
- `||ΔW||_F` is not divided by the base weight norm. The same number in two layers is not the same fraction of the original layer.
- Conv LoRA, PEFT `lora_A`/`lora_B`, and LyCORIS/LoKr are rejected. Any other key is skipped, and the down/up pair is still scored.
- One snapshot. The help text does not know whether a number rose or fell across saves.
- `resolveVenvPython` checks that the `.venv` or `venv` directory exists, not that the interpreter file exists.
- Two Refresh clicks before the button re-renders can start two python processes. The later response replaces the earlier one.
- The help copy is fixed English in `JobHeatmapGraph.tsx`. It is not loaded from this file.

### Known issue

- None.

### Testing

- `venv\Scripts\python.exe -m pytest testing/test_lora_heatmap.py`: 13 passed. Synthetic LoRA against `scale * ||up @ down||_F` and the singular values of that product. Cases: bf16, `in` or `out` below rank, zero up (`null` balance and ranks), missing alpha, alpha `0`, no layer index (row `other`), z-image block names including `noise_refiner` and dotted `diffusion_model.layers.N`, PEFT keys, LoKr keys, conv weights, CLI exit 0/1/2, corrupt bytes.
- `temp/flower.safetensors` (present, gitignored): 30 rows `layers_0`..`layers_29`, 7 columns `attention_to_k`, `attention_to_out_0`, `attention_to_q`, `attention_to_v`, `feed_forward_w1`, `feed_forward_w2`, `feed_forward_w3`, `file.step` 10, rank 16, no nulls. `frobenius` 0.0342..0.0754, `balance` 0.000256..0.000562, `stable_rank` 1.37..6.71, `effective_rank` 7.08..15.53. In-process score about 0.25 s. `npm test` integration through `runLoraHeatmap` about 1.8 s. Artifacts: `temp/flower_heatmap.json`, `temp/flower_heatmap.png` (renderer `temp/render_lora_heatmap.py`).
- `npm test` in `ui/` (Vitest `3.2.4`) covers `resolveVenvPython` (`.venv` over `venv`, then `python`), the route (404, empty folder, newer mtime wins over an alphabetically earlier file, corrupt stderr, invalid JSON), and the flower spawn above. `npx tsc -p tsconfig.worker.json --noEmit` passed after `startJob.ts` switched to `resolveVenvPython`.
- Manually checked at `http://localhost:8675/jobs/c10ca475-e8e9-49fd-aa18-e13dc62fe84e` with `npx next dev --turbopack -p 8675`, job `heatmap-flower-check`, training folder `output`. The tab did not request `/heatmap` until Refresh. Flower then showed `7 × 30`, `flower.safetensors`, `step 10`, and tooltip `layers_0 · attention_to_k: 4.1048` on stable rank. The scale read `1.3691`..`6.7094`. Switching to stable rank did not add a request. A newer `step_0099.safetensors` (`training_info.step` 99, one module) replaced the grid with `1 × 1` and `step 99`. Deleting both files showed `No saved LoRA yet.` The job row and folder were removed afterwards. `temp/flower.safetensors` was left in place.
- Help card checked at `http://localhost:3000/jobs/dd6ad640-9344-4bb9-a55a-fd64026c986b` with `npx next dev --turbopack -p 3000`, job `heatmap-help-verify`, `temp/flower.safetensors` hardlinked into `output/heatmap-help-verify`. Before Refresh the shared text and the three `||ΔW||_F` action lines were visible, status was `Press Refresh`, and the scale said `No values yet.` Toggles for `stable rank`, `||B|| / ||A||`, and `effective rank` replaced only that section. No `/heatmap` request until Refresh. Refresh drew `7 × 30`. Switching back to stable rank kept the grid, showed scale `1.3691`..`6.7094`, and left the request count at 1. `Empty cells have no value (zero factor).` was absent. The job row and folder were removed afterwards. `temp/flower.safetensors` was left in place.
