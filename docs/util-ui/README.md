# util-ui

`z_image_diffsynth` util-ui adds buttons to the job page and delivers each click to `ZImageDiffSynthTrainer` as a row in `RuntimeCommand`. The first button, `shrink_lora`, multiplies LoRA A and LoRA B by `0.9` on `end_step_hook`. The UI process does not hold the weights.

Every plan based on this file must start with a first item that requires reading this file (`docs/util-ui/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on util-ui in this repo. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections. Do not change the numbering.

Structure (top to bottom):

| Section                      | Contains                                                                                                         | Does not contain                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Intro (above)                | What util-ui does in this repo, where a click is applied, the plan rule                                         | Details covered below                     |
| `## 1. Done`                 | Implemented behavior as a contract. Subsections: `Buttons`, `Command queue`, `Scale`, `Trainer hook`             | Plans, limitations, test steps            |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                   | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing` | Behavior already described in `1. Done`   |

Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and describe it in `1. Done`.

## 1. Done

### Buttons

- `JobActionBar` prop `showExtensionActions` defaults off. Only `ui/src/app/jobs/[jobID]/page.tsx` sets it. `ui/src/components/JobsTable.tsx` does not.
- The slot fetches `GET /api/jobs/:id/ui-actions` when the prop is on, `canStop` is true, and `config.process[0].type` is `z_image_diffsynth_trainer`. Buttons render before `Play`.
- The icon map is `TriangleAlert` from `lucide-react`. Any other icon name is not rendered.
- `actions.json` entry: `id` `shrink_lora`, `icon` `TriangleAlert`, `title` `Multiply LoRA A and B by 0.9`. The title is the button `title`.
- The control is disabled while its `POST` is in flight. A click does not write `job_config`.
- A failed GET logs `Error loading ui actions:`. A failed POST logs `Error posting ui action:`.

### Command queue

- Prisma model `RuntimeCommand` is the SQLite table. Columns: `id`, `jobId`, `action`, `createdAt`. `jobId` references `Job` with `onDelete: Cascade`. Index is `(jobId, id)`. There is no payload column. The factor `0.9` lives in `scale_lora`.
- Button list file: `extensions_built_in/diffusion_models/z_image_diffsynth/util_ui/actions.json`. The Next route reads it from disk.
- `GET` returns `{ "actions": [ ... ] }` for `z_image_diffsynth_trainer` and `{ "actions": [] }` for any other process type. A missing job returns 404 `{ "error": "Job not found" }`. Invalid `job_config` returns 400 `{ "error": "job_config is invalid" }`.
- `POST` body is `{ "action": "shrink_lora" }`. One call inserts one `RuntimeCommand` row and does not write `RuntimeParams`. An action absent from `actions.json` returns 400 `{ "error": "unknown ui action" }`. Wrong process type returns 400 `{ "error": "ui actions require z_image_diffsynth_trainer" }`. Status other than `running`, or `stop` set, returns 409 `{ "error": "job is not running" }`. A body that is not JSON returns 400 `{ "error": "Invalid JSON body" }`.
- Drain selects `id, action` for this `jobId` ordered by `id`, applies the row, then deletes that id. The connection is not held across the multiply. An empty queue logs nothing and writes nothing.
- Drain runs only when `is_ui_trainer` is true and only on the main process, same gate as `_clear_runtime_columns`. A missing `RuntimeCommand` table is a no-op. A non-UI trainer is a no-op.
- `peft_dora` and no pairs write no weights, log the skip, and still delete the row.
- An unknown queued `action` raises `ValueError` with text `unknown util-ui action: {action}`. That row stays. Rows already applied in that drain are already deleted.

### Scale

- `scale_lora` builds pairs with `setup_stable_lora`. `peft_dora`, full-rank, frozen A, and unrecognized modules keep that function's skip rules.
- Each pair's A and each pair's B is multiplied by `0.9` through `shrink_matrix`. `apply_shrink` is not called.
- `pair.stable` on caller-owned pair objects is not written. `stable_lora.pt` is not read or written. Optimizer state is not read or written. Gradients are not scaled.
- fp32 uses in-place `mul_` under `torch.no_grad()`. Any other dtype goes through `shrink_matrix`. A non-fp32 matrix on CPU raises `RuntimeError` with text `stable_lora_shrink for non-fp32 weights requires CUDA (dtype={dtype}, device={device})`.
- Success log: `[zimage_diffsynth] lora_scale factor=0.9 pairs={n}`. Skip logs: `[zimage_diffsynth] lora_scale skipped: peft_dora` and `[zimage_diffsynth] lora_scale skipped: no LoRA A/B pairs`.

### Trainer hook

- `ZImageDiffSynthTrainer.end_step_hook` calls `super().end_step_hook()` and then `drain_runtime_commands`.
- `hook_before_optimizer_step` does not drain. It remains the Stable-LoRA path.

## 2. To do



## 3. Notes for continuing

### Limits

- Two clicks before the next step are two rows. Both matrices then sit at `0.81` of the value from before those clicks. The button title states the per-matrix factor.
- A crash after the multiply and before the delete applies that command again on the next step.
- One job is one process. A second consumer is outside this design.
- `RuntimeParams` stays a setpoint store.

### Known issue

- None.

### Testing

- `venv\Scripts\python.exe -m pytest extensions_built_in/diffusion_models/z_image_diffsynth/test_util_ui.py -q` reported `15 passed`. That covers both matrices, the stable flag, optimizer state, skip reasons, fp32, CPU bf16 refusal, two commands, an unknown action, an empty queue, a missing table, the non-UI and non-main gates, `peft_dora` consumption, `end_step_hook` order, and that `hook_before_optimizer_step` does not drain.
- From `ui/`, `npm test -- src/app/api/jobs/[jobID]/ui-actions/route.test.ts` reported `9 passed`.
- Browser, against `next dev` on port 3456 with a temporary running `z_image_diffsynth_trainer` job: the warning button was the leftmost control in the job bar, accessible name `Multiply LoRA A and B by 0.9`, and it sat left of pause. The first click disabled the button and inserted one `RuntimeCommand` row. The second click inserted a second row. `RuntimeParams` stayed empty. The jobs table did not show the button. The temporary job and both rows were deleted afterward. A live trainer step that consumes the rows and logs `factor=0.9` was not run.
