# Lightbox AI

A chat side panel in the dataset lightbox (`DatasetImageViewer` + `AgentChatPanel`). The browser talks only to this app. `POST /api/agent/chat` calls an OpenAI-compatible server (`OPENAI_API_URL`: `/models`, then streaming `/chat/completions`) and relays the reply as NDJSON.

Dev server, from `ui/`: `npx next dev --port 3000`.

Every plan based on this file must start with a first item that requires reading this file (`docs/lightbox-ai/README.md`) before any other step.

## File schema

Purpose: the single handoff document for continuing work on Lightbox AI. A reader must be able to resume work from this file alone.

Rules:

- Only facts that affect continuing the work: behavior, contracts, limits, open tasks, verified results, reusable artifacts. No background, motivation, history, or filler.
- Short declarative sentences. Bullets and tables over prose. No duplicated facts: each fact lives in exactly one section.
- Keep it current: when behavior changes, edit the existing entry. Remove entries that are no longer true. Do not append change logs.
- Every clarification is a contract in this file: a question the agent asked together with its answer, and every decision. Write it into the section the fact belongs to.
- Do not rename, reorder, or add top-level sections. Do not change the numbering.

Structure (top to bottom):


| Section                      | Contains                                                                                                                                                           | Does not contain                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| Intro (above)                | What the feature is, data flow in 2-3 sentences, dev server command, the plan rule                                                                                 | Details covered below                     |
| `## 1. Done`                 | Implemented behavior as a contract. Subsections: `Settings`, `Panel`, `Image and caption`, `API: POST /api/agent/chat` (request, response, timeouts, error table)  | Plans, limitations, test steps            |
| `## 2. To do`                | Unimplemented items, one bullet each, one line                                                                                                                     | Done items, ideas without a concrete task |
| `## 3. Notes for continuing` | Subsections: `Limits` (accepted constraints), `Known issue` (confirmed defects with exact error text), `Testing` (reference setup, what was verified, how to test) | Behavior already described in `1. Done`   |


Artifacts (commands, URLs, exact error strings, file paths, sample values) go inline in the section they belong to, in backticks. When an item moves from `2. To do` to done, delete it there and describe it in `1. Done`.

## 1. Done



### Settings

- `OPENAI_API_URL` (string, default empty) in the `settings` table. Edited on `/settings` as a plain text field. There is no upstream token.



### Panel

- A `w-80` column on the right, hidden below `md`. Enter sends, Shift+Enter is a newline. Send is enabled for non-empty trimmed text when nothing is in flight. While a request runs the button is Cancel and Enter does nothing.
- Shows user turns, "Thinking…" until the first delta, then the streamed reasoning and reply. Reasoning is a Thinking block: open while it streams, collapsed once the answer starts, toggleable. It stays in the panel and is never sent back.
- The reply and the Thinking block are rendered as GitHub-flavored markdown. Raw HTML is shown as text, and images are not rendered. User turns stay plain text.
- An empty reply is an `Empty reply` error.
- The conversation lives in panel memory only. The panel remounts on `key={imgPath}`, so each image has its own conversation. Closing the lightbox or switching images drops it and the draft.
- Cancel aborts the model reply. Cancel and a failed send both remove the unanswered user turn (roles stay alternating) and put that message into the textarea, replacing any text already there. Cancel shows no error. Failing or cancelling the first message also drops the stored image/caption pair. Unmount aborts the request silently.



### Image and caption

1. The first send stores `{ imgPath, caption }` in the panel. The caption is the lightbox field, including unsaved edits. An empty or not-yet-loaded caption is omitted.
2. Every request sends that pair plus the whole dialog in `messages`. The route stores nothing. It rebuilds only turn 0 from the pair. Later turns are plain strings.
3. If the loaded caption differs from the last one reflected in the dialog, the next user turn is prefixed with `Caption changed to:\n...` or `Caption removed.` The stored caption is not replaced. The reflected caption updates only after the reply is stored, so a failed send retries with the same notice.
4. Raster files (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`) become an `image_url` part with a base64 `data:` URL. The path and the `/api/img/` URL are not sent on. Video, audio, and `.svg` add no image part, and a caption is prefixed to the first turn as `Caption:\n...`.
5. First-turn order: `image_url`, caption text, user text.
6. `imgPath` must lie strictly inside `getDatasetsRoot()` (`path.resolve` + `path.relative`).



### API: `POST /api/agent/chat`

Request: `{ messages, imgPath?, caption? }`. `messages` is a non-empty array of `{ role: 'user' | 'assistant', content: string }`, trimmed, alternating, starting and ending with `user`. The model is the first `data[0].id` from `/models`.

Response: NDJSON (`application/x-ndjson`), one object per line:

- `{ "type": "reasoning", "delta" }` — from upstream `delta.reasoning_content`, or `delta.reasoning` when the first is absent. Inline `<think>` tags are not parsed.
- `{ "type": "content", "delta" }`
- `{ "type": "error", "message" }` — only after headers are sent. The stream then closes.
- A normal end is the stream closing. The panel counts a clean close with non-empty content as success.

Timeouts and disconnects:

- `/models`: 10s. `/chat/completions`: 120s idle timeout, reset on headers and on every chunk, so a long reasoning stream is not cut off.
- Client disconnect before upstream headers: upstream fetch aborted, 499 with an empty body, no error log. After headers: upstream read aborted, stream closed, no error event or log.
- A failure while reading the SSE body (invalid JSON, an upstream error payload, idle timeout) is an `error` event, not a new HTTP status.

Errors before the stream (JSON `{ error }`):


| Status | Message                                      | Cause                                                                        |
| ------ | -------------------------------------------- | ---------------------------------------------------------------------------- |
| 400    | `Invalid JSON`                               | body is not JSON                                                             |
| 400    | `Invalid messages`                           | invalid `messages`                                                           |
| 400    | `Invalid imgPath` / `Invalid caption`        | not a string                                                                 |
| 400    | `OpenAI API URL is not set (Settings)`       | empty URL                                                                    |
| 403    | `Access denied`                              | `imgPath` outside the datasets root, `..`, another drive, or the root itself |
| 404    | `File not found`                             | missing file or not a file                                                   |
| 413    | `Image is larger than 20 MB`                 | file over 20 MB                                                              |
| 499    | empty body                                   | client disconnected before chat headers                                      |
| 500    | `Failed to read image` / `Agent chat failed` | unexpected server failure                                                    |
| 502    | upstream error text                          | upstream failure or timeout                                                  |
| 502    | `No models available at the OpenAI API URL`  | empty `/models` list                                                         |




## 2. To do

- Model selection: a setting, and a vision model instead of the first one.
- An upstream API token (`Authorization` header).
- A control to start a new conversation without switching images.



## 3. Notes for continuing



### Limits

- The first model from `/models` is used. A model that cannot read images fails with the upstream error text.
- History is sent in full on every request and is not truncated. The image is re-read and re-sent every time. A context overflow is a 502 if the upstream rejects the request before the stream, or an `error` event if it fails after headers.
- The image and original caption stay on the first turn. Later caption edits reach the model only as a notice on the next user turn.
- A message sent before the caption has loaded is stored without a caption.



### Known issue

- `.webp` is sent as `image/webp`, and the LM Studio build used for testing rejects it (`'url' field must be a base64 encoded image.`). `.png` and `.jpeg` work.



### Testing

- Reference setup: LM Studio at `http://192.168.1.34:1234/v1`, model `huihui-qwen3.5-9b-abliterated`, image `datasets/dogs/000.png` with caption `dog`.
- Protocol checks (streaming not buffered, idle timeout, abort before/after headers, error table) are easiest against a local OpenAI mock. Restore the settings URL afterwards.
- The model does not need to be a reasoning model to test the panel, but the Thinking block only appears when upstream sends `reasoning_content` or `reasoning`.
- Manually checked in the lightbox against LM Studio: streaming with Thinking open then collapsed, follow-up turns without `reasoning` in `messages`, caption attached on the first turn only, Cancel during reasoning with the textarea empty and with text already typed, Cancel during content and on Thinking…, a later-turn Cancel, first-message Cancel followed by another send after a caption edit, failed-send rollback, closing the lightbox, Enter/ArrowRight/Escape in the field not closing or navigating the lightbox.
- Markdown in the reply and the Thinking block: bold, italic, strike, headings, lists, fenced code and tables scroll inside the `w-80` panel, links open in a new tab, raw HTML stays text, images are not requested. User turns stay literal.

