# Lightbox AI

The dataset lightbox has a side panel for chatting with an OpenAI-compatible API. The browser talks only to this app. A server route reads `OPENAI_API_URL` and calls `/models` and `/chat/completions`.

Dev server, from `ui/`: `npx next dev --port 3000`.

## 1. Done

### Settings

- `OPENAI_API_URL` (string, default empty) is stored in the `settings` table as key/value. There is no token.
- `/settings` has an "OpenAI Compatible API URL" field (`type="text"`, no custom validation). An empty value is saved as an empty string.

### Panel (`DatasetImageViewer` + `AgentChatPanel`)

- Layout: a `w-80` column on the right, hidden below the `md` breakpoint. Media and caption are on the left. Close and prev/next stay on the media column.
- The column shows each sent message, then "Thinking…" until the first delta, then the reasoning and the reply as they arrive, or the error. Reasoning is a Thinking block: open while it streams, collapsed once the answer text starts. After that the user can toggle it. A finished turn keeps the collapsed block. Reasoning stays in the panel and is not sent back on the next request. A "Message" textarea at the bottom has a button inside the field, bottom right. Send becomes Cancel while a request is in flight.
- Send is enabled when the trimmed text is non-empty and nothing is in flight. Enter sends, Shift+Enter inserts a newline. The input clears when the request starts. Enter does nothing while a request is in flight.
- Cancel aborts the request, drops the unanswered user turn, restores the text to the input if the input is still empty, and shows no error. Cancelling the first message also drops the stored image and caption pair. Text typed into the field during the request is kept.
- The conversation lives in panel memory only. Closing the lightbox drops it. The panel remounts on `key={imgPath}`, so each image has its own conversation. Unmount aborts the in-flight browser request. That abort does not roll the text back or show an error, because the panel is already gone.
- A failed send removes the unanswered user turn and restores the text to the input if the input is still empty. Roles in the stored conversation stay alternating.

### API: `POST /api/agent/chat`

- Request: `{ messages, imgPath?, caption? }`. Response: NDJSON (`Content-Type: application/x-ndjson`), one JSON object per line. The upstream call uses `stream: true`. There is no `{ reply }` body.
- Events: `{ "type": "reasoning", "delta" }`, `{ "type": "content", "delta" }`, and, only after the response headers are sent, `{ "type": "error", "message" }`. A normal end is the stream closing. The panel treats a clean close with non-empty content as success.
- Reasoning deltas come from upstream `delta.reasoning_content`, or `delta.reasoning` when the first field is absent. Inline `<think>` tags are not parsed.
- `messages`: non-empty array of `{ role: 'user' | 'assistant', content: string }`. Content is trimmed. The first and last turns must be `user`, and roles must alternate.
- Model: the first `data[0].id` from `GET {url}/models`. There is no model setting.
- If the client disconnects before upstream chat headers, the route aborts the upstream `/models` or `/chat/completions` fetch and returns 499 with an empty body and no error log. A timeout while waiting for those headers stays 502. After headers, a disconnect aborts the upstream read and closes the NDJSON stream with no error event and no error log.
- A failure while reading the upstream SSE body (invalid JSON, upstream error payload, or 120s with no chunk) is an `error` event, then the stream closes. It is not a new HTTP status. The 120s completion limit is an idle timeout: it resets when headers arrive and on every chunk, so a long reasoning stream is not cut off at 120s of total time. `/models` stays 10s.

| Status | Message | Cause |
|---|---|---|
| 400 | `Invalid messages` | invalid `messages` |
| 400 | `Invalid imgPath` / `Invalid caption` | not a string |
| 400 | `OpenAI API URL is not set (Settings)` | empty URL |
| 403 | `Access denied` | `imgPath` outside the datasets root, `..`, another drive, or the root itself |
| 404 | `File not found` | missing file or not a file |
| 413 | `Image is larger than 20 MB` | file over 20 MB |
| 499 | empty body | client disconnected before chat response headers; upstream fetch aborted |
| 502 | upstream error text | upstream failure or timeout |
| 502 | `No models available at the OpenAI API URL` | empty `/models` list |

### Image and caption attachment

1. The open image supplies a file path. The caption is the lightbox field, including unsaved edits.
2. The first send stores that pair in the panel. An empty caption, or one that has not loaded yet, is left out. A failed first send drops the pair, so the retry reads the field again.
3. Each later send repeats the stored pair, which still rebuilds turn 0, and sends `messages`, the whole dialog. If the loaded caption differs from the one last reflected in the dialog, that user turn is prefixed with `Caption changed to:\n...` or `Caption removed.` The stored caption is not replaced. A caption that has not loaded yet adds no notice. The reflected caption updates only after the reply is stored, so a failed send retries with the same notice.
4. The route stores nothing. It checks the path stays inside the datasets root, reads the file, and rebuilds only the first user turn. Later turns stay the plain strings from `messages`.
5. A raster file (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`) becomes an `image_url` part whose `url` is a base64 `data:` URL. The file path and the `/api/img/` URL are not sent on. Video, audio, and `.svg` add no image part; a stored caption is then prefixed to the first turn as `Caption:\n...`.
6. First-turn order: `image_url` when present, then the caption text, then the user text.

### Files

New:

- `ui/src/types/agentChat.ts` — `AgentChatTurn`, `AgentChatRequest`, `AgentChatStreamEvent`, and panel-only `AgentChatDisplayTurn` (`reasoning` is not part of the request)
- `ui/src/server/openai.ts` — `ContentPart`, `ChatMessage`, `resolveModel(signal)`, `streamChatCompletion(messages, signal)`. `/models` uses `AbortSignal.any` with the caller signal and a 10s timeout. `/chat/completions` uses `stream: true`, a 120s timeout until headers, then the same 120s as an idle timeout between chunks. The returned async iterable yields `{ type: 'reasoning' | 'content', delta }`.
- `ui/src/server/agentChat.ts` — validates the body, checks `imgPath`, reads the image, builds `ChatMessage[]`
- `ui/src/app/api/agent/chat/route.ts` — awaits upstream headers, then returns an NDJSON `ReadableStream`. Pre-stream failures stay JSON, including 499 when the client aborts. Mid-stream failures are `error` events. `cancel()` on the stream aborts the upstream read.

Edited:

- `ui/src/server/settings.ts` — `getOpenAIApiUrl`, cached like `getHFToken`. `flushCache` on settings save clears it
- `ui/src/app/api/settings/route.ts` — POST writes the keys in `SETTING_KEYS`, including `OPENAI_API_URL`. GET is unchanged
- `ui/src/hooks/useSettings.tsx` — `OPENAI_API_URL` on the type, the default, and the load mapping
- `ui/src/app/settings/page.tsx` — field after Dataset Folder Path
- `ui/src/components/AgentChatPanel.tsx` — conversation, in-flight state, streaming reasoning and reply, error, attachment snapshot. A per-request `AbortController`, aborted on Cancel and on unmount, passed to a `fetch` of the NDJSON stream. The auth header comes from the same helper as `apiClient`.
- `ui/src/components/DatasetImageViewer.tsx` — horizontal layout, panel remount on `imgPath` change

## 2. To do

- Model selection (a setting, and a vision model instead of the first one).
- An API token (`Authorization` header).
- A control to start a new conversation without switching images.

## 3. Notes for continuing

### Limits

- No API token. The upstream request has no `Authorization` header.
- The first model from `/models` is used. A model that cannot read images fails as 502 with the upstream error text.
- The image and the original caption stay on the first turn. A later edit does not rewrite that turn. The next user turn carries a notice of the new caption, or of its removal.
- History is kept in the panel only and is sent in full on every request. It is not truncated. A context overflow comes back as 502 with the upstream error text when the upstream rejects the request before the stream, or as an `error` event when it fails after headers. The image is re-read and re-sent on every request.
- A message sent before the caption has loaded is stored without a caption.
- Switching images drops the draft in the message field.
- Reasoning from `reasoning_content` or `reasoning` is shown in the panel and kept after the reply, but it is not sent back in `messages`. Inline `<think>` tags are not parsed.

### Known issue

- `.webp` is sent as `image/webp`. The LM Studio build used for the check rejects that data URL (`'url' field must be a base64 encoded image.`). `.png` and `.jpeg` are accepted.

### Verification

Setup: LM Studio at `http://192.168.1.34:1234/v1`, model `huihui-qwen3.5-9b-abliterated`, dataset image `datasets/dogs/000.png` (caption `dog`).

API (local OpenAI mock, then the settings URL restored):

- `stream: true`. NDJSON lines arrived as they were produced (a reasoning line, then a content line after a gap), not buffered until the upstream stream ended.
- Image + caption: first turn was `image_url` (`data:image/png;base64,`), `Caption:\ndog`, then the question. Later turns stayed strings. A `reasoning` key on a stored turn was not forwarded.
- Errors matched the table: 400 for an empty URL, invalid `messages`, and non-string `imgPath` / `caption`; 403 for path escapes; 404 for a missing file; 413 for a file over 20 MB; 502 for an empty `/models` list.
- Unreachable host: 502 `The operation was aborted due to timeout` after about 10s (`/models`). Idle timeout on chat headers and on a gap between chunks was about 120s.
- Client abort before headers: 499, empty body. Client abort after the first NDJSON byte closed the upstream read with no completion-failure log.

Lightbox (`000.png`, caption `dog`):

- The first request sent `imgPath`, `caption: "dog"`, and one `{ role, content }` turn. Thinking stayed open and grew while reasoning streamed, then the answer `Pong` was stored with Thinking collapsed. Opening Thinking again showed the reasoning.
- A follow-up sent three `{ role, content }` turns. The assistant turn was the answer `Pong` only, with no `reasoning` field. The reply was `Ping`. The new Thinking block was collapsed after the answer.
- Cancel during reasoning dropped that user turn, restored the draft, showed no error, and left the earlier answers in place.
