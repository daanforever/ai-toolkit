# Lightbox AI

The dataset lightbox has a side panel for chatting with an OpenAI-compatible API. The browser talks only to this app. A server route reads `OPENAI_API_URL` and calls `/models` and `/chat/completions`.

Dev server, from `ui/`: `npx next dev --port 3000`.

## 1. Done

### Settings

- `OPENAI_API_URL` (string, default empty) is stored in the `settings` table as key/value. There is no token.
- `/settings` has an "OpenAI Compatible API URL" field (`type="text"`, no custom validation). An empty value is saved as an empty string.

### Panel (`DatasetImageViewer` + `AgentChatPanel`)

- Layout: a `w-80` column on the right, hidden below the `md` breakpoint. Media and caption are on the left. Close and prev/next stay on the media column.
- The column shows each sent message, then "Thinking…", the reply, or the error. A "Message" textarea at the bottom has a Send icon inside the field, bottom right.
- Send is enabled when the trimmed text is non-empty and nothing is in flight. Enter sends, Shift+Enter inserts a newline. The input clears when the request starts.
- The conversation lives in panel memory only. Closing the lightbox drops it. The panel remounts on `key={imgPath}`, so each image has its own conversation. Unmount aborts the in-flight browser request. A canceled request does not roll back into the field or show an error, because the panel is already gone.
- A failed send removes the unanswered user turn and restores the text to the input if the input is still empty. Roles in the stored conversation stay alternating.

### API: `POST /api/agent/chat`

- Request: `{ messages, imgPath?, caption? }`. Response: `{ reply }`. Non-streaming (`stream: false`).
- `messages`: non-empty array of `{ role: 'user' | 'assistant', content: string }`. Content is trimmed. The first and last turns must be `user`, and roles must alternate.
- Model: the first `data[0].id` from `GET {url}/models`. There is no model setting.
- If the client disconnects, the route aborts the upstream `/models` or `/chat/completions` fetch and returns 499 with an empty body and no error log. A timeout stays 502.

| Status | Message | Cause |
|---|---|---|
| 400 | `Invalid messages` | invalid `messages` |
| 400 | `Invalid imgPath` / `Invalid caption` | not a string |
| 400 | `OpenAI API URL is not set (Settings)` | empty URL |
| 403 | `Access denied` | `imgPath` outside the datasets root, `..`, another drive, or the root itself |
| 404 | `File not found` | missing file or not a file |
| 413 | `Image is larger than 20 MB` | file over 20 MB |
| 499 | empty body | client disconnected; upstream fetch aborted |
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

- `ui/src/types/agentChat.ts` — `AgentChatTurn`, `AgentChatRequest`, shared by the panel and the route
- `ui/src/server/openai.ts` — `ContentPart`, `ChatMessage`, `resolveModel(signal)`, `createChatCompletion(messages, signal)`. Each fetch uses `AbortSignal.any` with the caller signal and the timeout (10s for `/models`, 120s for `/chat/completions`)
- `ui/src/server/agentChat.ts` — validates the body, checks `imgPath`, reads the image, builds `ChatMessage[]`
- `ui/src/app/api/agent/chat/route.ts` — returns `{ reply }`, or 499 when `request.signal` aborts

Edited:

- `ui/src/server/settings.ts` — `getOpenAIApiUrl`, cached like `getHFToken`. `flushCache` on settings save clears it
- `ui/src/app/api/settings/route.ts` — POST writes the keys in `SETTING_KEYS`, including `OPENAI_API_URL`. GET is unchanged
- `ui/src/hooks/useSettings.tsx` — `OPENAI_API_URL` on the type, the default, and the load mapping
- `ui/src/app/settings/page.tsx` — field after Dataset Folder Path
- `ui/src/components/AgentChatPanel.tsx` — conversation, in-flight state, reply, error, attachment snapshot. `AbortController` per mount, aborted on unmount, passed to the chat request
- `ui/src/components/DatasetImageViewer.tsx` — horizontal layout, panel remount on `imgPath` change

## 2. To do

- Streaming.
- Model selection (a setting, and a vision model instead of the first one).
- An API token (`Authorization` header).
- A control to start a new conversation without switching images.

## 3. Notes for continuing

### Limits

- No API token. The upstream request has no `Authorization` header.
- The first model from `/models` is used. A model that cannot read images fails as 502 with the upstream error text.
- The image and the original caption stay on the first turn. A later edit does not rewrite that turn. The next user turn carries a notice of the new caption, or of its removal.
- History is kept in the panel only and is sent in full on every request. It is not truncated. A context overflow comes back as 502 with the upstream error text. The image is re-read and re-sent on every request.
- A message sent before the caption has loaded is stored without a caption.
- Switching images drops the draft in the message field.
- Replies are plain text. `reasoning_content` from the upstream payload is ignored.

### Known issue

- `.webp` is sent as `image/webp`. The LM Studio build used for the check rejects that data URL (`'url' field must be a base64 encoded image.`). `.png` and `.jpeg` are accepted.

### Verification

Setup: LM Studio at `http://192.168.1.34:1234/v1`, model `qwen/qwen3-vl-8b`, dataset image `datasets/dogs/000.png` (a brown and white dog, caption `dog`).

API:

- A single user turn returned `{ "reply": "pong" }`.
- Three turns with `imgPath` and `caption: "dog"`: the first turn was `image_url` (`data:image/png;base64,`), `Caption:\ndog`, then the question. Later turns stayed strings. The model recognised the dog and the caption, and answered a follow-up about fur colour.
- The model remembered a word from an earlier user turn (`lantern`).
- `.jpeg` and uppercase `.PNG` were attached. `.mp4`, `.mp3`, and `.svg` were text only (a caption was prefixed on the mp4 turn). A caption without `imgPath` was quoted back.
- Errors matched the table above: 400 for invalid `messages` and non-string `imgPath` / `caption`, 403 for path escapes (outside root, `..`, another drive, the root itself), 404 for a missing file and a directory named `*.png`, 413 for a file of 20 MB + 1 byte.
- Unreachable host `http://192.168.1.99:1234/v1`: 502 `The operation was aborted due to timeout` after about 10s. That timeout is still logged. A client abort is not.
- Aborting the client closed the upstream socket during `/models` and during `/chat/completions`. A completed call still returned `{ "reply": "pong" }`.
- LM Studio, `000.png`: a short turn returned `pong` in about 1.3s. A long turn aborted at 8s, and the next short turn returned `pong` in about 1.9s.

Lightbox:

- The first request carried `imgPath` of `000.png` and `caption: "dog"`. A follow-up without an edit sent only the typed text, and `caption` stayed `dog`.
- Editing the caption to `a purple elephant` and sending again prefixed that user turn with `Caption changed to:\na purple elephant`. `caption` on the request stayed `dog`. The reply was `a purple elephant`.
- On the next image, clearing the caption prefixed the following turn with `Caption removed.` `caption` stayed `dog`. The reply said the caption was removed.
- With the API URL cleared, a send that included `Caption changed to:\na cat` returned 400, the turn rolled back, and the draft returned. After restoring the URL, the retry still started with that notice. The reply treated `dog` as the original caption and `a cat` as the change.
- Next, previous, and Close abort an in-flight request and clear the conversation. Returning to the same image starts empty, and a new send works. A send immediately after open returned `pong`.
- ArrowRight and Escape in the message field did not leave the image. Enter sent without closing the lightbox.
- A failed send removed the user turn and put the text back in the field. With the API URL cleared, the same rollback still happened.
