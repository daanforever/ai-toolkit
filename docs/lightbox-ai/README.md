# Lightbox AI

The lightbox panel keeps a conversation with an OpenAI-compatible API and shows each reply. The browser talks only to this app. A server route reads `OPENAI_API_URL` and calls `/models` and `/chat/completions`.

## What shipped

- Setting `OPENAI_API_URL` (string, default empty). Stored in the `settings` table as key/value. No token.
- `/settings`: an "OpenAI Compatible API URL" field (`type="text"`, no custom validation). An empty value is saved as an empty string.
- Dataset lightbox (`DatasetImageViewer`): a `w-80` column on the right. It shows the conversation (each sent message, then "Thinking…" or the reply or the error). A "Message" textarea sits at the bottom with a Send icon inside the field, bottom right. Send is enabled when the trimmed text is non-empty and nothing is in flight. Enter sends. Shift+Enter inserts a newline. The input clears when the request starts. The column is hidden below the `md` breakpoint.
- The conversation lives in the panel's memory. Closing the lightbox drops it. Changing the image remounts the panel (`key={imgPath}`), so each image has its own conversation and an in-flight reply from the previous image is discarded.
- A failed send rolls the unanswered user turn back out of the conversation and restores the text to the input when the input is still empty. Roles in the stored conversation stay alternating.
- `POST /api/agent/chat` with `{ messages, imgPath?, caption? }`. `messages` is a non-empty array of `{ role: 'user' | 'assistant', content: string }`. Content is trimmed. The first and last turns must be `user`, and roles must alternate. Anything else is 400 `Invalid messages`. An empty URL is 400 `OpenAI API URL is not set (Settings)`. The route returns `{ reply }`. Upstream failures are 502 with the upstream error text.
- `imgPath` and `caption`, when present, must be strings (otherwise 400 `Invalid imgPath` / `Invalid caption`). The panel sends them on every request of a conversation, from a snapshot taken at the first send. The server attaches them only to the first user turn, so a stateless call still rebuilds that turn. Later turns stay plain strings. An empty caption is omitted.
- The caption snapshot is the lightbox field at the first send, including unsaved edits. Until the caption has loaded, the panel sends no `caption`. Edits after the first send are not sent again. A failed first send drops the snapshot, so the retry reads the field again.
- `imgPath` is checked against `getDatasetsRoot()` with `path.resolve` and `path.relative` (a different drive, `..`, or the root directory itself is 403 `Access denied`). The server reads the file and sends a base64 `data:` URL. It does not send the Next.js `/api/img/` URL. Missing files and non-files are 404 `File not found`. Files larger than 20 MB are 413 `Image is larger than 20 MB`.
- Raster types only: `.jpg` / `.jpeg` (`image/jpeg`), `.png`, `.gif`, `.webp`, `.bmp`. Video, audio, and `.svg` are not attached; the first turn is still text, and a caption is prefixed as `Caption:\n...`. Image parts, when present, are ordered `image_url`, then the caption text, then the user text.
- Model choice: `GET {url}/models` and the first `data[0].id`. An empty list is 502 `No models available at the OpenAI API URL`. No model setting.
- Non-streaming (`stream: false`). History is not truncated.

## Files

New:

- `ui/src/types/agentChat.ts` — `AgentChatTurn`, `AgentChatRequest { messages, imgPath?, caption? }`, shared by the panel and the route
- `ui/src/server/openai.ts` — `ContentPart`, `ChatMessage`, `resolveModel`, `createChatCompletion`. Timeouts: 10s for `/models`, 120s for `/chat/completions`
- `ui/src/server/agentChat.ts` — validates the body, checks `imgPath`, reads the image, builds `ChatMessage[]`
- `ui/src/app/api/agent/chat/route.ts` — returns `{ reply }`

Edited:

- `ui/src/server/settings.ts` — `getOpenAIApiUrl`, cached like `getHFToken`. `flushCache` on settings save clears it
- `ui/src/components/AgentChatPanel.tsx` — conversation, in-flight state, reply, error, and the attachment snapshot
- `ui/src/hooks/useSettings.tsx` — `OPENAI_API_URL` on the type, the default, and the load mapping
- `ui/src/app/api/settings/route.ts` — POST writes the keys in `SETTING_KEYS`, including `OPENAI_API_URL`. GET is unchanged
- `ui/src/app/settings/page.tsx` — field after Dataset Folder Path
- `ui/src/components/DatasetImageViewer.tsx` — horizontal layout: media and caption on the left, panel on the right. The panel remounts when `imgPath` changes. Close and prev/next stay on the media column

## Limits

- No API token. The upstream request has no `Authorization` header.
- The first model from `/models` is used. A vision model is not selected on purpose. A model that cannot read images fails as 502 with the upstream error text.
- The image and caption are attached only on the first turn. Changing the caption later in the same conversation does not update what the model sees. Switch images and come back, or start a new conversation once that exists.
- History is kept in the panel only, is sent in full on every request, and is not truncated. A context overflow comes back as 502 with the upstream error text. The image is re-read and re-sent on every request of that conversation.
- A message sent before the caption has loaded is stored without a caption.
- Switching images drops the draft in the message field.
- Replies are plain text. `reasoning_content` from the upstream payload is ignored.
- `.webp` is sent as `image/webp`. The LM Studio build used for the check rejects that data URL (`'url' field must be a base64 encoded image.`). `.png` and `.jpeg` are accepted.

## Checked on 2026-10-06

Server: `http://192.168.1.34:1234/v1` (LM Studio), model `qwen/qwen3-vl-8b`. Dataset `datasets/dogs/000.png` (a brown and white dog, caption `dog`).

- One user turn `Reply with the single word pong.` returned `{ "reply": "pong" }`.
- Three turns with `imgPath` and `caption: "dog"`: the first turn was `image_url` (`data:image/png;base64,`), `Caption:\ndog`, then the question. Later turns stayed strings. The reply was `animal=dog; caption=dog`. A follow-up, `What colour is the fur?`, returned `Tan and white` / `Brown and white`.
- A second user turn that only asked for a remembered word returned `lantern`.
- `.jpeg` and an uppercase `.PNG` were attached and returned `pong`. `.mp4`, `.mp3`, and `.svg` were text only (a caption was prefixed on the mp4 turn) and returned `pong`. Caption without `imgPath` was quoted back as `frozen caption`.
- Invalid `messages`, non-string `imgPath` / `caption`: 400. Path outside the datasets root, `..`, another drive, or the root itself: 403. Missing file and a directory named `*.png`: 404. A file of 20 MB + 1 byte: 413.
- Unreachable host `http://192.168.1.99:1234/v1`: 502 `The operation was aborted due to timeout` in about 10s.
- In the lightbox, the first request carried `imgPath` of `000.png` and `caption: "dog"`. Editing the caption to `a purple elephant` afterwards did not change later requests; the model quoted `dog`. Next/previous cleared the conversation. ArrowRight and Escape in the message field left the lightbox on the same image. Enter sent and did not close it. A failed send removed the user turn and put the text back in the field.

## Where to continue

- Streaming.
- Model selection.
- An API token.
- A control to start a new conversation without switching images.
