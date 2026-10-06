# Lightbox AI (MVP)

The lightbox panel sends one message to an OpenAI-compatible API and shows the reply. The browser talks only to this app. A server route reads `OPENAI_API_URL` and calls `/models` and `/chat/completions`.

## What shipped

- Setting `OPENAI_API_URL` (string, default empty). Stored in the `settings` table as key/value. No token.
- `/settings`: an "OpenAI Compatible API URL" field (`type="text"`, no custom validation). An empty value is saved as an empty string.
- Dataset lightbox (`DatasetImageViewer`): a `w-80` column on the right. It shows the current exchange (the sent message, then "Thinking…" or the reply or the error). A "Message" textarea sits at the bottom with a Send icon inside the field, bottom right. Send is enabled when the trimmed text is non-empty and nothing is in flight. Enter sends. Shift+Enter inserts a newline. The input clears when the request starts. The column is hidden below the `md` breakpoint.
- `POST /api/agent/chat` with `{ message: string }`. Empty or non-string `message` is 400. An empty URL is 400 `OpenAI API URL is not set (Settings)`. The route builds one user message with a string `content`, calls the upstream API, and returns `{ reply }`. Upstream failures are 502 with the upstream error text.
- Model choice: `GET {url}/models` and the first `data[0].id`. An empty list is 502 `No models available at the OpenAI API URL`. No model setting.
- Non-streaming (`stream: false`). No message history. One exchange at a time. The panel does not know about the image.

## Files

New:

- `ui/src/types/agentChat.ts` — `AgentChatRequest { message: string }`, shared by the panel and the route. Later fields (`imgPath`, `caption`) go here.
- `ui/src/server/openai.ts` — `ContentPart`, `ChatMessage`, `resolveModel`, `createChatCompletion`. Timeouts: 10s for `/models`, 120s for `/chat/completions`.
- `ui/src/app/api/agent/chat/route.ts` — validates the body, builds the messages, returns `{ reply }`

Edited:

- `ui/src/server/settings.ts` — `getOpenAIApiUrl`, cached like `getHFToken`. `flushCache` on settings save clears it
- `ui/src/components/AgentChatPanel.tsx` — send, in-flight state, reply, and error. Still no image and no settings read
- `ui/src/hooks/useSettings.tsx` — `OPENAI_API_URL` on the type, the default, and the load mapping
- `ui/src/app/api/settings/route.ts` — POST writes the keys in `SETTING_KEYS`, including `OPENAI_API_URL`. GET is unchanged
- `ui/src/app/settings/page.tsx` — field after Dataset Folder Path
- `ui/src/components/DatasetImageViewer.tsx` — horizontal layout: media and caption on the left, panel on the right. Close and prev/next stay on the media column

## Limits

- No API token. The upstream request has no `Authorization` header.
- The first model from `/models` is used. A vision model is not selected on purpose.
- One turn. Sending again replaces the previous exchange. The exchange stays while the lightbox stays open, including when the image changes.
- Replies are plain text. `reasoning_content` from the upstream payload is ignored.

## Checked on 2026-10-06

Server: `http://192.168.1.34:1234/v1` (LM Studio). `GET /v1/models` returned `qwen/qwen3-vl-8b`.

- `POST /api/agent/chat` with `Reply with the single word pong.` returned `{ "reply": "pong" }` in about 0.4s (model already warm). The lightbox showed that reply.
- Unset URL: 400 `OpenAI API URL is not set (Settings)`.
- Empty or non-string `message`: 400 `Message is required`.
- Unreachable host `http://192.168.1.99:1234/v1`: 502 `The operation was aborted due to timeout` in about 10s. The panel showed that text, with Send disabled and the input cleared while it was in flight. A second Enter during that request did not start a second call.
- ArrowRight and Escape inside the message field left the lightbox on the same image. Enter in the caption field did not close it.

A cold call to this model, before these checks, took about 39s. The completion timeout is 120s so that wait is not treated as a failure.

## Next: image and caption in the request

Not built. The types are ready so the next change stays in message building.

- `ChatMessage.content` is `string | ContentPart[]`. `ContentPart` is `{ type: 'text'; text: string }` or `{ type: 'image_url'; image_url: { url: string } }`. Nothing constructs an `image_url` part yet. Text requests still send a string.
- The client will send `imgPath` on `AgentChatRequest` (same idea as `caption/get`). The server checks it against `getDatasetsRoot()` (same check as `api/caption/get`), reads the file, and sends a base64 `data:` URL in an `image_url` part. Do not send the Next.js `/api/img/` URL: the model server cannot be assumed to reach it or to pass auth.
- The caption is another `text` part. Still open: read the `.txt` on the server, or send the client's current unsaved text.
- `qwen/qwen3-vl-8b` can take images. "First model from `/models`" can still pick a model that cannot. A model setting or selector may be required then.

## Where to continue

- Image and caption in the request, as above.
- Then history, streaming, model selection, and a token.
