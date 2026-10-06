# Lightbox AI (MVP)

UI groundwork for an OpenAI-compatible API. There are no API calls yet: the URL is only stored, and the lightbox panel is a placeholder.

## What shipped

- Setting `OPENAI_API_URL` (string, default empty). Stored in the `settings` table as key/value. No token.
- `/settings`: an "OpenAI Compatible API URL" field (`type="text"`, no custom validation). An empty value is saved as an empty string.
- Dataset lightbox (`DatasetImageViewer`): a `w-80` column on the right. Empty state on top ("AI agent — coming soon"), a "Message" textarea at the bottom with a disabled Send icon inside the field, bottom right. Submit does not send anything. The column is hidden below the `md` breakpoint.

## Files

New:

- `ui/src/components/OpenAIApiUrlField.tsx` — URL field on `/settings`
- `ui/src/components/AgentChatPanel.tsx` — the panel. It does not know about the image and makes no network calls

Edited:

- `ui/src/hooks/useSettings.tsx` — `OPENAI_API_URL` on the type, the default, and the load mapping
- `ui/src/app/api/settings/route.ts` — POST writes the keys in `SETTING_KEYS`, including `OPENAI_API_URL`. GET is unchanged
- `ui/src/app/settings/page.tsx` — field after Dataset Folder Path
- `ui/src/components/DatasetImageViewer.tsx` — horizontal layout: media and caption on the left, panel on the right. Close and prev/next stay on the media column

There is no server getter for the URL. Nothing reads it except the settings form.

## Where to continue

- Dialog and send: `AgentChatPanel`. The textarea can be focused (arrow keys and Escape inside it do not change the image or close the lightbox — `isEditableTarget` in `DatasetImageViewer` already handles that). There is no message history.
- API calls: read `OPENAI_API_URL` from `useSettings` on the client, or add a getter next to `getHFToken` in `ui/src/server/settings.ts`.
- A token and other API parameters were left out of this MVP.
