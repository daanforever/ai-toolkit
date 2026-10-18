import { ChatStreamEvent, ChatTool, OpenAIRequestError } from '@/server/openai';

export const EDIT_TOOL_NAME = 'edit';

export const EDIT_TOOL: ChatTool = {
  type: 'function',
  function: {
    name: EDIT_TOOL_NAME,
    description:
      'Replace the caption of the current image with the full new caption. Do not call edit unless the user explicitly asks to change, rewrite, or write the caption.',
    parameters: {
      type: 'object',
      properties: {
        caption: {
          type: 'string',
          description: 'The full new caption.',
        },
      },
      required: ['caption'],
    },
  },
};

export type CollectedChatEvent =
  | { type: 'reasoning'; delta: string }
  | { type: 'content'; delta: string }
  | { type: 'edit'; caption: string };

interface ToolCall {
  name?: string;
  arguments: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function appendToolCall(calls: Map<number, ToolCall>, event: Extract<ChatStreamEvent, { type: 'tool_call' }>): void {
  const call = calls.get(event.index) ?? { arguments: '' };
  if (event.name) {
    if (call.name && call.name !== event.name) {
      console.error('Tool call name changed:', call.name, event.name);
      throw new OpenAIRequestError(502, 'Invalid tool call');
    }
    call.name = event.name;
  }
  call.arguments += event.arguments;
  calls.set(event.index, call);
}

export function resolveEditCaption(calls: Map<number, ToolCall>): string | null {
  if (calls.size === 0) return null;

  let caption: string | null = null;
  for (const [, call] of [...calls.entries()].sort((left, right) => left[0] - right[0])) {
    if (call.name !== EDIT_TOOL_NAME) {
      console.error('Unknown tool:', call.name);
      throw new OpenAIRequestError(502, 'Unknown tool');
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(call.arguments);
    } catch (error) {
      console.error('Invalid tool arguments:', error);
      throw new OpenAIRequestError(502, 'Invalid tool arguments');
    }
    if (!isRecord(parsed) || typeof parsed.caption !== 'string' || parsed.caption.trim() === '') {
      console.error('Invalid tool arguments:', parsed);
      throw new OpenAIRequestError(502, 'Invalid tool arguments');
    }
    caption = parsed.caption.trim();
  }
  return caption;
}

// Tool fragments are held until the upstream stream ends. A throw or an early
// return never emits `edit`, so a cancelled reply cannot apply a partial caption.
export async function* collectEditCalls(events: AsyncIterable<ChatStreamEvent>): AsyncGenerator<CollectedChatEvent> {
  const calls = new Map<number, ToolCall>();
  for await (const event of events) {
    if (event.type === 'tool_call') {
      appendToolCall(calls, event);
      continue;
    }
    yield event;
  }
  const caption = resolveEditCaption(calls);
  if (caption !== null) yield { type: 'edit', caption };
}
