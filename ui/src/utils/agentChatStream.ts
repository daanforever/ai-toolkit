import { AgentChatStreamEvent } from '@/types/agentChat';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function parseAgentChatStreamLine(line: string): AgentChatStreamEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch (error) {
    console.error('Agent chat stream line was invalid JSON:', error);
    return null;
  }
  if (!isRecord(value)) {
    console.error('Agent chat stream line was invalid:', value);
    return null;
  }
  if ((value.type === 'reasoning' || value.type === 'content') && typeof value.delta === 'string') {
    return { type: value.type, delta: value.delta };
  }
  if (value.type === 'error' && typeof value.message === 'string') {
    return { type: 'error', message: value.message };
  }
  if (value.type === 'edit' && typeof value.caption === 'string') {
    return { type: 'edit', caption: value.caption };
  }
  console.error('Agent chat stream line was invalid:', value);
  return null;
}

export function assistantTurnContent(content: string, caption?: string): string {
  if (!caption) return content;
  const note = `Caption updated to:\n${caption}`;
  if (content.trim() === '') return note;
  return `${content.trimEnd()}\n\n${note}`;
}
