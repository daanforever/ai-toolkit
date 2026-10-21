import { readFile, stat } from 'fs/promises';
import path from 'path';
import { ChatMessage, ContentPart } from '@/server/openai';
import { isPathInsideRoot } from '@/server/safePath';
import { getDatasetsRoot } from '@/server/settings';
import { AgentChatTurn } from '@/types/agentChat';

export class AgentChatRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'AgentChatRequestError';
    this.status = status;
  }
}

const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

const RASTER_MIME: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
};

export interface ParsedAgentChatRequest {
  messages: AgentChatTurn[];
  imgPath?: string;
  caption?: string;
  reasoning?: boolean;
  tools?: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isRole(value: unknown): value is AgentChatTurn['role'] {
  return value === 'user' || value === 'assistant';
}

function errorCode(error: unknown): string | undefined {
  if (typeof error === 'object' && error !== null && 'code' in error && typeof error.code === 'string') {
    return error.code;
  }
  return undefined;
}

function parseMessages(value: unknown): AgentChatTurn[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;

  const messages: AgentChatTurn[] = [];
  for (const item of value) {
    if (!isRecord(item) || !isRole(item.role) || typeof item.content !== 'string') return null;
    const content = item.content.trim();
    if (!content) return null;
    const previous = messages[messages.length - 1];
    if (previous?.role === item.role) return null;
    messages.push({ role: item.role, content });
  }

  const first = messages[0];
  const last = messages[messages.length - 1];
  if (!first || first.role !== 'user' || !last || last.role !== 'user') return null;
  return messages;
}

export function parseAgentChatRequest(value: unknown): ParsedAgentChatRequest | null {
  if (!isRecord(value)) return null;
  if ('imgPath' in value && value.imgPath !== undefined && typeof value.imgPath !== 'string') {
    throw new AgentChatRequestError(400, 'Invalid imgPath');
  }
  if ('caption' in value && value.caption !== undefined && typeof value.caption !== 'string') {
    throw new AgentChatRequestError(400, 'Invalid caption');
  }
  if ('reasoning' in value && value.reasoning !== undefined && typeof value.reasoning !== 'boolean') {
    throw new AgentChatRequestError(400, 'Invalid reasoning');
  }
  if ('tools' in value && value.tools !== undefined && typeof value.tools !== 'boolean') {
    throw new AgentChatRequestError(400, 'Invalid tools');
  }

  const messages = parseMessages(value.messages);
  if (!messages) return null;

  const imgPath = typeof value.imgPath === 'string' ? value.imgPath.trim() : '';
  const caption = typeof value.caption === 'string' ? value.caption : undefined;
  const reasoning = typeof value.reasoning === 'boolean' ? value.reasoning : undefined;
  const tools = typeof value.tools === 'boolean' ? value.tools : undefined;
  return {
    messages,
    ...(imgPath ? { imgPath } : {}),
    ...(caption !== undefined ? { caption } : {}),
    ...(reasoning !== undefined ? { reasoning } : {}),
    ...(tools !== undefined ? { tools } : {}),
  };
}

function captionText(caption: string | undefined): string | null {
  const trimmed = caption?.trim() ?? '';
  if (!trimmed) return null;
  return `Caption:\n${trimmed}`;
}

function firstUserMessage(text: string, caption: string | undefined, dataUrl: string | null): ChatMessage {
  const labeledCaption = captionText(caption);
  if (!dataUrl) {
    return { role: 'user', content: labeledCaption ? `${labeledCaption}\n\n${text}` : text };
  }

  const parts: ContentPart[] = [{ type: 'image_url', image_url: { url: dataUrl } }];
  if (labeledCaption) parts.push({ type: 'text', text: labeledCaption });
  parts.push({ type: 'text', text });
  return { role: 'user', content: parts };
}

async function readImageDataUrl(imgPath: string): Promise<string | null> {
  const datasetsRoot = await getDatasetsRoot();
  if (!isPathInsideRoot(imgPath, datasetsRoot)) {
    throw new AgentChatRequestError(403, 'Access denied');
  }

  const mime = RASTER_MIME[path.extname(imgPath).toLowerCase()];
  if (!mime) return null;

  let size: number;
  try {
    const fileStat = await stat(imgPath);
    if (!fileStat.isFile()) {
      throw new AgentChatRequestError(404, 'File not found');
    }
    size = fileStat.size;
  } catch (error) {
    if (error instanceof AgentChatRequestError) throw error;
    if (errorCode(error) === 'ENOENT') {
      throw new AgentChatRequestError(404, 'File not found');
    }
    console.error('Agent chat failed to stat image:', error);
    throw new AgentChatRequestError(500, 'Failed to read image');
  }

  if (size > MAX_IMAGE_BYTES) {
    throw new AgentChatRequestError(413, 'Image is larger than 20 MB');
  }

  try {
    const bytes = await readFile(imgPath);
    return `data:${mime};base64,${bytes.toString('base64')}`;
  } catch (error) {
    if (errorCode(error) === 'ENOENT') {
      throw new AgentChatRequestError(404, 'File not found');
    }
    console.error('Agent chat failed to read image:', error);
    throw new AgentChatRequestError(500, 'Failed to read image');
  }
}

export async function buildChatMessages(request: ParsedAgentChatRequest): Promise<ChatMessage[]> {
  const dataUrl = request.imgPath ? await readImageDataUrl(request.imgPath) : null;
  return request.messages.map((turn, index) => {
    if (index === 0) return firstUserMessage(turn.content, request.caption, dataUrl);
    return { role: turn.role, content: turn.content };
  });
}
