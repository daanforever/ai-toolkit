import { getOpenAIApiUrl, getOpenAIDefaultModel } from '@/server/settings';

export type ContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | ContentPart[];
}

export type ChatStreamEvent = { type: 'reasoning'; delta: string } | { type: 'content'; delta: string };

export class OpenAIRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'OpenAIRequestError';
    this.status = status;
  }
}

export class OpenAIRequestAborted extends Error {
  constructor() {
    super('Client aborted the request');
    this.name = 'OpenAIRequestAborted';
  }
}

const MODELS_TIMEOUT_MS = 10_000;
const COMPLETION_TIMEOUT_MS = 120_000;

interface ModelsResponse {
  data?: { id?: string }[];
}

interface IdleAbort {
  readonly signal: AbortSignal;
  reset(): void;
  dispose(): void;
}

async function requireBaseUrl(): Promise<string> {
  const apiUrl = (await getOpenAIApiUrl()).trim().replace(/\/+$/, '');
  if (!apiUrl) {
    throw new OpenAIRequestError(400, 'OpenAI API URL is not set (Settings)');
  }
  return apiUrl;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  if (!(error instanceof Error) || !error.message) {
    return 'Upstream request failed';
  }
  const cause = error.cause;
  if (cause instanceof Error && cause.message && cause.message !== error.message) {
    return `${error.message}: ${cause.message}`;
  }
  return error.message;
}

function upstreamSignal(signal: AbortSignal, timeoutMs: number): AbortSignal {
  return AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
}

function rethrowUpstream(signal: AbortSignal, error: unknown, label: string): never {
  if (signal.aborted) {
    throw new OpenAIRequestAborted();
  }
  console.error(label, error);
  throw new OpenAIRequestError(502, errorMessage(error));
}

function timeoutError(cause?: unknown): OpenAIRequestError {
  console.error('OpenAI chat completion failed:', cause ?? 'idle timeout');
  return new OpenAIRequestError(502, 'The operation was aborted due to timeout');
}

function errorText(body: Record<string, unknown>): string | null {
  if (!isRecord(body.error) || typeof body.error.message !== 'string' || !body.error.message) return null;
  return body.error.message;
}

async function readUpstreamError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (isRecord(body)) {
      const message = errorText(body);
      if (message) return message;
    }
  } catch (error) {
    console.error('OpenAI API returned a non-JSON error body:', error);
  }
  return response.statusText || `Upstream request failed (${response.status})`;
}

function createIdleAbort(timeoutMs: number): IdleAbort {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;

  const arm = () => {
    if (controller.signal.aborted) return;
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(() => {
      controller.abort(new DOMException('The operation was aborted due to timeout', 'TimeoutError'));
    }, timeoutMs);
  };

  arm();

  return {
    signal: controller.signal,
    reset() {
      arm();
    },
    dispose() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
    },
  };
}

export async function resolveModel(signal: AbortSignal): Promise<string> {
  const apiUrl = await requireBaseUrl();
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/models`, { signal: upstreamSignal(signal, MODELS_TIMEOUT_MS) });
  } catch (error) {
    rethrowUpstream(signal, error, 'OpenAI models request failed:');
  }
  if (!response.ok) {
    const message = await readUpstreamError(response);
    console.error('OpenAI models request failed:', response.status, message);
    throw new OpenAIRequestError(502, message);
  }
  let body: ModelsResponse;
  try {
    body = (await response.json()) as ModelsResponse;
  } catch (error) {
    console.error('OpenAI models response was invalid JSON:', error);
    throw new OpenAIRequestError(502, 'OpenAI API returned invalid JSON');
  }
  const modelId = body.data?.[0]?.id;
  if (modelId) return modelId;
  const defaultModel = (await getOpenAIDefaultModel()).trim();
  if (defaultModel) return defaultModel;
  console.error('OpenAI models list is empty');
  throw new OpenAIRequestError(502, 'No models available at the OpenAI API URL');
}

function eventBoundary(buffer: string): { index: number; length: number } | null {
  const lf = buffer.indexOf('\n\n');
  const crlf = buffer.indexOf('\r\n\r\n');
  if (lf < 0 && crlf < 0) return null;
  if (crlf < 0 || (lf >= 0 && lf < crlf)) return { index: lf, length: 2 };
  return { index: crlf, length: 4 };
}

function eventPayload(block: string): string | null {
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line === '' || line.startsWith(':')) continue;
    if (!line.startsWith('data:')) continue;
    const value = line.slice(5);
    dataLines.push(value.startsWith(' ') ? value.slice(1) : value);
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join('\n').trim();
  return payload || null;
}

function streamDelta(body: Record<string, unknown>): Record<string, unknown> | null {
  if (!Array.isArray(body.choices)) return null;
  const choice = body.choices[0];
  if (!isRecord(choice) || !isRecord(choice.delta)) return null;
  return choice.delta;
}

function deltasFromPayload(payload: string): ChatStreamEvent[] {
  let body: unknown;
  try {
    body = JSON.parse(payload);
  } catch (error) {
    console.error('OpenAI chat completion returned invalid JSON:', error);
    throw new OpenAIRequestError(502, 'OpenAI API returned invalid JSON');
  }
  if (!isRecord(body)) {
    console.error('OpenAI chat completion returned invalid JSON:', body);
    throw new OpenAIRequestError(502, 'OpenAI API returned invalid JSON');
  }
  const message = errorText(body);
  if (message) {
    console.error('OpenAI chat completion failed:', message);
    throw new OpenAIRequestError(502, message);
  }
  const delta = streamDelta(body);
  if (!delta) return [];

  const events: ChatStreamEvent[] = [];
  const reasoning = delta.reasoning_content ?? delta.reasoning;
  if (typeof reasoning === 'string' && reasoning) events.push({ type: 'reasoning', delta: reasoning });
  if (typeof delta.content === 'string' && delta.content) events.push({ type: 'content', delta: delta.content });
  return events;
}

function takeEvents(buffer: string): { events: ChatStreamEvent[]; rest: string; done: boolean } {
  const events: ChatStreamEvent[] = [];
  let rest = buffer;
  while (true) {
    const boundary = eventBoundary(rest);
    if (!boundary) return { events, rest, done: false };
    const block = rest.slice(0, boundary.index);
    rest = rest.slice(boundary.index + boundary.length);
    const payload = eventPayload(block);
    if (payload === null) continue;
    if (payload === '[DONE]') return { events, rest: '', done: true };
    events.push(...deltasFromPayload(payload));
  }
}

function flushTail(buffer: string): { events: ChatStreamEvent[]; done: boolean } {
  if (buffer.trim() === '') return { events: [], done: false };
  const payload = eventPayload(buffer);
  if (payload === null) return { events: [], done: false };
  if (payload === '[DONE]') return { events: [], done: true };
  return { events: deltasFromPayload(payload), done: false };
}

async function readChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal,
  idle: IdleAbort,
): Promise<Uint8Array | null> {
  let result: ReadableStreamReadResult<Uint8Array>;
  try {
    result = await reader.read();
  } catch (error) {
    if (signal.aborted) throw new OpenAIRequestAborted();
    if (idle.signal.aborted) throw timeoutError(error);
    console.error('OpenAI chat completion failed:', error);
    throw new OpenAIRequestError(502, errorMessage(error));
  }
  if (signal.aborted) throw new OpenAIRequestAborted();
  if (idle.signal.aborted) throw timeoutError();
  if (result.done) return null;
  return result.value;
}

async function cancelReader(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal,
  idle: IdleAbort,
): Promise<void> {
  try {
    await reader.cancel();
  } catch (error) {
    if (signal.aborted || idle.signal.aborted) return;
    console.error('OpenAI chat stream cancel failed:', error);
  }
}

async function* readChatStream(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
  idle: IdleAbort,
): AsyncGenerator<ChatStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const chunk = await readChunk(reader, signal, idle);
      if (chunk === null) break;
      idle.reset();
      buffer += decoder.decode(chunk, { stream: true });
      const taken = takeEvents(buffer);
      buffer = taken.rest;
      for (const event of taken.events) yield event;
      if (taken.done) return;
    }
    buffer += decoder.decode();
    const taken = takeEvents(buffer);
    for (const event of taken.events) yield event;
    if (taken.done) return;
    const tail = flushTail(taken.rest);
    for (const event of tail.events) yield event;
  } finally {
    idle.dispose();
    await cancelReader(reader, signal, idle);
  }
}

export async function streamChatCompletion(
  messages: ChatMessage[],
  signal: AbortSignal,
): Promise<AsyncIterable<ChatStreamEvent>> {
  const apiUrl = await requireBaseUrl();
  const model = await resolveModel(signal);
  // The idle timer covers headers and each gap between chunks. It resets when
  // headers arrive and on every chunk, so a long reasoning stream is not cut
  // off at 120s of total time.
  const idle = createIdleAbort(COMPLETION_TIMEOUT_MS);
  try {
    const response = await fetch(`${apiUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: true }),
      signal: AbortSignal.any([signal, idle.signal]),
    });
    if (!response.ok) {
      idle.dispose();
      const message = await readUpstreamError(response);
      console.error('OpenAI chat completion failed:', response.status, message);
      throw new OpenAIRequestError(502, message);
    }
    if (!response.body) {
      idle.dispose();
      console.error('OpenAI chat completion returned no body');
      throw new OpenAIRequestError(502, 'OpenAI API returned no message content');
    }
    idle.reset();
    return readChatStream(response.body, signal, idle);
  } catch (error) {
    const timedOut = idle.signal.aborted && !signal.aborted;
    idle.dispose();
    if (error instanceof OpenAIRequestError || error instanceof OpenAIRequestAborted) throw error;
    if (timedOut) throw timeoutError(error);
    rethrowUpstream(signal, error, 'OpenAI chat completion failed:');
  }
}
