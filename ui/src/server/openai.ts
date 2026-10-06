import { getOpenAIApiUrl } from '@/server/settings';

export type ContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | ContentPart[];
}

export class OpenAIRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'OpenAIRequestError';
    this.status = status;
  }
}

const MODELS_TIMEOUT_MS = 10_000;
const COMPLETION_TIMEOUT_MS = 120_000;

interface ModelsResponse {
  data?: { id?: string }[];
}

interface ChatCompletionResponse {
  choices?: { message?: { content?: unknown } }[];
  error?: { message?: unknown };
}

async function requireBaseUrl(): Promise<string> {
  const apiUrl = (await getOpenAIApiUrl()).trim().replace(/\/+$/, '');
  if (!apiUrl) {
    throw new OpenAIRequestError(400, 'OpenAI API URL is not set (Settings)');
  }
  return apiUrl;
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

async function readUpstreamError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ChatCompletionResponse;
    if (typeof body.error?.message === 'string' && body.error.message) {
      return body.error.message;
    }
  } catch (error) {
    console.error('OpenAI API returned a non-JSON error body:', error);
  }
  return response.statusText || `Upstream request failed (${response.status})`;
}

export async function resolveModel(): Promise<string> {
  const apiUrl = await requireBaseUrl();
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/models`, { signal: AbortSignal.timeout(MODELS_TIMEOUT_MS) });
  } catch (error) {
    console.error('OpenAI models request failed:', error);
    throw new OpenAIRequestError(502, errorMessage(error));
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
  if (!modelId) {
    console.error('OpenAI models list is empty');
    throw new OpenAIRequestError(502, 'No models available at the OpenAI API URL');
  }
  return modelId;
}

export async function createChatCompletion(messages: ChatMessage[]): Promise<string> {
  const apiUrl = await requireBaseUrl();
  const model = await resolveModel();
  let response: Response;
  try {
    response = await fetch(`${apiUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: false }),
      signal: AbortSignal.timeout(COMPLETION_TIMEOUT_MS),
    });
  } catch (error) {
    console.error('OpenAI chat completion failed:', error);
    throw new OpenAIRequestError(502, errorMessage(error));
  }
  if (!response.ok) {
    const message = await readUpstreamError(response);
    console.error('OpenAI chat completion failed:', response.status, message);
    throw new OpenAIRequestError(502, message);
  }
  let body: ChatCompletionResponse;
  try {
    body = (await response.json()) as ChatCompletionResponse;
  } catch (error) {
    console.error('OpenAI chat completion returned invalid JSON:', error);
    throw new OpenAIRequestError(502, 'OpenAI API returned invalid JSON');
  }
  const content = body.choices?.[0]?.message?.content;
  if (typeof content !== 'string') {
    console.error('OpenAI chat completion missing content:', body);
    throw new OpenAIRequestError(502, 'OpenAI API returned no message content');
  }
  return content;
}
