import { NextRequest, NextResponse } from 'next/server';
import { createChatCompletion, OpenAIRequestError } from '@/server/openai';
import { AgentChatRequest } from '@/types/agentChat';

function isAgentChatRequest(value: unknown): value is AgentChatRequest {
  if (typeof value !== 'object' || value === null || !('message' in value)) {
    return false;
  }
  const message = (value as { message: unknown }).message;
  return typeof message === 'string' && message.trim() !== '';
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch (error) {
    console.error('Agent chat received invalid JSON:', error);
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  if (!isAgentChatRequest(body)) {
    return NextResponse.json({ error: 'Message is required' }, { status: 400 });
  }

  try {
    const reply = await createChatCompletion([{ role: 'user', content: body.message.trim() }]);
    return NextResponse.json({ reply });
  } catch (error) {
    if (error instanceof OpenAIRequestError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    console.error('Agent chat failed:', error);
    return NextResponse.json({ error: 'Agent chat failed' }, { status: 500 });
  }
}
