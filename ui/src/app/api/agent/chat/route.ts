import { NextRequest, NextResponse } from 'next/server';
import { AgentChatRequestError, buildChatMessages, parseAgentChatRequest } from '@/server/agentChat';
import { createChatCompletion, OpenAIRequestError } from '@/server/openai';

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch (error) {
    console.error('Agent chat received invalid JSON:', error);
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  try {
    const parsed = parseAgentChatRequest(body);
    if (!parsed) {
      return NextResponse.json({ error: 'Invalid messages' }, { status: 400 });
    }
    const reply = await createChatCompletion(await buildChatMessages(parsed));
    return NextResponse.json({ reply });
  } catch (error) {
    if (error instanceof AgentChatRequestError || error instanceof OpenAIRequestError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    console.error('Agent chat failed:', error);
    return NextResponse.json({ error: 'Agent chat failed' }, { status: 500 });
  }
}
