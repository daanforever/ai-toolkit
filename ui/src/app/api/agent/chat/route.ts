import { NextRequest, NextResponse } from 'next/server';
import { EDIT_TOOL, CollectedChatEvent, collectEditCalls } from '@/server/agentTools';
import { AgentChatRequestError, buildChatMessages, parseAgentChatRequest } from '@/server/agentChat';
import { OpenAIRequestAborted, OpenAIRequestError, streamChatCompletion } from '@/server/openai';
import { AgentChatStreamEvent } from '@/types/agentChat';

const encoder = new TextEncoder();

function encodeNdjsonLine(event: AgentChatStreamEvent): Uint8Array {
  return encoder.encode(`${JSON.stringify(event)}\n`);
}

// Failures before upstream chat headers are JSON responses, including 499 when
// the client disconnects. After those headers, failures are NDJSON error events
// and the stream closes. A disconnect after headers closes the stream with no
// error event and aborts the upstream read.
function streamNdjson(events: AsyncIterable<CollectedChatEvent>, signal: AbortSignal, cancelUpstream: () => void): Response {
  let closed = false;
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const iterator = events[Symbol.asyncIterator]();
      try {
        while (!closed) {
          const next = await iterator.next();
          if (closed) {
            await iterator.return?.(undefined);
            return;
          }
          if (next.done) {
            closed = true;
            controller.close();
            return;
          }
          controller.enqueue(encodeNdjsonLine(next.value));
        }
      } catch (error) {
        if (closed) return;
        if (signal.aborted || error instanceof OpenAIRequestAborted) {
          closed = true;
          controller.close();
          return;
        }
        const message = error instanceof OpenAIRequestError ? error.message : 'Agent chat failed';
        if (!(error instanceof OpenAIRequestError)) {
          console.error('Agent chat failed:', error);
        }
        if (closed) return;
        controller.enqueue(encodeNdjsonLine({ type: 'error', message }));
        closed = true;
        controller.close();
      }
    },
    cancel() {
      closed = true;
      cancelUpstream();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Accel-Buffering': 'no',
    },
  });
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch (error) {
    if (request.signal.aborted) {
      return new NextResponse(null, { status: 499 });
    }
    console.error('Agent chat received invalid JSON:', error);
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const cancelUpstream = new AbortController();
  const signal = AbortSignal.any([request.signal, cancelUpstream.signal]);

  try {
    const parsed = parseAgentChatRequest(body);
    if (!parsed) {
      return NextResponse.json({ error: 'Invalid messages' }, { status: 400 });
    }
    const events = await streamChatCompletion(
      await buildChatMessages(parsed),
      signal,
      parsed.reasoning ?? true,
      parsed.tools === true && parsed.imgPath ? [EDIT_TOOL] : undefined,
    );
    return streamNdjson(collectEditCalls(events), signal, () => cancelUpstream.abort());
  } catch (error) {
    if (error instanceof OpenAIRequestAborted) {
      return new NextResponse(null, { status: 499 });
    }
    if (error instanceof AgentChatRequestError || error instanceof OpenAIRequestError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    console.error('Agent chat failed:', error);
    return NextResponse.json({ error: 'Agent chat failed' }, { status: 500 });
  }
}
