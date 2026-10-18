import { describe, expect, it } from 'vitest';
import { OpenAIRequestError, takeChatStreamEvents } from '@/server/openai';

function sse(payloads: string[]): string {
  return payloads.map(payload => `data: ${payload}\n\n`).join('');
}

function payload(delta: unknown, finish: string | null = null): string {
  return JSON.stringify({ choices: [{ delta, finish_reason: finish }] });
}

describe('takeChatStreamEvents', () => {
  it('reads a tool call split across chunks, then [DONE]', () => {
    const buffer = sse([
      payload({
        tool_calls: [
          { index: 0, id: 'call-1', type: 'function', function: { name: 'edit', arguments: '' } },
        ],
      }),
      payload({ tool_calls: [{ index: 0, type: 'function', function: { arguments: '{"caption":"brown dog"}' } }] }),
      payload({}, 'tool_calls'),
      '[DONE]',
    ]);
    const taken = takeChatStreamEvents(buffer);
    expect(taken.done).toBe(true);
    expect(taken.rest).toBe('');
    expect(taken.events).toEqual([
      { type: 'tool_call', index: 0, id: 'call-1', name: 'edit', arguments: '' },
      { type: 'tool_call', index: 0, arguments: '{"caption":"brown dog"}' },
    ]);
  });

  it('keeps reasoning and content around tool fragments', () => {
    const buffer = sse([
      payload({ reasoning_content: 'look' }),
      payload({
        content: 'ok',
        tool_calls: [{ index: 0, type: 'function', function: { arguments: '{' } }],
      }),
    ]);
    const taken = takeChatStreamEvents(buffer);
    expect(taken.events).toEqual([
      { type: 'reasoning', delta: 'look' },
      { type: 'content', delta: 'ok' },
      { type: 'tool_call', index: 0, arguments: '{' },
    ]);
  });

  it('leaves a content-only stream unchanged', () => {
    const buffer = sse([payload({ content: 'ok' }), '[DONE]']);
    const taken = takeChatStreamEvents(buffer);
    expect(taken.done).toBe(true);
    expect(taken.events).toEqual([{ type: 'content', delta: 'ok' }]);
  });

  it('throws the upstream error text', () => {
    const buffer = sse([JSON.stringify({ error: { message: 'tools not supported' } })]);
    expect(() => takeChatStreamEvents(buffer)).toThrow(OpenAIRequestError);
    try {
      takeChatStreamEvents(buffer);
    } catch (error) {
      expect(error).toMatchObject({ status: 502, message: 'tools not supported' });
    }
  });

  it('rejects a tool call without an index', () => {
    const buffer = sse([payload({ tool_calls: [{ type: 'function', function: { name: 'edit' } }] })]);
    expect(() => takeChatStreamEvents(buffer)).toThrow(/Invalid tool call/);
  });
});
