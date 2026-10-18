import { describe, expect, it } from 'vitest';
import { collectEditCalls } from '@/server/agentTools';
import { ChatStreamEvent, OpenAIRequestError } from '@/server/openai';

async function collect(events: ChatStreamEvent[]): Promise<Awaited<ReturnType<typeof readAll>>> {
  return readAll(collectEditCalls(fromList(events)));
}

async function* fromList(events: ChatStreamEvent[]): AsyncGenerator<ChatStreamEvent> {
  for (const event of events) yield event;
}

async function readAll(events: AsyncIterable<{ type: string; delta?: string; caption?: string }>) {
  const collected = [];
  for await (const event of events) collected.push(event);
  return collected;
}

function tool(index: number, args: string, name?: string): ChatStreamEvent {
  return { type: 'tool_call', index, ...(name ? { name } : {}), arguments: args };
}

describe('collectEditCalls', () => {
  it('joins argument fragments and emits one edit', async () => {
    const events = await collect([
      tool(0, '', 'edit'),
      tool(0, '{"cap'),
      tool(0, 'tion":"brown dog"}'),
    ]);
    expect(events).toEqual([{ type: 'edit', caption: 'brown dog' }]);
  });

  it('passes reasoning and content through and appends the edit', async () => {
    const events = await collect([
      { type: 'reasoning', delta: 'think' },
      tool(0, '{"caption":"', 'edit'),
      { type: 'content', delta: 'done' },
      tool(0, 'a brown dog"}'),
    ]);
    expect(events).toEqual([
      { type: 'reasoning', delta: 'think' },
      { type: 'content', delta: 'done' },
      { type: 'edit', caption: 'a brown dog' },
    ]);
  });

  it('keeps a split unicode caption intact', async () => {
    const events = await collect([tool(0, '{"caption":"со', 'edit'), tool(0, 'бака"}')]);
    expect(events).toEqual([{ type: 'edit', caption: 'собака' }]);
  });

  it('lets the last edit win', async () => {
    const events = await collect([
      tool(0, '{"caption":"first"}', 'edit'),
      tool(1, '{"caption":"second"}', 'edit'),
    ]);
    expect(events).toEqual([{ type: 'edit', caption: 'second' }]);
  });

  it('emits no edit when the reply has no tool call', async () => {
    const events = await collect([{ type: 'content', delta: 'ok' }]);
    expect(events).toEqual([{ type: 'content', delta: 'ok' }]);
  });

  it('rejects an unknown tool', async () => {
    await expect(collect([tool(0, '{"caption":"x"}', 'noop')])).rejects.toMatchObject({
      status: 502,
      message: 'Unknown tool',
    });
  });

  it('rejects a missing tool name', async () => {
    await expect(collect([tool(0, '{"caption":"x"}')])).rejects.toBeInstanceOf(OpenAIRequestError);
  });

  it('rejects invalid JSON', async () => {
    await expect(collect([tool(0, '{"caption":', 'edit')])).rejects.toMatchObject({
      message: 'Invalid tool arguments',
    });
  });

  it('rejects a missing caption', async () => {
    await expect(collect([tool(0, '{}', 'edit')])).rejects.toMatchObject({ message: 'Invalid tool arguments' });
  });

  it('rejects a non-string caption', async () => {
    await expect(collect([tool(0, '{"caption":1}', 'edit')])).rejects.toMatchObject({
      message: 'Invalid tool arguments',
    });
  });

  it('rejects a blank caption', async () => {
    await expect(collect([tool(0, '{"caption":"  "}', 'edit')])).rejects.toMatchObject({
      message: 'Invalid tool arguments',
    });
  });

  it('does not skip an earlier invalid call when a later edit is valid', async () => {
    await expect(
      collect([tool(0, '{', 'edit'), tool(1, '{"caption":"ok"}', 'edit')]),
    ).rejects.toMatchObject({ message: 'Invalid tool arguments' });
  });

  it('emits no edit when the stream throws before it ends', async () => {
    async function* source(): AsyncGenerator<ChatStreamEvent> {
      yield tool(0, '{"caption":"partial', 'edit');
      throw new Error('aborted');
    }
    const seen: string[] = [];
    await expect(
      (async () => {
        for await (const event of collectEditCalls(source())) seen.push(event.type);
      })(),
    ).rejects.toThrow('aborted');
    expect(seen).not.toContain('edit');
  });

  it('emits no edit when the consumer stops early', async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>(resolve => {
      release = resolve;
    });
    async function* source(): AsyncGenerator<ChatStreamEvent> {
      yield tool(0, '{"caption":"brown dog"}', 'edit');
      await gate;
    }
    const generator = collectEditCalls(source());
    const stopping = generator.return(undefined);
    release();
    const result = await stopping;
    expect(result.done).toBe(true);
    expect(result.value).toBeUndefined();
  });
});
