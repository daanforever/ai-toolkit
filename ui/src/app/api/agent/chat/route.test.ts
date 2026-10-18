import { createServer, IncomingMessage, Server, ServerResponse } from 'http';
import { AddressInfo } from 'net';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import path from 'path';
import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

const settings = vi.hoisted(() => ({
  apiUrl: '',
  datasetsRoot: '',
}));

vi.mock('@/server/settings', () => ({
  getOpenAIApiUrl: async () => settings.apiUrl,
  getOpenAIDefaultModel: async () => '',
  getDatasetsRoot: async () => settings.datasetsRoot,
}));

import { POST } from '@/app/api/agent/chat/route';

const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

interface MockHandle {
  url: string;
  bodies: unknown[];
  close: () => Promise<void>;
}

function sseChunk(delta: unknown, finish: string | null = null): string {
  return `data: ${JSON.stringify({ choices: [{ delta, finish_reason: finish }] })}\n\n`;
}

async function listen(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<MockHandle> {
  const bodies: unknown[] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', chunk => chunks.push(Buffer.from(chunk)));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      if (raw) {
        try {
          bodies.push(JSON.parse(raw));
        } catch (error) {
          bodies.push(raw);
          console.error('Mock upstream body was not JSON:', error);
        }
      }
      handler(req, res);
    });
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${address.port}/v1`,
    bodies,
    close: () => new Promise(resolve => server.close(() => resolve())),
  };
}

function models(res: ServerResponse): void {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ data: [{ id: 'test-model' }] }));
}

async function ndjson(response: Response): Promise<unknown[]> {
  const text = await response.text();
  return text
    .split('\n')
    .filter(line => line.trim() !== '')
    .map(line => JSON.parse(line) as unknown);
}

function chatRequest(body: unknown, signal?: AbortSignal): NextRequest {
  return new NextRequest('http://localhost/api/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
}

describe('POST /api/agent/chat tools', () => {
  const servers: MockHandle[] = [];
  const root = mkdtempSync(path.join(tmpdir(), 'lightbox-edit-'));
  const imagePath = path.join(root, 'sample.png');
  writeFileSync(imagePath, PNG);
  settings.datasetsRoot = root;

  afterEach(async () => {
    while (servers.length > 0) {
      const server = servers.pop();
      if (server) await server.close();
    }
  });

  async function start(reply: (res: ServerResponse) => void): Promise<MockHandle> {
    const server = await listen((req, res) => {
      const url = req.url ?? '';
      if (url.endsWith('/models')) {
        models(res);
        return;
      }
      reply(res);
    });
    settings.apiUrl = server.url;
    servers.push(server);
    return server;
  }

  function chatBodies(server: MockHandle): Record<string, unknown>[] {
    return server.bodies.filter(body => typeof body === 'object' && body !== null && 'messages' in body) as Record<
      string,
      unknown
    >[];
  }

  it('omits tools and keeps a content stream unchanged when no image is open', async () => {
    const server = await start(res => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.end(`${sseChunk({ content: 'ok' })}data: [DONE]\n\n`);
    });
    const response = await POST(chatRequest({ messages: [{ role: 'user', content: 'hi' }] }));
    expect(response.status).toBe(200);
    expect(await ndjson(response)).toEqual([{ type: 'content', delta: 'ok' }]);
    const body = chatBodies(server)[0];
    expect(body.tools).toBeUndefined();
    expect(body.stream).toBe(true);
  });

  it('sends the edit tool only when an image is open', async () => {
    const server = await start(res => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.end(
        `${sseChunk({
          tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'edit', arguments: '' } }],
        })}${sseChunk({
          tool_calls: [{ index: 0, type: 'function', function: { arguments: '{"caption":"brown dog"}' } }],
        })}${sseChunk({}, 'tool_calls')}data: [DONE]\n\n`,
      );
    });
    const response = await POST(
      chatRequest({
        messages: [{ role: 'user', content: 'rewrite the caption' }],
        imgPath: imagePath,
        caption: 'dog',
        reasoning: false,
      }),
    );
    expect(await ndjson(response)).toEqual([{ type: 'edit', caption: 'brown dog' }]);
    const body = chatBodies(server)[0];
    expect(body.tools).toEqual([
      expect.objectContaining({
        type: 'function',
        function: expect.objectContaining({ name: 'edit' }),
      }),
    ]);
    expect(body.reasoning_effort).toBe('none');
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
  });

  it('relays text and then the edit', async () => {
    await start(res => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.end(
        `${sseChunk({ content: 'Updated.' })}${sseChunk({
          tool_calls: [
            { index: 0, id: 'c1', type: 'function', function: { name: 'edit', arguments: '{"caption":"brown dog"}' } },
          ],
        })}data: [DONE]\n\n`,
      );
    });
    const response = await POST(
      chatRequest({ messages: [{ role: 'user', content: 'rewrite' }], imgPath: imagePath }),
    );
    expect(await ndjson(response)).toEqual([
      { type: 'content', delta: 'Updated.' },
      { type: 'edit', caption: 'brown dog' },
    ]);
  });

  it('emits an error event for malformed tool arguments', async () => {
    await start(res => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.end(
        `${sseChunk({
          tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'edit', arguments: '{' } }],
        })}data: [DONE]\n\n`,
      );
    });
    const response = await POST(
      chatRequest({ messages: [{ role: 'user', content: 'rewrite' }], imgPath: imagePath }),
    );
    expect(response.status).toBe(200);
    expect(await ndjson(response)).toEqual([{ type: 'error', message: 'Invalid tool arguments' }]);
  });

  it('returns the upstream error when the server rejects tools', async () => {
    await start(res => {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'tools not supported' } }));
    });
    const response = await POST(
      chatRequest({ messages: [{ role: 'user', content: 'rewrite' }], imgPath: imagePath }),
    );
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: 'tools not supported' });
  });

  it('does not emit an edit when the client aborts mid tool call', async () => {
    let release: () => void = () => {};
    const opened = new Promise<void>(resolve => {
      release = resolve;
    });
    await start(res => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write(
        sseChunk({
          tool_calls: [{ index: 0, id: 'c1', type: 'function', function: { name: 'edit', arguments: '{"cap' } }],
        }),
      );
      release();
    });
    const controller = new AbortController();
    const pending = POST(
      chatRequest({ messages: [{ role: 'user', content: 'rewrite' }], imgPath: imagePath }, controller.signal),
    );
    await opened;
    controller.abort();
    const response = await pending;
    const text = await response.text();
    expect(text).not.toContain('"type":"edit"');
    expect(response.status === 200 || response.status === 499).toBe(true);
  });
});
