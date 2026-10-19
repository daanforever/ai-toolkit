import { EventEmitter } from 'events';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveVenvPython, TOOLKIT_ROOT } from '@/paths';

const state = vi.hoisted(() => ({
  job: null as { id: string; name: string } | null,
  training: '',
}));

const spawnMock = vi.hoisted(() => vi.fn());

vi.mock('@prisma/client', () => ({
  PrismaClient: class {
    job = {
      findUnique: async () => state.job,
    };
  },
}));

vi.mock('@/server/settings', () => ({
  getTrainingFolder: async () => state.training,
}));

vi.mock('child_process', () => ({
  spawn: (...args: unknown[]) => spawnMock(...args),
}));

import { GET } from '@/app/api/jobs/[jobID]/heatmap/route';

const roots: string[] = [];

function makeTraining(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'heatmap-route-'));
  roots.push(root);
  return root;
}

afterEach(() => {
  state.job = null;
  state.training = '';
  spawnMock.mockReset();
  while (roots.length > 0) {
    const root = roots.pop();
    if (root) fs.rmSync(root, { recursive: true, force: true });
  }
});

function request(jobID: string): Promise<Response> {
  return GET(new NextRequest(`http://localhost/api/jobs/${jobID}/heatmap`), {
    params: Promise.resolve({ jobID }) as unknown as { jobID: string },
  });
}

function fakeChild(stdout: string, stderr: string, code: number) {
  const child = new EventEmitter() as EventEmitter & {
    stdout: EventEmitter;
    stderr: EventEmitter;
    kill: ReturnType<typeof vi.fn>;
  };
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.kill = vi.fn();
  process.nextTick(() => {
    if (stdout) child.stdout.emit('data', Buffer.from(stdout));
    if (stderr) child.stderr.emit('data', stderr);
    child.emit('close', code);
  });
  return child;
}

describe('GET /api/jobs/:id/heatmap', () => {
  it('returns 404 when the job does not exist', async () => {
    const response = await request('missing');
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: 'Job not found' });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('returns available false when the job folder has no safetensors', async () => {
    const training = makeTraining();
    fs.mkdirSync(path.join(training, 'empty-job'));
    fs.writeFileSync(path.join(training, 'empty-job', 'log.txt'), 'no weights');
    state.training = training;
    state.job = { id: 'job-1', name: 'empty-job' };

    const response = await request('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ available: false });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('scores the newest safetensors file', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'flower');
    fs.mkdirSync(jobFolder);
    const older = path.join(jobFolder, 'a_old.safetensors');
    const newer = path.join(jobFolder, 'z_new.safetensors');
    fs.writeFileSync(older, 'old');
    fs.writeFileSync(newer, 'new');
    fs.utimesSync(older, new Date('2020-01-01T00:00:00Z'), new Date('2020-01-01T00:00:00Z'));
    fs.utimesSync(newer, new Date('2024-06-01T00:00:00Z'), new Date('2024-06-01T00:00:00Z'));
    state.training = training;
    state.job = { id: 'job-2', name: 'flower' };

    const payload = {
      rows: ['layers_0'],
      cols: ['attention_to_q'],
      metrics: { frobenius: [[1.5]] },
      file: { filename: 'z_new.safetensors', path: newer, mtime_ms: 1, step: 10, epoch: 1 },
    };
    spawnMock.mockImplementation(() => fakeChild(JSON.stringify(payload), '', 0));

    const response = await request('job-2');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ...payload, available: true });
    expect(spawnMock).toHaveBeenCalledTimes(1);
    const [python, args] = spawnMock.mock.calls[0];
    expect(python).toBe(resolveVenvPython(TOOLKIT_ROOT));
    expect(args[0]).toBe(path.join(TOOLKIT_ROOT, 'toolkit', 'lora_heatmap.py'));
    expect(args[1]).toBe(newer);
  });

  it('returns the script error for a corrupt file', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'broken');
    fs.mkdirSync(jobFolder);
    fs.writeFileSync(path.join(jobFolder, 'broken.safetensors'), 'not a header');
    state.training = training;
    state.job = { id: 'job-3', name: 'broken' };
    spawnMock.mockImplementation(() => fakeChild('', 'header too small\n', 1));

    const response = await request('job-3');
    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body.error).toContain('header too small');
  });

  it('returns 500 when the script prints invalid JSON', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'badjson');
    fs.mkdirSync(jobFolder);
    fs.writeFileSync(path.join(jobFolder, 'weights.safetensors'), 'x');
    state.training = training;
    state.job = { id: 'job-4', name: 'badjson' };
    spawnMock.mockImplementation(() => fakeChild('not-json', '', 0));

    const response = await request('job-4');
    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body.error).toContain('invalid JSON');
  });
});
