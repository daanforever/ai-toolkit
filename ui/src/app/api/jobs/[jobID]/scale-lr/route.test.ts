import { EventEmitter } from 'events';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveVenvPython, TOOLKIT_ROOT } from '@/paths';

const state = vi.hoisted(() => ({
  job: null as {
    id: string;
    name: string;
    job_config: string;
    runtimeParams: {
      runtime_scale_lr_by_index?: boolean | null;
      runtime_scale_lr_mean?: number | null;
      runtime_scale_lr_std?: number | null;
    } | null;
  } | null,
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

import { GET } from '@/app/api/jobs/[jobID]/scale-lr/route';

const roots: string[] = [];

function makeTraining(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'scale-lr-route-'));
  roots.push(root);
  return root;
}

function jobConfig(params: Record<string, unknown>): string {
  return JSON.stringify({
    config: { process: [{ train: { optimizer_params: params } }] },
  });
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
  return GET(new NextRequest(`http://localhost/api/jobs/${jobID}/scale-lr`), {
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

function enabledJob(
  name: string,
  params: Record<string, unknown>,
  runtimeParams: NonNullable<typeof state.job>['runtimeParams'] = null,
) {
  return {
    id: 'job-1',
    name,
    job_config: jobConfig(params),
    runtimeParams,
  };
}

describe('GET /api/jobs/:id/scale-lr', () => {
  it('returns 404 when the job does not exist', async () => {
    const response = await request('missing');
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: 'Job not found' });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('returns available false when scale_lr_by_index is off', async () => {
    state.job = enabledJob('idle', {
      scale_lr_by_index: false,
      scale_lr_mean: 15,
      scale_lr_std: 0.25,
    });

    const response = await request('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      available: false,
      reason: 'scale_lr_by_index is off',
    });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('lets a runtime false switch override config', async () => {
    state.job = enabledJob(
      'idle',
      { scale_lr_by_index: true, scale_lr_mean: 15, scale_lr_std: 0.25 },
      { runtime_scale_lr_by_index: false, runtime_scale_lr_mean: 15, runtime_scale_lr_std: 0.25 },
    );

    const response = await request('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      available: false,
      reason: 'scale_lr_by_index is off',
    });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('rejects a non-positive std without spawning', async () => {
    state.job = enabledJob('idle', {
      scale_lr_by_index: true,
      scale_lr_mean: 15,
      scale_lr_std: 0,
    });

    const response = await request('job-1');
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.available).toBe(false);
    expect(body.reason).toContain('scale_lr_std');
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('returns available false when the job folder has no safetensors', async () => {
    const training = makeTraining();
    fs.mkdirSync(path.join(training, 'empty-job'));
    state.training = training;
    state.job = enabledJob('empty-job', {
      scale_lr_by_index: true,
      scale_lr_mean: 15,
      scale_lr_std: 0.25,
    });

    const response = await request('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ available: false });
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('scores the newest safetensors file with config mean and std', async () => {
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
    state.job = enabledJob('flower', {
      scale_lr_by_index: true,
      scale_lr_mean: 15,
      scale_lr_std: 0.25,
      scale_lr_mask: ['layers'],
    });

    const payload = {
      indices: [0, 1],
      weights: [0.5, 1],
      n: 2,
      file: { filename: 'z_new.safetensors', step: 10 },
    };
    spawnMock.mockImplementation(() => fakeChild(JSON.stringify(payload), '', 0));

    const response = await request('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      ...payload,
      available: true,
      mean: 15,
      std: 0.25,
      mask: ['layers'],
    });
    expect(spawnMock).toHaveBeenCalledTimes(1);
    const [python, args] = spawnMock.mock.calls[0];
    expect(python).toBe(resolveVenvPython(TOOLKIT_ROOT));
    expect(args[0]).toBe(path.join(TOOLKIT_ROOT, 'toolkit', 'scale_lr_curve.py'));
    expect(args[1]).toBe(newer);
    expect(args).toEqual(expect.arrayContaining(['--mean', '15', '--std', '0.25', '--mask', 'layers']));
  });

  it('uses runtime mean and std over the job config', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'flower');
    fs.mkdirSync(jobFolder);
    const file = path.join(jobFolder, 'lora.safetensors');
    fs.writeFileSync(file, 'weights');
    state.training = training;
    state.job = enabledJob(
      'flower',
      {
        scale_lr_by_index: true,
        scale_lr_mean: 1,
        scale_lr_std: 0.2,
        scale_lr_mask: ['attn'],
      },
      { runtime_scale_lr_mean: 15, runtime_scale_lr_std: 0.25 },
    );

    const payload = { indices: [0, 1], weights: [1, 0.2], n: 2, file: { filename: 'lora.safetensors', step: 4 } };
    spawnMock.mockImplementation(() => fakeChild(JSON.stringify(payload), '', 0));

    const response = await request('job-1');
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.mean).toBe(15);
    expect(body.std).toBe(0.25);
    expect(body.mask).toEqual(['attn']);
    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args).toEqual(expect.arrayContaining(['--mean', '15', '--std', '0.25', '--mask', 'attn']));
    expect(args).not.toContain('0.2');
  });

  it('returns the script error', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'broken');
    fs.mkdirSync(jobFolder);
    fs.writeFileSync(path.join(jobFolder, 'broken.safetensors'), 'not a header');
    state.training = training;
    state.job = enabledJob('broken', {
      scale_lr_by_index: true,
      scale_lr_mean: 15,
      scale_lr_std: 0.25,
    });
    spawnMock.mockImplementation(() => fakeChild('', 'No layer indices found in LoRA keys.\n', 1));

    const response = await request('job-1');
    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body.error).toContain('No layer indices');
  });

  it('returns 500 when the script prints invalid JSON', async () => {
    const training = makeTraining();
    const jobFolder = path.join(training, 'badjson');
    fs.mkdirSync(jobFolder);
    fs.writeFileSync(path.join(jobFolder, 'weights.safetensors'), 'x');
    state.training = training;
    state.job = enabledJob('badjson', {
      scale_lr_by_index: true,
      scale_lr_mean: 15,
      scale_lr_std: 0.25,
    });
    spawnMock.mockImplementation(() => fakeChild('not-json', '', 0));

    const response = await request('job-1');
    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body.error).toContain('invalid JSON');
  });
});
