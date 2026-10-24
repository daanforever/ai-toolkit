import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

const state = vi.hoisted(() => ({
  job: null as {
    id: string;
    job_config: string;
    status: string;
    stop: boolean;
  } | null,
  commands: [] as { jobId: string; action: string }[],
}));

vi.mock('@prisma/client', () => ({
  PrismaClient: class {
    job = {
      findUnique: async () => state.job,
    };
    runtimeCommand = {
      create: async ({ data }: { data: { jobId: string; action: string } }) => {
        state.commands.push(data);
        return { id: state.commands.length, ...data, createdAt: '2026-01-01T00:00:00.000Z' };
      },
    };
    runtimeParams = {
      upsert: async () => {
        throw new Error('RuntimeParams must not be written');
      },
    };
  },
}));

import { GET, POST } from '@/app/api/jobs/[jobID]/ui-actions/route';

function jobConfig(type: string): string {
  return JSON.stringify({
    job: 'extension',
    config: { process: [{ type }] },
    meta: { name: 't', version: '1' },
  });
}

function runningJob(type = 'z_image_diffsynth_trainer', status = 'running', stop = false) {
  state.job = {
    id: 'job-1',
    job_config: jobConfig(type),
    status,
    stop,
  };
}

afterEach(() => {
  state.job = null;
  state.commands = [];
});

function get(jobID: string): Promise<Response> {
  return GET(new NextRequest(`http://localhost/api/jobs/${jobID}/ui-actions`), {
    params: Promise.resolve({ jobID }) as unknown as { jobID: string },
  });
}

function post(jobID: string, body: unknown): Promise<Response> {
  return POST(
    new NextRequest(`http://localhost/api/jobs/${jobID}/ui-actions`, {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'content-type': 'application/json' },
    }),
    { params: Promise.resolve({ jobID }) as unknown as { jobID: string } },
  );
}

describe('GET /api/jobs/:id/ui-actions', () => {
  it('returns 404 when the job does not exist', async () => {
    const response = await get('missing');
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: 'Job not found' });
  });

  it('returns the shrink button for z_image_diffsynth_trainer', async () => {
    runningJob();
    const response = await get('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      actions: [
        {
          id: 'shrink_lora',
          icon: 'TriangleAlert',
          title: 'Multiply LoRA A and B by 0.9',
        },
      ],
    });
  });

  it('returns an empty list for another process type', async () => {
    runningJob('diffusion_trainer');
    const response = await get('job-1');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ actions: [] });
  });
});

describe('POST /api/jobs/:id/ui-actions', () => {
  it('returns 404 when the job does not exist', async () => {
    const response = await post('missing', { action: 'shrink_lora' });
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: 'Job not found' });
    expect(state.commands).toEqual([]);
  });

  it('returns 400 for another process type', async () => {
    runningJob('diffusion_trainer');
    const response = await post('job-1', { action: 'shrink_lora' });
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: 'ui actions require z_image_diffsynth_trainer' });
    expect(state.commands).toEqual([]);
  });

  it('returns 400 for an unknown action', async () => {
    runningJob();
    const response = await post('job-1', { action: 'nope' });
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: 'unknown ui action' });
    expect(state.commands).toEqual([]);
  });

  it('returns 409 when the job is not running', async () => {
    runningJob('z_image_diffsynth_trainer', 'stopped', false);
    const response = await post('job-1', { action: 'shrink_lora' });
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({ error: 'job is not running' });
    expect(state.commands).toEqual([]);
  });

  it('returns 409 when stop is set', async () => {
    runningJob('z_image_diffsynth_trainer', 'running', true);
    const response = await post('job-1', { action: 'shrink_lora' });
    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({ error: 'job is not running' });
    expect(state.commands).toEqual([]);
  });

  it('inserts one RuntimeCommand and does not write RuntimeParams', async () => {
    runningJob();
    const response = await post('job-1', { action: 'shrink_lora' });
    expect(response.status).toBe(200);
    expect(state.commands).toEqual([{ jobId: 'job-1', action: 'shrink_lora' }]);
    expect(await response.json()).toMatchObject({ id: 1, jobId: 'job-1', action: 'shrink_lora' });
  });
});
