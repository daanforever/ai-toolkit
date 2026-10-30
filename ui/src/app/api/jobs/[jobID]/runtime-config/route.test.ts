import { NextRequest } from 'next/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

const state = vi.hoisted(() => ({
  job: null as { id: string; job_config: string } | null,
  writes: [] as Array<{ update: Record<string, unknown>; create: Record<string, unknown> }>,
}));

vi.mock('@prisma/client', () => ({
  PrismaClient: class {
    job = {
      findUnique: async () => state.job,
    };
    runtimeParams = {
      upsert: async ({
        update,
        create,
      }: {
        update: Record<string, unknown>;
        create: Record<string, unknown>;
      }) => {
        state.writes.push({ update, create });
        return { ...create, ...update };
      },
    };
  },
}));

import { PATCH } from '@/app/api/jobs/[jobID]/runtime-config/route';

function jobConfig(arch: string, networkType: string): string {
  return JSON.stringify({
    config: {
      process: [
        {
          model: { arch },
          network: { type: networkType },
        },
      ],
    },
  });
}

function zImageLoraJob() {
  return {
    id: 'job-1',
    job_config: jobConfig('zimage_diffsynth', 'lora'),
  };
}

afterEach(() => {
  state.job = null;
  state.writes = [];
});

function patch(body: unknown, jobID = 'job-1') {
  return PATCH(
    new NextRequest(`http://localhost/api/jobs/${jobID}/runtime-config`, {
      method: 'PATCH',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
    }),
    { params: Promise.resolve({ jobID }) },
  );
}

function patchRaw(raw: string, jobID = 'job-1') {
  return PATCH(
    new NextRequest(`http://localhost/api/jobs/${jobID}/runtime-config`, {
      method: 'PATCH',
      body: raw,
      headers: { 'Content-Type': 'application/json' },
    }),
    { params: Promise.resolve({ jobID }) },
  );
}

describe('PATCH /api/jobs/:id/runtime-config linear_alpha', () => {
  it('returns 404 when the job does not exist', async () => {
    const response = await patch({ linear_alpha: 8 });
    expect(response.status).toBe(404);
    expect(state.writes).toEqual([]);
  });

  it('writes runtime_linear_alpha for a zimage_diffsynth LoRA job', async () => {
    state.job = zImageLoraJob();
    const response = await patch({ linear_alpha: 8, lr: 1e-4 });
    expect(response.status).toBe(200);
    expect(state.writes).toHaveLength(1);
    expect(state.writes[0].update).toMatchObject({
      runtime_linear_alpha: 8,
      runtime_lr: 1e-4,
    });
    expect(state.writes[0].create.jobId).toBe('job-1');
  });

  it('rejects 0, negative, non-integer, non-finite, and string values without writing', async () => {
    state.job = zImageLoraJob();
    for (const linear_alpha of [0, -1, 8.5, Number.POSITIVE_INFINITY, Number.NaN, '8']) {
      state.writes = [];
      const response = await PATCH(
        { json: async () => ({ linear_alpha, lr: 1e-4 }) } as unknown as NextRequest,
        { params: Promise.resolve({ jobID: 'job-1' }) },
      );
      expect(response.status).toBe(400);
      expect((await response.json()).error).toBe('linear_alpha must be an integer > 0');
      expect(state.writes).toEqual([]);
    }
  });

  it('rejects invalid JSON without writing', async () => {
    state.job = zImageLoraJob();
    const response = await patchRaw('{"linear_alpha":');
    expect(response.status).toBe(400);
    expect((await response.json()).error).toBe('Invalid JSON body');
    expect(state.writes).toEqual([]);
  });

  it('rejects a non-LoRA zimage job', async () => {
    state.job = {
      id: 'job-1',
      job_config: jobConfig('zimage_diffsynth', 'lycoris'),
    };
    const response = await patch({ linear_alpha: 8, lr: 1e-4 });
    expect(response.status).toBe(400);
    expect((await response.json()).error).toBe(
      'linear_alpha is only supported for zimage_diffsynth LoRA jobs',
    );
    expect(state.writes).toEqual([]);
  });

  it('rejects a non-zimage LoRA job', async () => {
    state.job = {
      id: 'job-1',
      job_config: jobConfig('flex1', 'lora'),
    };
    const response = await patch({ linear_alpha: 8 });
    expect(response.status).toBe(400);
    expect(state.writes).toEqual([]);
  });

  it('rejects an unreadable job_config', async () => {
    state.job = { id: 'job-1', job_config: '{not json' };
    const response = await patch({ linear_alpha: 8 });
    expect(response.status).toBe(400);
    expect((await response.json()).error).toBe('job_config is invalid');
    expect(state.writes).toEqual([]);
  });

  it('still accepts other fields for a non-zimage job', async () => {
    state.job = {
      id: 'job-1',
      job_config: jobConfig('flex1', 'lokr'),
    };
    const response = await patch({ lr: 2e-4 });
    expect(response.status).toBe(200);
    expect(state.writes[0].update).toEqual({ runtime_lr: 2e-4 });
    expect(state.writes[0].update).not.toHaveProperty('runtime_linear_alpha');
  });
});

describe('PATCH /api/jobs/:id/runtime-config dataset_num_repeats', () => {
  it('stores a JSON array of repeats', async () => {
    state.job = zImageLoraJob();
    const response = await patch({ dataset_num_repeats: [1, 4], network_weights: [1, 0.5] });
    expect(response.status).toBe(200);
    expect(state.writes[0].update).toMatchObject({
      runtime_dataset_num_repeats: JSON.stringify([1, 4]),
      runtime_network_weights: JSON.stringify([1, 0.5]),
    });
  });

  it('rejects an empty array, zero, negatives, and non-integers without writing', async () => {
    state.job = zImageLoraJob();
    const cases = [
      { body: { dataset_num_repeats: [] }, error: 'dataset_num_repeats must be a non-empty array' },
      { body: { dataset_num_repeats: [0] }, error: 'dataset_num_repeats[0] must be an integer >= 1' },
      { body: { dataset_num_repeats: [-2] }, error: 'dataset_num_repeats[0] must be an integer >= 1' },
      { body: { dataset_num_repeats: [1.5] }, error: 'dataset_num_repeats[0] must be an integer >= 1' },
      { body: { dataset_num_repeats: [1, 2.2] }, error: 'dataset_num_repeats[1] must be an integer >= 1' },
    ];
    for (const { body, error } of cases) {
      state.writes = [];
      const response = await patch(body);
      expect(response.status).toBe(400);
      expect((await response.json()).error).toBe(error);
      expect(state.writes).toEqual([]);
    }
  });
});
