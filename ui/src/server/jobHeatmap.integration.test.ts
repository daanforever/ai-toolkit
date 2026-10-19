import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';
import { TOOLKIT_ROOT, resolveVenvPython } from '@/paths';
import { runLoraHeatmap } from '@/server/jobHeatmap';

const flower = path.join(TOOLKIT_ROOT, 'temp', 'flower.safetensors');

describe('runLoraHeatmap', () => {
  it('scores the flower LoRA with the project venv', async () => {
    if (!fs.existsSync(flower)) return;
    const python = resolveVenvPython(TOOLKIT_ROOT);
    expect(fs.existsSync(python)).toBe(true);
    const payload = await runLoraHeatmap(python, TOOLKIT_ROOT, flower);
    const rows = payload.rows as unknown[];
    const cols = payload.cols as unknown[];
    const metrics = payload.metrics as Record<string, unknown[][]>;
    expect(rows).toHaveLength(30);
    expect(cols).toHaveLength(7);
    expect(metrics.frobenius).toHaveLength(30);
    expect(metrics.stable_rank[0]).toHaveLength(7);
    const file = payload.file as { step: number };
    expect(file.step).toBe(10);
  }, 60_000);
});
