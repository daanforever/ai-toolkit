import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

/** Same budget as the LoRA heatmap. The curve reads key names and a small tensor. */
export const SCALE_LR_TIMEOUT_MS = 60_000;

const OFF_REASON = 'scale_lr_by_index is off';

export type ScaleLrResolution =
  | { ok: true; mean: number; std: number; mask: string[] }
  | { ok: false; reason: string };

interface RuntimeScaleLr {
  runtime_scale_lr_by_index?: boolean | null;
  runtime_scale_lr_mean?: number | null;
  runtime_scale_lr_std?: number | null;
}

export interface ScaleLrJobInput {
  job_config: string;
  runtimeParams?: RuntimeScaleLr | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function optimizerParams(jobConfig: string): Record<string, unknown> | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jobConfig);
  } catch (error) {
    if (error instanceof SyntaxError) return null;
    throw error;
  }
  if (!isRecord(parsed)) return null;
  const config = parsed.config;
  if (!isRecord(config) || !Array.isArray(config.process)) return null;
  const process = config.process[0];
  if (!isRecord(process)) return null;
  const train = process.train;
  if (!isRecord(train)) return null;
  const params = train.optimizer_params;
  return isRecord(params) ? params : null;
}

function enabledFlag(runtime: RuntimeScaleLr | null, params: Record<string, unknown> | null): boolean {
  if (runtime && runtime.runtime_scale_lr_by_index != null) {
    return runtime.runtime_scale_lr_by_index === true;
  }
  const value = params?.scale_lr_by_index;
  return value === true || value === 1;
}

function meanStd(
  runtime: RuntimeScaleLr | null,
  params: Record<string, unknown> | null,
): { mean: number | null; std: number | null } {
  const runtimeMeanSet = runtime != null && runtime.runtime_scale_lr_mean != null;
  const runtimeStdSet = runtime != null && runtime.runtime_scale_lr_std != null;
  if (runtimeMeanSet || runtimeStdSet) {
    return {
      mean: finiteNumber(runtime?.runtime_scale_lr_mean),
      std: finiteNumber(runtime?.runtime_scale_lr_std),
    };
  }
  return {
    mean: finiteNumber(params?.scale_lr_mean),
    std: finiteNumber(params?.scale_lr_std),
  };
}

function configMask(value: unknown): string[] | 'invalid' {
  if (value == null) return [];
  if (!Array.isArray(value)) return 'invalid';
  const mask: string[] = [];
  for (const item of value) {
    if (typeof item !== 'string' || item.length === 0) return 'invalid';
    mask.push(item);
  }
  return mask;
}

/**
 * Runtime mean/std win when either column is set. The mask stays on the job
 * config: the trainer clears runtime columns after apply, and the saved config
 * is what the next start uses.
 */
export function resolveScaleLrSettings(job: ScaleLrJobInput): ScaleLrResolution {
  const params = optimizerParams(job.job_config);
  const runtime = job.runtimeParams ?? null;
  if (!enabledFlag(runtime, params)) {
    return { ok: false, reason: OFF_REASON };
  }

  const { mean, std } = meanStd(runtime, params);
  if (mean == null) {
    return { ok: false, reason: 'scale_lr_mean must be a finite number' };
  }
  if (std == null || !(std > 0)) {
    return { ok: false, reason: 'scale_lr_std must be a finite number > 0' };
  }

  const mask = configMask(params?.scale_lr_mask);
  if (mask === 'invalid') {
    return { ok: false, reason: 'scale_lr_mask must be an array of non-empty strings' };
  }
  return { ok: true, mean, std, mask };
}

export function runScaleLrCurve(
  pythonPath: string,
  toolkitRoot: string,
  filePath: string,
  mean: number,
  std: number,
  mask: string[],
): Promise<Record<string, unknown>> {
  const script = path.join(toolkitRoot, 'toolkit', 'scale_lr_curve.py');
  if (!fs.existsSync(script)) {
    return Promise.reject(new Error(`scale_lr_curve.py not found at ${script}`));
  }

  const args = [script, filePath, '--mean', String(mean), '--std', String(std)];
  for (const item of mask) {
    args.push('--mask', item);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath, args, {
      cwd: toolkitRoot,
      windowsHide: true,
      env: {
        ...process.env,
        CUDA_VISIBLE_DEVICES: '-1',
        PYTHONIOENCODING: 'utf-8',
      },
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const finish = (settle: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      settle();
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(() => reject(new Error(`scale-lr timed out after ${SCALE_LR_TIMEOUT_MS}ms`)));
    }, SCALE_LR_TIMEOUT_MS);

    child.stdout.on('data', (chunk: Buffer | string) => {
      stdout += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer | string) => {
      stderr += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    });
    child.on('error', error => {
      finish(() => reject(error));
    });
    child.on('close', code => {
      finish(() => {
        if (code !== 0) {
          reject(new Error(stderr.trim() || `scale-lr exited with code ${code}`));
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(stdout);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          reject(new Error(`scale-lr returned invalid JSON: ${message}`));
          return;
        }
        if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          reject(new Error('scale-lr returned a non-object JSON value'));
          return;
        }
        const payload = parsed as Record<string, unknown>;
        if (!Array.isArray(payload.indices) || !Array.isArray(payload.weights) || typeof payload.n !== 'number') {
          reject(new Error('scale-lr response is missing indices, weights, or n'));
          return;
        }
        resolve(payload);
      });
    });
  });
}
