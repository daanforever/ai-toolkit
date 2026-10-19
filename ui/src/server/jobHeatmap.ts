import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

/** flower.safetensors (210 modules, 70MB) scores in ~0.25s. 60s covers larger checkpoints. */
export const HEATMAP_TIMEOUT_MS = 60_000;

export function pickLatestSafetensors(jobFolder: string): string | null {
  if (!fs.existsSync(jobFolder)) return null;
  let latest: { file: string; mtime: number } | null = null;
  for (const name of fs.readdirSync(jobFolder)) {
    if (!name.endsWith('.safetensors')) continue;
    const file = path.join(jobFolder, name);
    const stats = fs.statSync(file);
    if (!stats.isFile()) continue;
    if (latest == null || stats.mtimeMs > latest.mtime) {
      latest = { file, mtime: stats.mtimeMs };
    }
  }
  return latest == null ? null : latest.file;
}

export function runLoraHeatmap(
  pythonPath: string,
  toolkitRoot: string,
  filePath: string,
): Promise<Record<string, unknown>> {
  const script = path.join(toolkitRoot, 'toolkit', 'lora_heatmap.py');
  if (!fs.existsSync(script)) {
    return Promise.reject(new Error(`lora_heatmap.py not found at ${script}`));
  }

  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath, [script, filePath], {
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
      finish(() => reject(new Error(`heatmap timed out after ${HEATMAP_TIMEOUT_MS}ms`)));
    }, HEATMAP_TIMEOUT_MS);

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
          reject(new Error(stderr.trim() || `heatmap exited with code ${code}`));
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(stdout);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          reject(new Error(`heatmap returned invalid JSON: ${message}`));
          return;
        }
        if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          reject(new Error('heatmap returned a non-object JSON value'));
          return;
        }
        resolve(parsed as Record<string, unknown>);
      });
    });
  });
}
