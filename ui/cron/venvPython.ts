import fs from 'fs';
import path from 'path';

/** Python from `.venv` or `venv` under the toolkit root, otherwise `python` on PATH. */
export function resolveVenvPython(toolkitRoot: string): string {
  const windows = process.platform === 'win32';
  let venvRoot: string | null = null;
  if (fs.existsSync(path.join(toolkitRoot, '.venv'))) {
    venvRoot = path.join(toolkitRoot, '.venv');
  } else if (fs.existsSync(path.join(toolkitRoot, 'venv'))) {
    venvRoot = path.join(toolkitRoot, 'venv');
  }
  if (venvRoot == null) {
    return 'python';
  }
  if (windows) {
    return path.join(venvRoot, 'Scripts', 'python.exe');
  }
  return path.join(venvRoot, 'bin', 'python');
}
