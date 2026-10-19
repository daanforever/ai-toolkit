import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import { resolveVenvPython } from '@/paths';

const roots: string[] = [];

function makeRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'venv-python-'));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length > 0) {
    const root = roots.pop();
    if (root) fs.rmSync(root, { recursive: true, force: true });
  }
});

function expected(root: string, venvName: '.venv' | 'venv'): string {
  if (process.platform === 'win32') {
    return path.join(root, venvName, 'Scripts', 'python.exe');
  }
  return path.join(root, venvName, 'bin', 'python');
}

describe('resolveVenvPython', () => {
  it('prefers .venv over venv', () => {
    const root = makeRoot();
    fs.mkdirSync(path.join(root, '.venv'));
    fs.mkdirSync(path.join(root, 'venv'));
    expect(resolveVenvPython(root)).toBe(expected(root, '.venv'));
  });

  it('uses venv when .venv is absent', () => {
    const root = makeRoot();
    fs.mkdirSync(path.join(root, 'venv'));
    expect(resolveVenvPython(root)).toBe(expected(root, 'venv'));
  });

  it('falls back to python on PATH', () => {
    const root = makeRoot();
    expect(resolveVenvPython(root)).toBe('python');
  });
});
