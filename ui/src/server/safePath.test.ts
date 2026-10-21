import path from 'path';
import { describe, expect, it } from 'vitest';
import { isPathInsideAny, isPathInsideRoot } from '@/server/safePath';

describe('isPathInsideRoot', () => {
  it('accepts a file when the root uses the other slash', () => {
    const root = path.resolve('output');
    const file = path.join(root, '045_eps_9', 'samples', '1792618818662__000001780_2.jpg');
    const rootSlashed = root.split(path.sep).join('/');
    const fileSlashed = file.split(path.sep).join(path.sep === '\\' ? '\\' : '/');
    expect(isPathInsideRoot(fileSlashed, rootSlashed)).toBe(true);
    expect(isPathInsideRoot(file.split(path.sep).join('/'), root)).toBe(true);
  });

  it('rejects a sibling directory that shares the root prefix', () => {
    const root = path.resolve('output');
    const sibling = path.join(`${root}-other`, 'a.jpg');
    expect(isPathInsideRoot(path.join(root, 'a.jpg'), root)).toBe(true);
    expect(isPathInsideRoot(sibling, root)).toBe(false);
  });

  it('rejects a path that escapes the root', () => {
    const root = path.resolve('output');
    const outside = path.resolve(root, '..', 'secret.jpg');
    expect(isPathInsideRoot(outside, root)).toBe(false);
  });

  it('allows a filename that contains two dots', () => {
    const root = path.resolve('output');
    expect(isPathInsideRoot(path.join(root, 'a..b.jpg'), root)).toBe(true);
  });

  it('matches the reported windows sample path only against the output root', () => {
    if (process.platform !== 'win32') return;
    const file = 'D:\\prod\\output\\045_eps_9\\samples\\1792618818662__000001780_2.jpg';
    expect(isPathInsideAny(file, ['D:/prod/datasets', 'D:/prod/output', 'D:\\prod\\data'])).toBe(true);
    expect(isPathInsideRoot(file, 'D:/prod/datasets')).toBe(false);
    expect(isPathInsideRoot(file, 'D:\\prod\\data')).toBe(false);
  });
});
