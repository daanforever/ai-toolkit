import path from 'path';

export function isPathInsideRoot(filePath: string, root: string): boolean {
  if (typeof filePath !== 'string' || typeof root !== 'string' || filePath === '' || root === '') {
    return false;
  }
  const relative = path.relative(path.resolve(root), path.resolve(filePath));
  if (relative === '' || path.isAbsolute(relative)) {
    return false;
  }
  return relative !== '..' && !relative.startsWith(`..${path.sep}`);
}

export function isPathInsideAny(filePath: string, roots: string[]): boolean {
  return roots.some(root => isPathInsideRoot(filePath, root));
}
