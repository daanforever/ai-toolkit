'use client';

import { useState } from 'react';

function readStoredBoolean(key: string, defaultValue: boolean): boolean {
  if (typeof window === 'undefined') return defaultValue;
  try {
    const stored = localStorage.getItem(key);
    if (stored === 'true') return true;
    if (stored === 'false') return false;
    return defaultValue;
  } catch (error) {
    console.error('Failed to read persisted boolean:', error);
    return defaultValue;
  }
}

function writeStoredBoolean(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, String(value));
  } catch (error) {
    console.error('Failed to write persisted boolean:', error);
  }
}

export function usePersistedBoolean(key: string, defaultValue: boolean): [boolean, (value: boolean) => void] {
  const [value, setValue] = useState(() => readStoredBoolean(key, defaultValue));

  const setPersistedValue = (next: boolean) => {
    setValue(next);
    writeStoredBoolean(key, next);
  };

  return [value, setPersistedValue];
}
