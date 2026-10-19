'use client';

import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { apiClient } from '@/utils/api';

export interface HeatmapFileInfo {
  filename: string;
  path: string;
  mtime_ms: number;
  step: number | null;
  epoch: number | null;
}

export interface HeatmapData {
  available: boolean;
  file?: HeatmapFileInfo;
  rows?: string[];
  cols?: string[];
  metrics?: Record<string, (number | null)[][]>;
}

export type HeatmapStatus = 'idle' | 'loading' | 'success' | 'error';

function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;
    if (data && typeof data === 'object' && 'error' in data && typeof (data as { error: unknown }).error === 'string') {
      return (data as { error: string }).error;
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'Failed to load heatmap';
}

export default function useJobHeatmap(jobID: string) {
  const [data, setData] = useState<HeatmapData | null>(null);
  const [status, setStatus] = useState<HeatmapStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setStatus('idle');
    setError(null);
  }, [jobID]);

  const refresh = useCallback(async () => {
    if (!jobID) return;
    setStatus('loading');
    setError(null);
    try {
      const response = await apiClient.get<HeatmapData>(`/api/jobs/${jobID}/heatmap`);
      setData(response.data);
      setStatus('success');
    } catch (err) {
      setData(null);
      setStatus('error');
      setError(errorMessage(err));
    }
  }, [jobID]);

  return { data, status, error, refresh };
}
