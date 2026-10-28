'use client';

import { Job } from '@prisma/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import axios from 'axios';
import { apiClient } from '@/utils/api';

interface ScaleLrFile {
  filename?: string;
  step?: number | null;
}

interface ScaleLrData {
  available: boolean;
  reason?: string;
  indices?: number[];
  weights?: number[];
  n?: number;
  mean?: number;
  std?: number;
  mask?: string[];
  file?: ScaleLrFile;
}

interface Props {
  job: Job;
}

const OFF_REASON = 'scale_lr_by_index is off';

function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;
    if (data && typeof data === 'object' && 'error' in data && typeof (data as { error: unknown }).error === 'string') {
      return (data as { error: string }).error;
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'Failed to load scale LR curve';
}

function formatWeight(value: number): string {
  if (!Number.isFinite(value)) return '';
  return value.toPrecision(4);
}

function formatSetting(value: number): string {
  if (!Number.isFinite(value)) return '';
  return String(value);
}

export default function JobScaleLrGraph({ job }: Props) {
  const [data, setData] = useState<ScaleLrData | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef(0);

  const load = useCallback(async () => {
    const request = ++requestRef.current;
    setStatus('loading');
    setError(null);
    try {
      const response = await apiClient.get<ScaleLrData>(`/api/jobs/${job.id}/scale-lr`);
      if (requestRef.current !== request) return;
      setData(response.data);
      setStatus('success');
    } catch (err) {
      if (requestRef.current !== request) return;
      setData(null);
      setStatus('error');
      setError(errorMessage(err));
    }
  }, [job.id]);

  useEffect(() => {
    setData(null);
    void load();
    return () => {
      requestRef.current += 1;
    };
  }, [load]);

  const points = useMemo(() => {
    if (!data?.available || !Array.isArray(data.indices) || !Array.isArray(data.weights)) return [];
    if (data.indices.length !== data.weights.length || data.indices.length < 2) return [];
    const rows: { index: number; weight: number }[] = [];
    for (let i = 0; i < data.indices.length; i++) {
      const index = data.indices[i];
      const weight = data.weights[i];
      if (typeof index !== 'number' || typeof weight !== 'number' || !Number.isFinite(weight)) return [];
      rows.push({ index, weight });
    }
    return rows;
  }, [data]);

  const hasCurve = points.length >= 2;
  const off = status === 'success' && data?.available === false && data.reason === OFF_REASON;
  const noFile = status === 'success' && data?.available === false && !data.reason;
  const rejected = status === 'success' && data?.available === false && !!data.reason && data.reason !== OFF_REASON;
  const missingCurve = status === 'success' && data?.available === true && !hasCurve;

  let statusText = 'Loading...';
  if (status === 'loading') statusText = 'Loading...';
  else if (status === 'error') statusText = 'Error';
  else if (off) statusText = 'Scale LR by index is off';
  else if (noFile) statusText = 'No saved LoRA yet';
  else if (rejected) statusText = 'Error';
  else if (hasCurve) statusText = `${data?.n ?? points.length} layers`;
  else statusText = 'Error';

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="h-2 w-2 rounded-full bg-amber-400" />
          <h2 className="text-gray-100 text-sm font-medium">Scale LR by index</h2>
          <span className="text-xs text-gray-400 truncate">{statusText}</span>
        </div>
        <button
          type="button"
          onClick={() => {
            void load();
          }}
          disabled={status === 'loading'}
          className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700 disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      <div className="px-4 pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-72 relative">
          {status === 'loading' && !hasCurve && (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              Loading scale LR...
            </div>
          )}
          {status === 'error' && (
            <div className="h-full w-full flex items-center justify-center text-sm text-red-300 px-6 text-center">
              {error || 'Failed to load scale LR curve.'}
            </div>
          )}
          {off && (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              Scale LR by index is off.
            </div>
          )}
          {noFile && (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              No saved LoRA yet.
            </div>
          )}
          {rejected && (
            <div className="h-full w-full flex items-center justify-center text-sm text-red-300 px-6 text-center">
              {data?.reason}
            </div>
          )}
          {missingCurve && (
            <div className="h-full w-full flex items-center justify-center text-sm text-red-300 px-6 text-center">
              Scale LR response is missing weights.
            </div>
          )}
          {hasCurve && status !== 'error' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points} margin={{ top: 16, right: 16, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="index"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={28}
                  tickFormatter={(value: number) => `layer ${value}`}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={72}
                  tickFormatter={formatWeight}
                  label={{ value: 'w', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.55)' }}
                />
                <Tooltip
                  cursor={{ stroke: 'rgba(251,191,36,0.35)', strokeWidth: 1 }}
                  contentStyle={{
                    background: 'rgba(17,24,39,0.96)',
                    border: '1px solid rgba(31,41,55,1)',
                    borderRadius: 10,
                    color: 'rgba(255,255,255,0.9)',
                    fontSize: 12,
                  }}
                  labelStyle={{ color: 'rgba(255,255,255,0.75)' }}
                  labelFormatter={(label: unknown) => `layer ${label}`}
                  formatter={(value: unknown) => [formatWeight(Number(value)), 'w']}
                />
                <Line
                  type="monotone"
                  dataKey="weight"
                  name="w"
                  stroke="rgba(251,191,36,1)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        {hasCurve && data && (
          <div className="mt-2 text-[11px] text-gray-500">
            w · mean {formatSetting(Number(data.mean))} · std {formatSetting(Number(data.std))}
            {data.file?.filename ? ` · ${data.file.filename}` : ''}
            {data.file?.step != null ? ` · step ${data.file.step}` : ''}
            {data.mask && data.mask.length > 0 ? ` · mask ${data.mask.join(', ')}` : ''}
          </div>
        )}
      </div>
    </div>
  );
}
