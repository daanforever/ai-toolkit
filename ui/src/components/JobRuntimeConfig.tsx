'use client';

import { Job } from '@prisma/client';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { migrateJobConfig } from '@/app/jobs/new/jobConfig';
import { useNestedState } from '@/utils/hooks';
import { JobConfig } from '@/types';
import { defaultJobConfig } from '@/app/jobs/new/jobConfig';

const CONTENT_OR_STYLE_OPTIONS = [
  { value: 'balanced', label: 'Balanced' },
  { value: 'content', label: 'High Noise' },
  { value: 'style', label: 'Low Noise' },
  { value: 'gaussian', label: 'Gaussian (Normal)' },
  { value: 'fixed_cycle', label: 'Fixed Cycle' },
];

const TIMESTEP_TYPE_OPTIONS = [
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'linear', label: 'Linear' },
  { value: 'shift', label: 'Shift' },
  { value: 'weighted', label: 'Weighted' },
  { value: 'gaussian', label: 'Gaussian' },
];

const TRAIN_PATH = 'config.process[0].train';
const OPTIMIZER_PARAMS_PATH = 'config.process[0].train.optimizer_params';

export interface JobRuntimeConfigProps {
  job: Job;
  onRefresh?: () => void;
}

function parseJobConfig(raw: string | null): JobConfig | null {
  if (!raw || typeof raw !== 'string') return null;
  try {
    const parsed = JSON.parse(raw) as JobConfig;
    if (!parsed?.config?.process?.[0]?.train) return null;
    return migrateJobConfig(parsed);
  } catch {
    return null;
  }
}

export default function JobRuntimeConfig({ job, onRefresh }: JobRuntimeConfigProps) {
  const initialConfig = useMemo(() => parseJobConfig(job.job_config), [job.job_config]);
  const [config, setValue] = useNestedState<JobConfig | null>(initialConfig);

  useEffect(() => {
    const next = parseJobConfig(job.job_config);
    if (next) setValue(next, undefined);
  }, [job.job_config]);

  const [applyStatus, setApplyStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const train = config?.config?.process?.[0]?.train;
  const optimizerParams = train?.optimizer_params;
  const trainAny = train as Record<string, unknown> | undefined;
  const hasMaxLr = optimizerParams && 'max_lr' in optimizerParams;
  const hasMinLr = optimizerParams && 'min_lr' in optimizerParams;

  const timestepType = train?.timestep_type ?? defaultJobConfig.config.process[0].train.timestep_type;
  const contentOrStyle = train?.content_or_style ?? defaultJobConfig.config.process[0].train.content_or_style;
  const weightDecay = optimizerParams?.weight_decay ?? defaultJobConfig.config.process[0].train.optimizer_params.weight_decay;
  const maxLr = hasMaxLr && typeof (optimizerParams as { max_lr?: number }).max_lr === 'number'
    ? (optimizerParams as { max_lr: number }).max_lr
    : null;
  const minLr = hasMinLr && typeof (optimizerParams as { min_lr?: number }).min_lr === 'number'
    ? (optimizerParams as { min_lr: number }).min_lr
    : null;
  const gaussianMean = trainAny?.gaussian_mean != null
    ? Number(trainAny.gaussian_mean)
    : 500;
  const gaussianStd = trainAny?.gaussian_std != null
    ? Number(trainAny.gaussian_std)
    : 0.2;
  const batchSize = trainAny?.batch_size != null
    ? Number(trainAny.batch_size)
    : 1;

  const datasets = config?.config?.process?.[0]?.datasets;

  const handleApply = useCallback(async () => {
    if (!config || !train) return;
    setApplyStatus('loading');
    setErrorMessage(null);

    const fullConfig = { ...config };
    const process = fullConfig.config?.process?.[0];
    if (!process?.train) {
      setApplyStatus('error');
      setErrorMessage('Invalid config structure');
      return;
    }
    if (!process.train.optimizer_params) {
      process.train.optimizer_params = { weight_decay: 1e-4 };
    }
    process.train.timestep_type = timestepType;
    process.train.content_or_style = contentOrStyle;
    process.train.optimizer_params.weight_decay = weightDecay;
    if (hasMaxLr) {
      (process.train.optimizer_params as Record<string, number>).max_lr = maxLr ?? 1e-4;
    }
    if (hasMinLr) {
      (process.train.optimizer_params as Record<string, number>).min_lr = minLr ?? 1e-6;
    }
    (process.train as Record<string, number>).gaussian_mean = gaussianMean;
    (process.train as Record<string, number>).gaussian_std = gaussianStd;
    (process.train as Record<string, number>).batch_size = batchSize;

    try {
      const postRes = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: job.id,
          name: job.name,
          gpu_ids: job.gpu_ids,
          job_config: fullConfig,
        }),
      });
      if (!postRes.ok) {
        const data = await postRes.json().catch(() => ({}));
        throw new Error(data.error || postRes.statusText);
      }

      const patchBody: {
        max_lr?: number;
        min_lr?: number;
        weight_decay?: number;
        content_or_style?: string;
        timestep_type?: string;
        gaussian_mean?: number;
        gaussian_std?: number;
        network_weights?: number[];
        batch_size?: number;
      } = {
        weight_decay: weightDecay,
        content_or_style: contentOrStyle,
        timestep_type: timestepType,
        gaussian_mean: gaussianMean,
        gaussian_std: gaussianStd,
        batch_size: batchSize,
      };
      if (hasMaxLr && maxLr != null) patchBody.max_lr = maxLr;
      if (hasMinLr && minLr != null) patchBody.min_lr = minLr;
      if (process.datasets?.length) {
        patchBody.network_weights = process.datasets.map((d: { network_weight?: number }) => d.network_weight ?? 1);
      }

      const patchRes = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patchBody),
      });
      if (!patchRes.ok) {
        const data = await patchRes.json().catch(() => ({}));
        throw new Error(data.error || patchRes.statusText);
      }

      setApplyStatus('success');
      onRefresh?.();
    } catch (e) {
      setApplyStatus('error');
      setErrorMessage(e instanceof Error ? e.message : 'Failed to apply');
    }
  }, [
    config,
    job.id,
    job.name,
    job.gpu_ids,
    train,
    timestepType,
    contentOrStyle,
    weightDecay,
    maxLr,
    hasMaxLr,
    minLr,
    hasMinLr,
    gaussianMean,
    gaussianStd,
    batchSize,
    datasets,
    onRefresh,
  ]);

  if (!initialConfig) {
    return (
      <div className="p-6 text-gray-400">
        Config is unavailable or corrupted.
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6 p-6">
      <h2 className="text-lg font-medium text-gray-200">Runtime config</h2>
      <p className="text-sm text-gray-400">
        Values are loaded from job_config. Apply saves the config and immediately applies the parameters to the running job.
      </p>

      <div className="space-y-4">
        <div className="flex items-end gap-4 flex-wrap">
          {hasMaxLr && (
            <div className="space-y-2 flex-1 min-w-[140px]">
              <p className="text-xs text-gray-400">Max LR</p>
              <input
                type="number"
                min={1e-6}
                step="any"
                placeholder="e.g. 1e-4"
                value={maxLr ?? ''}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? null : parseFloat(v);
                  setValue(num != null && Number.isFinite(num) ? num : undefined, 'config.process[0].train.optimizer_params.max_lr');
                  setApplyStatus('idle');
                }}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
          )}
          {hasMinLr && (
            <div className="space-y-2 flex-1 min-w-[140px]">
              <p className="text-xs text-gray-400">Min LR</p>
              <input
                type="number"
                min={1e-6}
                step="any"
                placeholder="e.g. 1e-6"
                value={minLr ?? ''}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? null : parseFloat(v);
                  setValue(num != null && Number.isFinite(num) ? num : undefined, 'config.process[0].train.optimizer_params.min_lr');
                  setApplyStatus('idle');
                }}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
          )}
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Weight decay</p>
            <input
              type="number"
              min={0}
              step="any"
              placeholder="e.g. 0.01 or 0"
              value={weightDecay}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setValue(v, `${OPTIMIZER_PARAMS_PATH}.weight_decay`);
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-xs text-gray-400">Timestep Type / Timestep Bias</p>
          <div className="flex items-center gap-2 flex-wrap">
            <select
              value={timestepType}
              onChange={(e) => {
                setValue(e.target.value, `${TRAIN_PATH}.timestep_type`);
                setApplyStatus('idle');
              }}
              className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-w-[120px]"
            >
              {TIMESTEP_TYPE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <select
              value={contentOrStyle}
              onChange={(e) => {
                setValue(e.target.value, `${TRAIN_PATH}.content_or_style`);
                setApplyStatus('idle');
              }}
              className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-w-[140px]"
            >
              {CONTENT_OR_STYLE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-xs text-gray-400">Gaussian (mean / std)</p>
              <input
                type="number"
                min={0}
                max={999}
                step="any"
                placeholder="mean (0–999)"
                value={gaussianMean}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_mean`);
                  setApplyStatus('idle');
                }}
                className="w-24 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
              <input
                type="number"
                min={1e-6}
                step="any"
                placeholder="std (&gt;0)"
                value={gaussianStd}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_std`);
                  setApplyStatus('idle');
                }}
                className="w-24 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-xs text-gray-400">Batch size</p>
              <input
                type="number"
                min={1}
                max={128}
                step={1}
                placeholder="e.g. 1"
                value={batchSize}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? 1 : parseInt(v, 10);
                  if (Number.isInteger(num) && num >= 1) {
                    setValue(num, 'config.process[0].train.batch_size');
                    setApplyStatus('idle');
                  }
                }}
                className="w-20 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {Array.isArray(datasets) && datasets.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Datasets — Network weight</p>
            <div className="flex items-center gap-2 flex-wrap">
              {datasets.map((d: { network_weight?: number }, i: number) => (
                <div key={i} className="space-y-1 min-w-[100px]">
                  <label className="block text-xs text-gray-500">Dataset {i + 1}</label>
                  <input
                    type="number"
                    min={1e-6}
                    step="any"
                    value={d.network_weight ?? 1}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      if (Number.isFinite(v) && v > 0) {
                        setValue(v, `config.process[0].datasets[${i}].network_weight`);
                        setApplyStatus('idle');
                      }
                    }}
                    className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="pt-2">
          <button
            type="button"
            onClick={handleApply}
            disabled={applyStatus === 'loading'}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {applyStatus === 'loading' ? '…' : 'Apply'}
          </button>
          {applyStatus === 'success' && <span className="ml-3 text-xs text-green-500">Applied.</span>}
          {applyStatus === 'error' && (
            <span className="ml-3 text-xs text-rose-500">{errorMessage ?? 'Failed to apply.'}</span>
          )}
        </div>
      </div>
    </div>
  );
}
