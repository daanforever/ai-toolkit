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
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'gaussian_bimodal', label: 'Bimodal Gaussian' },
  { value: 'fixed_cycle', label: 'Fixed Cycle' },
];

const TIMESTEP_TYPE_OPTIONS = [
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'linear', label: 'Linear' },
  { value: 'shift', label: 'Shift' },
  { value: 'weighted', label: 'Weighted' },
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'gaussian_bimodal', label: 'Bimodal Gaussian' },
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

/** Last path segment (folder name), supports both / and \ separators. */
function folderNameFromPath(folderPath: string | undefined): string {
  if (!folderPath || typeof folderPath !== 'string') return '';
  const normalized = folderPath.replace(/\\/g, '/').replace(/\/+$/, '');
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash === -1 ? normalized : normalized.slice(lastSlash + 1);
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
  const hasMinLr = optimizerParams && 'min_lr' in optimizerParams;

  const timestepType = train?.timestep_type ?? defaultJobConfig.config.process[0].train.timestep_type;
  const contentOrStyle = train?.content_or_style ?? defaultJobConfig.config.process[0].train.content_or_style;
  const weightDecay = optimizerParams?.weight_decay ?? defaultJobConfig.config.process[0].train.optimizer_params.weight_decay;
  const beta1 = optimizerParams?.beta1 != null
    ? Number(optimizerParams.beta1)
    : null;
  const beta2 = optimizerParams?.beta2 != null
    ? Number(optimizerParams.beta2)
    : 0.99;
  const lr = typeof train?.lr === 'number' && Number.isFinite(train.lr)
    ? train.lr
    : (defaultJobConfig.config.process[0].train.lr ?? 1e-4);
  const minLr = hasMinLr && typeof (optimizerParams as { min_lr?: number }).min_lr === 'number'
    ? (optimizerParams as { min_lr: number }).min_lr
    : null;
  const gaussianMean = trainAny?.gaussian_mean != null
    ? Number(trainAny.gaussian_mean)
    : 500;
  const gaussianStd = trainAny?.gaussian_std != null
    ? Number(trainAny.gaussian_std)
    : 0.2;
  const gaussianMean2 = trainAny?.gaussian_mean_2 != null
    ? Number(trainAny.gaussian_mean_2)
    : 750;
  const gaussianStd2 = trainAny?.gaussian_std_2 != null
    ? Number(trainAny.gaussian_std_2)
    : 0.2;
  const showGaussianPeak2 =
    contentOrStyle === 'gaussian_bimodal' || timestepType === 'gaussian_bimodal';
  const batchSize = trainAny?.batch_size != null
    ? Number(trainAny.batch_size)
    : 1;
  const gradientAccumulation = trainAny?.gradient_accumulation != null
    ? Number(trainAny.gradient_accumulation)
    : 1;
  const minSnrGamma = trainAny?.min_snr_gamma != null
    ? Number(trainAny.min_snr_gamma)
    : 5;

  const saveAny = config?.config?.process?.[0]?.save as Record<string, unknown> | undefined;
  const saveEvery = saveAny?.save_every != null
    ? Number(saveAny.save_every)
    : 250;

  const sampleAny = config?.config?.process?.[0]?.sample as Record<string, unknown> | undefined;
  const sampleEvery = sampleAny?.sample_every != null
    ? Number(sampleAny.sample_every)
    : 250;

  const loggingAny = config?.config?.process?.[0]?.logging as { debug?: boolean } | undefined;
  const debug = loggingAny?.debug ?? false;

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
    (process.train.optimizer_params as Record<string, number | null>).beta1 = beta1;
    (process.train.optimizer_params as Record<string, number>).beta2 = beta2;
    (process.train as Record<string, number>).lr = lr;
    if (hasMinLr) {
      (process.train.optimizer_params as Record<string, number>).min_lr = minLr ?? 1e-6;
    }
    (process.train as Record<string, number>).gaussian_mean = gaussianMean;
    (process.train as Record<string, number>).gaussian_std = gaussianStd;
    (process.train as Record<string, number>).gaussian_mean_2 = gaussianMean2;
    (process.train as Record<string, number>).gaussian_std_2 = gaussianStd2;
    (process.train as Record<string, number>).batch_size = batchSize;
    (process.train as Record<string, number>).gradient_accumulation = gradientAccumulation;
    (process.train as Record<string, number>).min_snr_gamma = minSnrGamma;

    if (!process.save) {
      process.save = { save_every: 250, dtype: 'bf16', max_step_saves_to_keep: 4, save_format: 'safetensors', push_to_hub: false };
    }
    (process.save as Record<string, number>).save_every = saveEvery;

    if (!process.sample) {
      process.sample = {
        sampler: 'flowmatch',
        sample_every: 250,
        width: 1024,
        height: 1024,
        samples: [],
        neg: '',
        seed: 42,
      };
    }
    (process.sample as Record<string, number>).sample_every = sampleEvery;

    if (!process.logging) {
      process.logging = { debug: false };
    }
    (process.logging as Record<string, boolean>).debug = debug;

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
        lr?: number;
        min_lr?: number;
        weight_decay?: number;
        beta1?: number | null;
        beta2?: number;
        content_or_style?: string;
        timestep_type?: string;
        gaussian_mean?: number;
        gaussian_std?: number;
        gaussian_mean_2?: number;
        gaussian_std_2?: number;
        network_weights?: number[];
        batch_size?: number;
        gradient_accumulation?: number;
        save_every?: number;
        sample_every?: number;
        min_snr_gamma?: number;
        debug?: boolean;
      } = {
        weight_decay: weightDecay,
        beta1: beta1 === 0 ? null : beta1,
        beta2: beta2,
        content_or_style: contentOrStyle,
        timestep_type: timestepType,
        gaussian_mean: gaussianMean,
        gaussian_std: gaussianStd,
        gaussian_mean_2: gaussianMean2,
        gaussian_std_2: gaussianStd2,
        batch_size: batchSize,
        gradient_accumulation: gradientAccumulation,
        save_every: saveEvery,
        sample_every: sampleEvery,
        min_snr_gamma: minSnrGamma,
        debug,
      };
      if (lr != null && Number.isFinite(lr)) patchBody.lr = lr;
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
    beta1,
    beta2,
    lr,
    minLr,
    hasMinLr,
    gaussianMean,
    gaussianStd,
    gaussianMean2,
    gaussianStd2,
    batchSize,
    gradientAccumulation,
    saveEvery,
    sampleEvery,
    minSnrGamma,
    debug,
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
          {train && (
            <div className="space-y-2 flex-1 min-w-[140px]">
              <p className="text-xs text-gray-400">LR</p>
              <input
                type="number"
                min={1e-6}
                step="any"
                placeholder="e.g. 1e-4"
                value={lr ?? ''}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? null : parseFloat(v);
                  setValue(num != null && Number.isFinite(num) ? num : undefined, 'config.process[0].train.lr');
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
            <p className="text-xs text-gray-400">Min SNR gamma</p>
            <input
              type="number"
              min={0}
              max={100}
              step="any"
              placeholder="e.g. 2 or 5"
              value={minSnrGamma}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setValue(v, 'config.process[0].train.min_snr_gamma');
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-end gap-4 flex-wrap">
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
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Beta1</p>
            <input
              type="number"
              min={0}
              max={0.999}
              step="0.01"
              placeholder="e.g. 0.9"
              value={beta1 ?? ''}
              onChange={(e) => {
                const v = e.target.value.trim();
                const num = v === '' ? null : parseFloat(v);
                setValue(num != null && Number.isFinite(num) ? num : null, `${OPTIMIZER_PARAMS_PATH}.beta1`);
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Beta2</p>
            <input
              type="number"
              min={0.001}
              max={0.9999}
              step="0.001"
              placeholder="e.g. 0.99"
              value={beta2}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setValue(v, `${OPTIMIZER_PARAMS_PATH}.beta2`);
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-end gap-4 flex-wrap">
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
            <p className="text-xs text-gray-400">Save every / Sample every (steps)</p>
            <div className="flex items-center gap-2 flex-wrap">
              <input
                type="number"
                min={1}
                max={10000}
                step={1}
                placeholder="e.g. 250"
                value={saveEvery}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? 250 : parseInt(v, 10);
                  if (Number.isInteger(num) && num >= 1) {
                    setValue(num, 'config.process[0].save.save_every');
                    setApplyStatus('idle');
                  }
                }}
                className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none min-w-[120px]"
              />
              <input
                type="number"
                min={1}
                max={10000}
                step={1}
                placeholder="e.g. 250"
                value={sampleEvery}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  const num = v === '' ? 250 : parseInt(v, 10);
                  if (Number.isInteger(num) && num >= 1) {
                    setValue(num, 'config.process[0].sample.sample_every');
                    setApplyStatus('idle');
                  }
                }}
                className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none min-w-[120px]"
              />
            </div>
          </div>
        </div>

        <div className="flex items-end gap-4 flex-wrap">
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Gaussian mean</p>
            <input
              type="number"
              min={0}
              max={999}
              step="any"
              placeholder="e.g. 500"
              value={gaussianMean}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_mean`);
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Gaussian std</p>
            <input
              type="number"
              min={1e-6}
              step="any"
              placeholder="e.g. 0.2"
              value={gaussianStd}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_std`);
                setApplyStatus('idle');
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        {showGaussianPeak2 ? (
          <div className="flex items-end gap-4 flex-wrap">
            <div className="space-y-2 flex-1 min-w-[140px]">
              <p className="text-xs text-gray-400">Gaussian mean 2</p>
              <input
                type="number"
                min={0}
                max={999}
                step="any"
                placeholder="e.g. 750"
                value={gaussianMean2}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_mean_2`);
                  setApplyStatus('idle');
                }}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div className="space-y-2 flex-1 min-w-[140px]">
              <p className="text-xs text-gray-400">Gaussian std 2</p>
              <input
                type="number"
                min={1e-6}
                step="any"
                placeholder="e.g. 0.2"
                value={gaussianStd2}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v)) setValue(v, `${TRAIN_PATH}.gaussian_std_2`);
                  setApplyStatus('idle');
                }}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        ) : null}

        <div className="flex items-end gap-4 flex-wrap">
          <div className="space-y-2 flex-1 min-w-[140px]">
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
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div className="space-y-2 flex-1 min-w-[140px]">
            <p className="text-xs text-gray-400">Gradient accumulation</p>
            <input
              type="number"
              min={1}
              max={64}
              step={1}
              placeholder="e.g. 1"
              value={gradientAccumulation}
              onChange={(e) => {
                const v = e.target.value.trim();
                const num = v === '' ? 1 : parseInt(v, 10);
                if (Number.isInteger(num) && num >= 1) {
                  setValue(num, 'config.process[0].train.gradient_accumulation');
                  setApplyStatus('idle');
                }
              }}
              className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          <div className="space-y-2 flex items-center gap-2">
            <p className="text-xs text-gray-400">Debug (logging)</p>
            <button
              type="button"
              role="switch"
              aria-checked={debug}
              onClick={() => {
                setValue(!debug, 'config.process[0].logging.debug');
                setApplyStatus('idle');
              }}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 ${debug ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition-transform ${debug ? 'translate-x-5' : 'translate-x-1'}`}
              />
            </button>
          </div>
        </div>

        {Array.isArray(datasets) && datasets.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Datasets — Network weight</p>
            <div className="flex items-center gap-2 flex-wrap">
              {datasets.map((d: { folder_path?: string; network_weight?: number }, i: number) => {
                const name = folderNameFromPath(d.folder_path) || `Dataset ${i + 1}`;
                return (
                <div key={i} className="space-y-1 min-w-[100px]">
                  <label className="block text-xs text-gray-500">Dataset {name}</label>
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
              ); })}
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
