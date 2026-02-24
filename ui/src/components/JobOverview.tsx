import { Job } from '@prisma/client';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import GPUWidget from '@/components/GPUWidget';
import CPUWidget from '@/components/CPUWidget';
import FilesWidget from '@/components/FilesWidget';
import { getTotalSteps } from '@/utils/jobs';
import { Cpu, HardDrive, Info, Gauge } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import useJobLog from '@/hooks/useJobLog';

interface JobOverviewProps {
  job: Job;
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => job.gpu_ids.split(',').map(id => parseInt(id)), [job.gpu_ids]);
  const { log, setLog, status: statusLog, refresh: refreshLog } = useJobLog(job.id, 2000);
  const logRef = useRef<HTMLDivElement>(null);
  // Track whether we should auto-scroll to bottom
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);
  const [newMaxLr, setNewMaxLr] = useState('');
  const [newGaussianMean, setNewGaussianMean] = useState('');
  const [newGaussianStd, setNewGaussianStd] = useState('');
  const [newWeightDecay, setNewWeightDecay] = useState('');
  const [newContentOrStyle, setNewContentOrStyle] = useState('balanced');
  const [newTimestepType, setNewTimestepType] = useState('sigmoid');
  const [runtimeConfigStatus, setRuntimeConfigStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

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
  ];

  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const { cpuInfo, isCPUInfoLoaded } = useCPUInfo(5000);
  const totalSteps = getTotalSteps(job);
  const progress = (job.step / totalSteps) * 100;
  const isStopping = job.stop && job.status === 'running';

  const logLines: string[] = useMemo(() => {
    // split at line breaks on \n or \r\n but not \r
    let splits: string[] = log.split(/\n|\r\n/);

    splits = splits.map(line => {
      return line.split(/\r/).pop();
    }) as string[];

    // only return last 100 lines max
    const maxLines = 1000;
    if (splits.length > maxLines) {
      splits = splits.slice(splits.length - maxLines);
    }

    return splits;
  }, [log]);

  // Handle scroll events to determine if user has scrolled away from bottom
  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      // Consider "at bottom" if within 10 pixels of the bottom
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
      setIsScrolledToBottom(isAtBottom);
    }
  };

  // Auto-scroll to bottom only if we were already at the bottom
  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom]);

  const handleApplyRuntimeMaxLr = async () => {
    const value = parseFloat(newMaxLr);
    if (!Number.isFinite(value) || value <= 0) {
      setRuntimeConfigStatus('error');
      return;
    }
    setRuntimeConfigStatus('loading');
    try {
      const res = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_lr: value }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || res.statusText);
      }
      setRuntimeConfigStatus('success');
    } catch (e) {
      setRuntimeConfigStatus('error');
    }
  };

  const handleApplyRuntimeGaussian = async () => {
    const meanVal = newGaussianMean.trim() === '' ? null : parseFloat(newGaussianMean);
    const stdVal = newGaussianStd.trim() === '' ? null : parseFloat(newGaussianStd);
    if (meanVal !== null && (!Number.isFinite(meanVal) || meanVal < 0 || meanVal > 1)) {
      setRuntimeConfigStatus('error');
      return;
    }
    if (stdVal !== null && (!Number.isFinite(stdVal) || stdVal <= 0)) {
      setRuntimeConfigStatus('error');
      return;
    }
    if (meanVal === null && stdVal === null) {
      setRuntimeConfigStatus('error');
      return;
    }
    const body: { gaussian_mean?: number; gaussian_std?: number } = {};
    if (meanVal !== null) body.gaussian_mean = meanVal;
    if (stdVal !== null) body.gaussian_std = stdVal;
    setRuntimeConfigStatus('loading');
    try {
      const res = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || res.statusText);
      }
      setRuntimeConfigStatus('success');
    } catch (e) {
      setRuntimeConfigStatus('error');
    }
  };

  const handleApplyRuntimeWeightDecay = async () => {
    const value = parseFloat(newWeightDecay);
    if (!Number.isFinite(value) || value < 0) {
      setRuntimeConfigStatus('error');
      return;
    }
    setRuntimeConfigStatus('loading');
    try {
      const res = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weight_decay: value }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || res.statusText);
      }
      setRuntimeConfigStatus('success');
    } catch (e) {
      setRuntimeConfigStatus('error');
    }
  };

  const handleApplyRuntimeContentOrStyle = async () => {
    if (!newContentOrStyle) {
      setRuntimeConfigStatus('error');
      return;
    }
    setRuntimeConfigStatus('loading');
    try {
      const res = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content_or_style: newContentOrStyle }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || res.statusText);
      }
      setRuntimeConfigStatus('success');
    } catch (e) {
      setRuntimeConfigStatus('error');
    }
  };

  const handleApplyRuntimeTimestepType = async () => {
    if (!newTimestepType) {
      setRuntimeConfigStatus('error');
      return;
    }
    setRuntimeConfigStatus('loading');
    try {
      const res = await fetch(`/api/jobs/${job.id}/runtime-config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timestep_type: newTimestepType }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || res.statusText);
      }
      setRuntimeConfigStatus('success');
    } catch (e) {
      setRuntimeConfigStatus('error');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'bg-emerald-500/10 text-emerald-500';
      case 'stopping':
        return 'bg-amber-500/10 text-amber-500';
      case 'stopped':
        return 'bg-gray-500/10 text-gray-400';
      case 'completed':
        return 'bg-blue-500/10 text-blue-500';
      case 'error':
        return 'bg-rose-500/10 text-rose-500';
      default:
        return 'bg-gray-500/10 text-gray-400';
    }
  };

  let status = job.status;
  if (isStopping) {
    status = 'stopping';
  }

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      {/* Job Information Panel */}
      <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <h2 className="text-gray-100">
            <Info className="w-5 h-5 mr-2 -mt-1 text-amber-400 inline-block" /> {job.info}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(job.status)}`}>{job.status}</span>
        </div>

        <div className="p-4 space-y-6 flex flex-col flex-grow">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-gray-200">
                Step {job.step} of {totalSteps}
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div className="h-2 rounded-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
            </div>
          </div>

          {/* Job Info Grid */}
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            <div className="flex items-center space-x-4">
              <HardDrive className="w-5 h-5 text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Job Name</p>
                <p className="text-sm font-medium text-gray-200">{job.name}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Cpu className="w-5 h-5 text-purple-400" />
              <div>
                <p className="text-xs text-gray-400">Assigned GPUs</p>
                <p className="text-sm font-medium text-gray-200">GPUs: {job.gpu_ids}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Gauge className="w-5 h-5 text-green-400" />
              <div>
                <p className="text-xs text-gray-400">Speed</p>
                <p className="text-sm font-medium text-gray-200">{job.speed_string == '' ? '?' : job.speed_string}</p>
              </div>
            </div>
          </div>

          {/* Runtime max LR and weight decay — one row */}
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px] space-y-2">
              <p className="text-xs text-gray-400">New max LR</p>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="1e-6"
                  step="any"
                  placeholder="e.g. 1e-4"
                  value={newMaxLr}
                  onChange={(e) => {
                    setNewMaxLr(e.target.value);
                    setRuntimeConfigStatus('idle');
                  }}
                  className="flex-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleApplyRuntimeMaxLr}
                  disabled={runtimeConfigStatus === 'loading' || !newMaxLr.trim()}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {runtimeConfigStatus === 'loading' ? '…' : 'Apply'}
                </button>
              </div>
              {runtimeConfigStatus === 'success' && (
                <p className="text-xs text-green-500">Applied.</p>
              )}
              {runtimeConfigStatus === 'error' && (
                <p className="text-xs text-rose-500">Failed to apply.</p>
              )}
            </div>
            <div className="flex-1 min-w-[200px] space-y-2">
              <p className="text-xs text-gray-400">New weight decay</p>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="0"
                  step="any"
                  placeholder="e.g. 0.01 or 0"
                  value={newWeightDecay}
                  onChange={(e) => {
                    setNewWeightDecay(e.target.value);
                    setRuntimeConfigStatus('idle');
                  }}
                  className="flex-1 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleApplyRuntimeWeightDecay}
                  disabled={runtimeConfigStatus === 'loading' || !newWeightDecay.trim()}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {runtimeConfigStatus === 'loading' ? '…' : 'Apply'}
                </button>
              </div>
            </div>
          </div>

          {/* Runtime Timestep Type and Runtime Timestep Bias — row 1 */}
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Runtime Timestep Type / Timestep Bias</p>
            <div className="flex items-center gap-2 flex-wrap">
              <select
                value={newTimestepType}
                onChange={(e) => {
                  setNewTimestepType(e.target.value);
                  setRuntimeConfigStatus('idle');
                }}
                className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-w-[120px]"
                title="Timestep Type"
              >
                {TIMESTEP_TYPE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleApplyRuntimeTimestepType}
                disabled={runtimeConfigStatus === 'loading'}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {runtimeConfigStatus === 'loading' ? '…' : 'Apply'}
              </button>
              <span className="text-gray-500 mx-1">|</span>
              <select
                value={newContentOrStyle}
                onChange={(e) => {
                  setNewContentOrStyle(e.target.value);
                  setRuntimeConfigStatus('idle');
                }}
                className="rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 focus:border-blue-500 focus:outline-none min-w-[140px]"
                title="Timestep Bias"
              >
                {CONTENT_OR_STYLE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleApplyRuntimeContentOrStyle}
                disabled={runtimeConfigStatus === 'loading'}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {runtimeConfigStatus === 'loading' ? '…' : 'Apply'}
              </button>
            </div>
            {(runtimeConfigStatus === 'success' || runtimeConfigStatus === 'error') && (
              <p className={`text-xs ${runtimeConfigStatus === 'success' ? 'text-green-500' : 'text-rose-500'}`}>
                {runtimeConfigStatus === 'success' ? 'Applied.' : 'Failed to apply.'}
              </p>
            )}
          </div>

          {/* Runtime Gaussian (mean / std) — row 2 */}
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Runtime Gaussian (mean / std)</p>
            <div className="flex items-center gap-2 flex-wrap">
              <input
                type="number"
                min="0"
                max="1"
                step="any"
                placeholder="mean (0–1)"
                value={newGaussianMean}
                onChange={(e) => {
                  setNewGaussianMean(e.target.value);
                  setRuntimeConfigStatus('idle');
                }}
                className="w-24 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
              <input
                type="number"
                min="1e-6"
                step="any"
                placeholder="std (&gt;0)"
                value={newGaussianStd}
                onChange={(e) => {
                  setNewGaussianStd(e.target.value);
                  setRuntimeConfigStatus('idle');
                }}
                className="w-24 rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={handleApplyRuntimeGaussian}
                disabled={runtimeConfigStatus === 'loading' || (!newGaussianMean.trim() && !newGaussianStd.trim())}
                className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {runtimeConfigStatus === 'loading' ? '…' : 'Apply'}
              </button>
            </div>
            {(runtimeConfigStatus === 'success' || runtimeConfigStatus === 'error') && (
              <p className={`text-xs ${runtimeConfigStatus === 'success' ? 'text-green-500' : 'text-rose-500'}`}>
                {runtimeConfigStatus === 'success' ? 'Applied.' : 'Failed to apply.'}
              </p>
            )}
          </div>

          {/* Log - Now using flex-grow to fill remaining space */}
          <div className="bg-gray-950 rounded-lg p-4 relative flex-grow min-h-60">
            <div
              ref={logRef}
              className="text-xs text-gray-300 absolute inset-0 p-4 overflow-y-auto"
              onScroll={handleScroll}
            >
              {statusLog === 'loading' && 'Loading log...'}
              {statusLog === 'error' && 'Error loading log'}
              {['success', 'refreshing'].includes(statusLog) && (
                <div>
                  {logLines.map((line, index) => {
                    return <pre key={index}>{line}</pre>;
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* GPU Widget Panel */}
      <div className="col-span-1">
        <div>{isCPUInfoLoaded && cpuInfo && <CPUWidget cpu={cpuInfo} />}</div>
        <div className="mt-4">{isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}</div>
        <div className="mt-4">
          <FilesWidget jobID={job.id} />
        </div>
      </div>
    </div>
  );
}
