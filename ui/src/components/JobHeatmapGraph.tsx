'use client';

import { Job } from '@prisma/client';
import { useMemo, useState } from 'react';
import useJobHeatmap from '@/hooks/useJobHeatmap';

interface Props {
  job: Job;
}

const METRICS: { id: string; label: string }[] = [
  { id: 'frobenius', label: '||ΔW||_F' },
  { id: 'balance', label: '||B|| / ||A||' },
  { id: 'stable_rank', label: 'stable rank' },
  { id: 'effective_rank', label: 'effective rank' },
];

function formatNum(value: number) {
  if (!Number.isFinite(value)) return '';
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(3);
  if (Math.abs(value) >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}

function heatColor(t: number) {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(20 + x * 220);
  const g = Math.round(50 + x * 170);
  const b = Math.round(120 - x * 70);
  return `rgb(${r}, ${g}, ${b})`;
}

function formatStamp(mtimeMs: number) {
  const date = new Date(mtimeMs);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString();
}

export default function JobHeatmapGraph({ job }: Props) {
  const { data, status, error, refresh } = useJobHeatmap(job.id);
  const [metric, setMetric] = useState(METRICS[0].id);
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);

  const rows = data?.rows ?? [];
  const cols = data?.cols ?? [];
  const matrix = data?.metrics?.[metric] ?? null;

  const range = useMemo(() => {
    if (!matrix) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const row of matrix) {
      for (const value of row) {
        if (value == null || !Number.isFinite(value)) continue;
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    return { min, max };
  }, [matrix]);

  const hasGrid = status === 'success' && data?.available === true && rows.length > 0 && cols.length > 0 && matrix != null;
  const file = data?.file;

  let statusText = 'Press Refresh';
  if (status === 'loading') statusText = 'Loading...';
  else if (status === 'error') statusText = 'Error';
  else if (status === 'success' && data?.available === false) statusText = 'No saved LoRA yet';
  else if (hasGrid) statusText = `${rows.length} × ${cols.length}`;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="h-2 w-2 rounded-full bg-amber-400" />
          <h2 className="text-gray-100 text-sm font-medium">Heatmap graph</h2>
          <span className="text-xs text-gray-400 truncate">{statusText}</span>
        </div>
        <button
          type="button"
          onClick={() => {
            setTip(null);
            void refresh();
          }}
          disabled={status === 'loading'}
          className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700 disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      <div className="px-4 pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 min-h-96 relative p-3">
          {status === 'idle' && (
            <div className="h-80 w-full flex items-center justify-center text-sm text-gray-400">
              Press Refresh to read the latest saved LoRA.
            </div>
          )}
          {status === 'loading' && (
            <div className="h-80 w-full flex items-center justify-center text-sm text-gray-400">Loading heatmap...</div>
          )}
          {status === 'error' && (
            <div className="h-80 w-full flex items-center justify-center text-sm text-red-300 px-6 text-center">
              {error || 'Failed to load heatmap.'}
            </div>
          )}
          {status === 'success' && data?.available === false && (
            <div className="h-80 w-full flex items-center justify-center text-sm text-gray-400">No saved LoRA yet.</div>
          )}
          {status === 'success' && data?.available === true && !hasGrid && (
            <div className="h-80 w-full flex items-center justify-center text-sm text-red-300">
              Heatmap response is missing {metric}.
            </div>
          )}
          {hasGrid && matrix && (
            <div className="overflow-auto max-h-[36rem]">
              <div
                className="inline-grid gap-px"
                style={{ gridTemplateColumns: `max-content repeat(${cols.length}, 1.5rem)` }}
              >
                <div />
                {cols.map(col => (
                  <div key={col} className="h-40 flex items-end justify-center" title={col}>
                    <span className="max-h-40 overflow-hidden text-[10px] leading-none text-gray-400 [writing-mode:vertical-rl] rotate-180">
                      {col}
                    </span>
                  </div>
                ))}
                {rows.map((row, rowIndex) => (
                  <Row
                    key={row}
                    row={row}
                    cols={cols}
                    values={matrix[rowIndex] ?? []}
                    range={range}
                    onTip={setTip}
                  />
                ))}
              </div>
            </div>
          )}
          {tip && (
            <div
              className="fixed z-50 pointer-events-none px-2 py-1 rounded-md text-xs bg-gray-900 text-gray-100 border border-gray-700 shadow-lg"
              style={{ left: tip.x, top: tip.y }}
            >
              {tip.text}
            </div>
          )}
        </div>
      </div>

      <div className="px-4 pb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Metric</label>
            <div className="flex flex-wrap gap-2">
              {METRICS.map(item => (
                <ToggleButton key={item.id} checked={metric === item.id} onClick={() => setMetric(item.id)} label={item.label} />
              ))}
            </div>
          </div>
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Scale</label>
            {range ? (
              <div>
                <div
                  className="h-3 rounded-sm border border-gray-800"
                  style={{ background: `linear-gradient(to right, ${heatColor(0)}, ${heatColor(1)})` }}
                />
                <div className="mt-1 flex justify-between text-[11px] text-gray-400">
                  <span>{formatNum(range.min)}</span>
                  <span>{formatNum(range.max)}</span>
                </div>
                <div className="mt-2 text-[11px] text-gray-500">Empty cells have no value (zero factor).</div>
              </div>
            ) : (
              <div className="text-sm text-gray-400">No values yet.</div>
            )}
            {file && (
              <div className="mt-3 text-[11px] text-gray-400 break-all">
                {file.filename}
                {file.step != null ? ` · step ${file.step}` : ''}
                {file.mtime_ms ? ` · ${formatStamp(file.mtime_ms)}` : ''}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({
  row,
  cols,
  values,
  range,
  onTip,
}: {
  row: string;
  cols: string[];
  values: (number | null)[];
  range: { min: number; max: number } | null;
  onTip: (tip: { text: string; x: number; y: number } | null) => void;
}) {
  return (
    <>
      <div className="text-[11px] text-gray-400 pr-2 flex items-center whitespace-nowrap">{row}</div>
      {cols.map((col, index) => {
        const value = values[index] ?? null;
        const t = value == null || range == null ? 0 : range.max === range.min ? 1 : (value - range.min) / (range.max - range.min);
        const background = value == null ? 'rgba(255,255,255,0.04)' : heatColor(t);
        const label = value == null ? `${row} · ${col}: n/a` : `${row} · ${col}: ${formatNum(value)}`;
        return (
          <div
            key={col}
            title={label}
            className="h-5 rounded-[2px]"
            style={{
              background,
              boxShadow: value == null ? 'inset 0 0 0 1px rgba(255,255,255,0.08)' : undefined,
            }}
            onMouseEnter={event => {
              const rect = event.currentTarget.getBoundingClientRect();
              const left = Math.min(rect.left, window.innerWidth - 240);
              onTip({ text: label, x: Math.max(8, left), y: rect.bottom + 6 });
            }}
            onMouseLeave={() => onTip(null)}
          />
        );
      })}
    </>
  );
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
