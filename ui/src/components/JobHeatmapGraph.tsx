'use client';

import { Job } from '@prisma/client';
import { useMemo, useState } from 'react';
import useJobHeatmap from '@/hooks/useJobHeatmap';

interface Props {
  job: Job;
}

type MetricHelp = {
  paragraphs: string[];
  actions: string[];
};

const SHARED_HELP = [
  'This is a snapshot of one saved LoRA file, not a chart over training steps. Each column is a model layer: early layers on the left, later layers on the right. Each row is one part of that layer. The name comes from the file.',
  'When the rows are q, k, and v, they are where the layer looks: what to search for, what to compare, and what to take. out is how that look is written back. w1, w2, and w3 are what the layer does with what it saw.',
  'Color compares cells only with each other on this map. The brightest cell is the largest value here. The darkest is the smallest. If the whole map is one color, the values are close together: read the numbers on the scale to see whether that shared value is small or large. Brightness is size, not a grade. An empty cell means one of the two adapter halves is still zero, usually at the very start of training, so there is nothing to measure.',
];

const METRICS: { id: string; label: string; help: MetricHelp }[] = [
  {
    id: 'frobenius',
    label: '||ΔW||_F',
    help: {
      paragraphs: [
        'LoRA does not rewrite the whole model. It adds a small correction. This number is how loud that correction is in this piece of the layer. It does not compare the correction with the original weights, so the same number in two layers is not the same kind of change.',
        'Think of two recordings of the same sentence. The words match, but one is quiet and the other is loud. A brighter cell is where the correction is louder than the other cells on this map.',
      ],
      actions: [
        'A few cells much larger than the rest: those parts took most of the change. If your samples look right, leave them. If the change shows up in the wrong place, train fewer steps or lower the learning rate.',
        'The numbers on the scale stay tiny late in training: the correction is still quiet everywhere. Train longer or raise the learning rate.',
        'The numbers on the scale are large and the samples look like copies of the training images: stop earlier or lower the learning rate. The colors alone cannot say this. They only compare cells with each other.',
      ],
    },
  },
  {
    id: 'balance',
    label: '||B|| / ||A||',
    help: {
      paragraphs: [
        'Each LoRA has two halves. The first starts full of random numbers and chooses which features to look at. The second starts empty and, during training, writes what to do with those features. This number is how large the second half has grown compared with the first.',
        'At the start the second half is silent, so the number is tiny or the cell is empty. That is normal for the first steps. On a long run the second half often ends up larger than the first. Compare cells with each other. There is no single correct value.',
      ],
      actions: [
        'Tiny numbers across the whole map late in training: the second half is still quiet. Train longer, or raise the learning rate of the second half (the up weights) above the first.',
        "A few cells much larger than the rest: those pieces' second half grew far ahead of the first. If the samples look blown out or unstable, lower the learning rate. A second half that is simply larger than the first, across the map, is a normal end to a long run.",
        "Empty cells at the start: keep training. Empty cells late in training: that piece's second half is still zero, so it did not learn.",
      ],
    },
  },
  {
    id: 'stable_rank',
    label: 'stable rank',
    help: {
      paragraphs: [
        'The LoRA rank is a ceiling: how many different tricks the adapter is allowed to learn. This number says how many of them carry real strength. Near 1, almost all the strength sits in one direction. Near the rank, the strength is spread across the directions the rank allows.',
        'Picture a row of microphones, one for each rank, and one person talking. The microphones are all there, but only one voice is working. A simple subject can honestly sit near 1. Most trained LoRAs use only a fraction of the rank they were given. This is one snapshot, so it does not show whether the number fell during training.',
      ],
      actions: [
        'Low values (near 1): decrease the rank. The extra directions are idle. Do this to make a smaller adapter, not because a low number means the training failed.',
        'High values (near the rank): the adapter is using the room it was given. Increase the rank only if the samples still miss detail you want. A high number is not automatically better. LoRA often works because the change stays simple, and a higher rank does not always learn more.',
      ],
    },
  },
  {
    id: 'effective_rank',
    label: 'effective rank',
    help: {
      paragraphs: [
        'This also counts how many different directions are in the correction, including the quiet ones. Stable rank asks which direction holds the strength. Effective rank asks how many directions are present at all, so it is usually higher.',
        'Picture a soup with one strong flavor and several quieter ones. Near 1 means one direction. Near the rank means many directions are present, including quiet ones.',
      ],
      actions: [
        'Low values (near 1): decrease the rank. One direction does not need the extra room.',
        'High values (near the rank) while stable rank is also high: the rank budget is full. Increase the rank only if the samples still miss detail.',
        'High values while stable rank stays near 1: quiet directions exist, but one direction holds almost all the strength. Decrease the rank. Those quiet directions are not a reason to add more.',
      ],
    },
  },
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
  else if (hasGrid) statusText = `${cols.length} × ${rows.length}`;

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
                style={{ gridTemplateColumns: `max-content repeat(${rows.length}, 1.5rem)` }}
              >
                <div className="sticky left-0 z-10 bg-gray-950" />
                {rows.map(layer => (
                  <div key={layer} className="flex items-end justify-center" title={layer}>
                    <span className="whitespace-nowrap text-[10px] leading-none text-gray-400 [writing-mode:vertical-rl] rotate-180">
                      {layer}
                    </span>
                  </div>
                ))}
                {cols.map((typeName, typeIndex) => (
                  <TypeRow
                    key={typeName}
                    typeName={typeName}
                    typeIndex={typeIndex}
                    layers={rows}
                    matrix={matrix}
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
        <HeatmapHelp metricId={metric} />
      </div>
    </div>
  );
}

function HeatmapHelp({ metricId }: { metricId: string }) {
  const selected = METRICS.find(item => item.id === metricId) ?? METRICS[0];
  return (
    <div className="mt-3 bg-gray-950 border border-gray-800 rounded-lg p-3 text-sm leading-relaxed text-gray-400">
      <h3 className="mb-2 text-xs font-medium text-gray-200">What this map shows</h3>
      <div className="space-y-2">
        {SHARED_HELP.map(text => (
          <p key={text}>{text}</p>
        ))}
      </div>
      <h3 className="mb-2 mt-4 text-xs font-medium text-gray-200">{selected.label}</h3>
      <div className="space-y-2">
        {selected.help.paragraphs.map(text => (
          <p key={text}>{text}</p>
        ))}
      </div>
      <p className="mt-2">What to do:</p>
      <ul className="mt-1 list-disc space-y-1 pl-5">
        {selected.help.actions.map(text => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </div>
  );
}

function TypeRow({
  typeName,
  typeIndex,
  layers,
  matrix,
  range,
  onTip,
}: {
  typeName: string;
  typeIndex: number;
  layers: string[];
  matrix: (number | null)[][];
  range: { min: number; max: number } | null;
  onTip: (tip: { text: string; x: number; y: number } | null) => void;
}) {
  return (
    <>
      <div className="sticky left-0 z-10 flex items-center whitespace-nowrap bg-gray-950 pr-2 text-[11px] text-gray-400">
        {typeName}
      </div>
      {layers.map((layer, layerIndex) => {
        const value = matrix[layerIndex]?.[typeIndex] ?? null;
        const t = value == null || range == null ? 0 : range.max === range.min ? 1 : (value - range.min) / (range.max - range.min);
        const background = value == null ? 'rgba(255,255,255,0.04)' : heatColor(t);
        const label = value == null ? `${layer} · ${typeName}: n/a` : `${layer} · ${typeName}: ${formatNum(value)}`;
        return (
          <div
            key={layer}
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
