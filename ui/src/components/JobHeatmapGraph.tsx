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
  'A snapshot of the newest saved LoRA file, not a chart over training steps. Each column is a model block (ordered by block name, then index). Each row is a module type inside the block, named after its keys in the file: q, k, v, and out are the attention query, key, value, and output projections; w1, w2, and w3 are the feed-forward projections.',
  'Each cell is the selected metric for one module. Color is linear from the smallest to the largest value on this map, so it compares cells only with each other. Brightness is relative size, not quality. If the whole map is one color, the values are close: read the scale to see whether they are small or large.',
  'An empty cell has no value: the module does not exist in that block, or, for the ratio and the ranks, its up or down matrix is all zeros (usually at the start of training).',
];

const METRICS: { id: string; label: string; help: MetricHelp }[] = [
  {
    id: 'frobenius',
    label: '||ΔW||_F',
    help: {
      paragraphs: [
        'The Frobenius norm of the weight update this LoRA adds to one module: ΔW = (alpha / rank) · up · down. It is the square root of the sum of the squares of all entries of ΔW, so it measures the total magnitude of the update. Zero means the module is unchanged.',
        'It is an absolute magnitude. It is not divided by the norm of the original weights, so the same value in two modules is not the same relative change. Matrices with more entries also tend to have larger norms. A brighter cell holds a larger update than a darker cell on this map.',
      ],
      actions: [
        'A few cells much larger than the rest: those modules took most of the update. If your samples look right, leave them. If the change shows up in the wrong place, train fewer steps or lower the learning rate.',
        'Values on the scale stay close to zero late in training: the update is still small everywhere. Train longer or raise the learning rate.',
        'Values on the scale are large and the samples look like copies of the training images: stop earlier or lower the learning rate. The colors alone cannot say this. They only compare cells with each other.',
      ],
    },
  },
  {
    id: 'balance',
    label: '||B|| / ||A||',
    help: {
      paragraphs: [
        'The ratio of the Frobenius norms of the two LoRA factors: ||up|| / ||down||, also written ||B|| / ||A||. The down matrix (A, rank × in) starts as random numbers. The up matrix (B, out × rank) starts at zero, so ΔW is zero and training starts from the original model. The ratio shows how far B has grown relative to A.',
        'It does not measure how strong the update is. Scaling A up and B down by the same factor leaves ΔW unchanged, so the ratio also reflects initialization and the learning rates of the two factors. It is near zero at the start and the cell is empty while B is still exactly zero. There is no single correct value: compare cells with each other.',
      ],
      actions: [
        'Values close to zero across the whole map late in training: B has barely moved from zero, so the update is still small. Train longer, or give the up weights a higher learning rate than the down weights (the idea behind LoRA+).',
        'A few cells far above the rest: B grew much more there relative to A. If the samples look over-saturated or unstable, lower the learning rate. A uniformly higher ratio across the whole map is not a problem by itself.',
        'Empty cells at the start: keep training. Empty cells late in training: up or down is all zeros in that module, so it learned nothing.',
      ],
    },
  },
  {
    id: 'stable_rank',
    label: 'stable rank',
    help: {
      paragraphs: [
        'The stable rank of ΔW: ||ΔW||_F² / σ_max², the sum of the squared singular values divided by the largest one squared. It is a smooth lower bound on the true rank. It lies between 1 and the LoRA rank and does not change if ΔW is rescaled. Near 1: almost all of the update lies along one direction, the one with the largest singular value. Near the LoRA rank: the update is spread evenly across all the directions the rank allows.',
        'Small singular values barely change it, so it shows how many directions carry most of the update, not how many exist. Compare it with the LoRA rank to see how much of the allocated rank is used. A simple subject can legitimately sit near 1, and most trained LoRAs use only a fraction of their rank. This is one snapshot, so it does not show how the value changed during training.',
      ],
      actions: [
        'Low values (near 1): decrease the rank. The extra directions carry almost none of the update. Do this to make a smaller adapter, not because a low value means the training failed.',
        'High values (near the rank): the adapter uses all the directions it was given. Increase the rank only if the samples still miss detail you want. A high value is not automatically better: LoRA often works because the update stays simple, and a higher rank does not always learn more.',
      ],
    },
  },
  {
    id: 'effective_rank',
    label: 'effective rank',
    help: {
      paragraphs: [
        'The effective rank of ΔW (Roy and Vetterli): exp(H), where H = −Σ p_i ln p_i is the entropy of the normalized singular values p_i = σ_i / Σσ_j. It is a smooth count of how many singular directions contribute. It lies between 1 and the LoRA rank and does not change if ΔW is rescaled. Equal singular values give the LoRA rank. One dominant singular value gives about 1.',
        'Stable rank compares the total squared magnitude with that of the top direction alone, so the top direction dominates it. Effective rank uses the whole spectrum, so weaker directions count too and it is usually higher. A large gap between the two means one strong direction plus several weaker ones.',
      ],
      actions: [
        'Low values (near 1): decrease the rank. One direction does not need the extra room.',
        'High values (near the rank) while stable rank is also high: the rank is fully used. Increase the rank only if the samples still miss detail.',
        'High values while stable rank stays near 1: weaker directions exist, but one direction holds almost all of the update. Decrease the rank. The weaker directions are not a reason to add more.',
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
