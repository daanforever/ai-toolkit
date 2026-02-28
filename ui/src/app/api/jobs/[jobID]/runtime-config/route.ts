import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> }
) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let body: { max_lr?: number; min_lr?: number; gaussian_mean?: number; gaussian_std?: number; weight_decay?: number; content_or_style?: string; timestep_type?: string; network_weights?: number[]; batch_size?: number };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400 }
    );
  }

  const CONTENT_OR_STYLE_VALUES = ['balanced', 'content', 'style', 'gaussian', 'fixed_cycle'] as const;
  const TIMESTEP_TYPE_VALUES = ['sigmoid', 'linear', 'shift', 'weighted', 'gaussian'] as const;

  const data: { runtime_max_lr?: number; runtime_min_lr?: number; runtime_gaussian_mean?: number; runtime_gaussian_std?: number; runtime_weight_decay?: number; runtime_content_or_style?: string; runtime_timestep_type?: string; runtime_network_weights?: string; runtime_batch_size?: number } = {};

  const maxLr = body.max_lr;
  if (maxLr !== undefined) {
    if (typeof maxLr !== 'number' || !Number.isFinite(maxLr) || maxLr <= 0) {
      return NextResponse.json(
        { error: 'max_lr must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_max_lr = maxLr;
  }

  const minLr = body.min_lr;
  if (minLr !== undefined) {
    if (typeof minLr !== 'number' || !Number.isFinite(minLr) || minLr <= 0) {
      return NextResponse.json(
        { error: 'min_lr must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_min_lr = minLr;
  }

  const gaussianMean = body.gaussian_mean;
  if (gaussianMean !== undefined) {
    if (typeof gaussianMean !== 'number' || !Number.isFinite(gaussianMean) || gaussianMean < 0 || gaussianMean > 999) {
      return NextResponse.json(
        { error: 'gaussian_mean must be a number in [0, 999]' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_mean = gaussianMean;
  }

  const gaussianStd = body.gaussian_std;
  if (gaussianStd !== undefined) {
    if (typeof gaussianStd !== 'number' || !Number.isFinite(gaussianStd) || gaussianStd <= 0) {
      return NextResponse.json(
        { error: 'gaussian_std must be a positive number' },
        { status: 400 }
      );
    }
    data.runtime_gaussian_std = gaussianStd;
  }

  const weightDecay = body.weight_decay;
  if (weightDecay !== undefined) {
    if (typeof weightDecay !== 'number' || !Number.isFinite(weightDecay) || weightDecay < 0) {
      return NextResponse.json(
        { error: 'weight_decay must be a non-negative number' },
        { status: 400 }
      );
    }
    data.runtime_weight_decay = weightDecay;
  }

  const contentOrStyle = body.content_or_style;
  if (contentOrStyle !== undefined) {
    if (typeof contentOrStyle !== 'string' || !CONTENT_OR_STYLE_VALUES.includes(contentOrStyle)) {
      return NextResponse.json(
        { error: 'content_or_style must be one of: balanced, content, style, gaussian, fixed_cycle' },
        { status: 400 }
      );
    }
    data.runtime_content_or_style = contentOrStyle;
  }

  const timestepType = body.timestep_type;
  if (timestepType !== undefined) {
    if (typeof timestepType !== 'string' || !TIMESTEP_TYPE_VALUES.includes(timestepType)) {
      return NextResponse.json(
        { error: 'timestep_type must be one of: sigmoid, linear, shift, weighted, gaussian' },
        { status: 400 }
      );
    }
    data.runtime_timestep_type = timestepType;
  }

  const networkWeights = body.network_weights;
  if (networkWeights !== undefined) {
    if (!Array.isArray(networkWeights) || networkWeights.length === 0) {
      return NextResponse.json(
        { error: 'network_weights must be a non-empty array' },
        { status: 400 }
      );
    }
    for (let i = 0; i < networkWeights.length; i++) {
      const w = networkWeights[i];
      if (typeof w !== 'number' || !Number.isFinite(w) || w <= 0) {
        return NextResponse.json(
          { error: `network_weights[${i}] must be a positive number` },
          { status: 400 }
        );
      }
    }
    data.runtime_network_weights = JSON.stringify(networkWeights);
  }

  const batchSize = body.batch_size;
  if (batchSize !== undefined) {
    if (typeof batchSize !== 'number' || !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 128) {
      return NextResponse.json(
        { error: 'batch_size must be an integer between 1 and 128' },
        { status: 400 }
      );
    }
    data.runtime_batch_size = batchSize;
  }

  if (Object.keys(data).length === 0) {
    return NextResponse.json(
      { error: 'At least one of max_lr, min_lr, gaussian_mean, gaussian_std, weight_decay, content_or_style, timestep_type, network_weights, batch_size must be provided' },
      { status: 400 }
    );
  }

  const updated = await prisma.job.update({
    where: { id: jobID },
    data,
  });

  return NextResponse.json(updated);
}
