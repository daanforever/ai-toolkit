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

  let body: { max_lr?: number; gaussian_mean?: number; gaussian_std?: number };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400 }
    );
  }

  const data: { runtime_max_lr?: number; runtime_gaussian_mean?: number; runtime_gaussian_std?: number } = {};

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

  const gaussianMean = body.gaussian_mean;
  if (gaussianMean !== undefined) {
    if (typeof gaussianMean !== 'number' || !Number.isFinite(gaussianMean) || gaussianMean < 0 || gaussianMean > 1) {
      return NextResponse.json(
        { error: 'gaussian_mean must be a number in [0, 1]' },
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

  if (Object.keys(data).length === 0) {
    return NextResponse.json(
      { error: 'At least one of max_lr, gaussian_mean, gaussian_std must be provided' },
      { status: 400 }
    );
  }

  const updated = await prisma.job.update({
    where: { id: jobID },
    data,
  });

  return NextResponse.json(updated);
}
