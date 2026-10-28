import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { pickLatestSafetensors } from '@/server/jobHeatmap';
import { resolveScaleLrSettings, runScaleLrCurve } from '@/server/jobScaleLr';
import { TOOLKIT_ROOT, resolveVenvPython } from '@/paths';

export const runtime = 'nodejs';
export const maxDuration = 60;

const prisma = new PrismaClient();

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
    include: { runtimeParams: true },
  });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const settings = resolveScaleLrSettings(job);
  if (!settings.ok) {
    return NextResponse.json({ available: false, reason: settings.reason });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const filePath = pickLatestSafetensors(jobFolder);
  if (!filePath) {
    return NextResponse.json({ available: false });
  }

  try {
    const payload = await runScaleLrCurve(
      resolveVenvPython(TOOLKIT_ROOT),
      TOOLKIT_ROOT,
      filePath,
      settings.mean,
      settings.std,
      settings.mask,
    );
    return NextResponse.json({
      ...payload,
      available: true,
      mean: settings.mean,
      std: settings.std,
      mask: settings.mask,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
