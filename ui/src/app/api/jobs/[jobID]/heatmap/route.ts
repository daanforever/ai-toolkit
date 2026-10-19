import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { pickLatestSafetensors, runLoraHeatmap } from '@/server/jobHeatmap';
import { TOOLKIT_ROOT, resolveVenvPython } from '@/paths';

export const runtime = 'nodejs';
export const maxDuration = 60;

const prisma = new PrismaClient();

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const filePath = pickLatestSafetensors(jobFolder);
  if (!filePath) {
    return NextResponse.json({ available: false });
  }

  try {
    const payload = await runLoraHeatmap(resolveVenvPython(TOOLKIT_ROOT), TOOLKIT_ROOT, filePath);
    return NextResponse.json({ ...payload, available: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
