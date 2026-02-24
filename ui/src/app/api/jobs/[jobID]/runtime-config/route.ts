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

  let body: { max_lr?: number };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON body' },
      { status: 400 }
    );
  }

  const maxLr = body.max_lr;
  if (typeof maxLr !== 'number' || !Number.isFinite(maxLr) || maxLr <= 0) {
    return NextResponse.json(
      { error: 'max_lr must be a positive number' },
      { status: 400 }
    );
  }

  const updated = await prisma.job.update({
    where: { id: jobID },
    data: { runtime_max_lr: maxLr },
  });

  return NextResponse.json(updated);
}
