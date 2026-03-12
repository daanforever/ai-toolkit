import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, job.name);

  if (fs.existsSync(trainingFolder)) {
    const entries = fs.readdirSync(trainingFolder);
    for (const entry of entries) {
      if (entry === '.job_config.json' || entry === 'config.yaml') {
        continue;
      }
      const entryPath = path.join(trainingFolder, entry);
      try {
        const stat = fs.lstatSync(entryPath);
        if (stat.isDirectory()) {
          fs.rmSync(entryPath, { recursive: true, force: true });
        } else {
          fs.unlinkSync(entryPath);
        }
      } catch (error) {
        console.error(`Error removing entry during reset: ${entryPath}`, error);
      }
    }
  }

  return NextResponse.json({ success: true });
}

