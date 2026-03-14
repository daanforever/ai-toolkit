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

  if (!fs.existsSync(trainingFolder)) {
    return NextResponse.json({ success: true });
  }

  // Remove samples directory
  const samplesPath = path.join(trainingFolder, 'samples');
  if (fs.existsSync(samplesPath)) {
    try {
      const stat = fs.lstatSync(samplesPath);
      if (stat.isDirectory()) {
        fs.rmSync(samplesPath, { recursive: true, force: true });
      }
    } catch (error) {
      console.error(`Error removing samples during clear: ${samplesPath}`, error);
    }
  }

  // Remove all loss_log.* files
  const entries = fs.readdirSync(trainingFolder);
  for (const entry of entries) {
    if (entry.startsWith('loss_log.')) {
      const entryPath = path.join(trainingFolder, entry);
      try {
        const stat = fs.lstatSync(entryPath);
        if (stat.isFile()) {
          fs.unlinkSync(entryPath);
        }
      } catch (error) {
        console.error(`Error removing loss_log file during clear: ${entryPath}`, error);
      }
    }
  }

  return NextResponse.json({ success: true });
}
