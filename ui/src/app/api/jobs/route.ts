import { NextResponse } from 'next/server';
import path from 'path';
import { Job, PrismaClient } from '@prisma/client';
import { getTrainingFolder } from '@/server/settings';
import { projectJobConfig, writeJobConfigYamlIfPresent } from '../../../../cron/loadJobConfig';

const prisma = new PrismaClient();

async function projectStoredJob(job: Job, trainingRoot: string): Promise<Job> {
  const projection = projectJobConfig(path.join(trainingRoot, job.name), job.job_config);
  if (projection.action === 'invalid') {
    console.error(`config.yaml for job ${job.name} was not applied: ${projection.message}`);
    return job;
  }
  if (projection.action === 'keep') {
    return job;
  }
  return prisma.job.update({
    where: { id: job.id },
    data: { job_config: projection.jobConfig },
  });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  try {
    const trainingRoot = await getTrainingFolder();
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      if (!job) {
        return NextResponse.json(job);
      }
      return NextResponse.json(await projectStoredJob(job, trainingRoot));
    }

    const jobs = await prisma.job.findMany({
      orderBy: { created_at: 'desc' },
    });
    const projected: Job[] = [];
    for (const job of jobs) {
      projected.push(await projectStoredJob(job, trainingRoot));
    }
    return NextResponse.json({ jobs: projected });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config, gpu_ids } = body;

    if (id) {
      const existing = await prisma.job.findUnique({
        where: { id },
      });
      let jobConfigToStore = JSON.stringify(job_config);
      if (existing) {
        const trainingRoot = await getTrainingFolder();
        const saved = writeJobConfigYamlIfPresent(path.join(trainingRoot, existing.name), job_config);
        jobConfigToStore = saved.jobConfig;
      }
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: jobConfigToStore,
        },
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
