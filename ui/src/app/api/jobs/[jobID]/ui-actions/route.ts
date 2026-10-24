import fs from 'fs';
import path from 'path';
import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { TOOLKIT_ROOT } from '@/paths';

const prisma = new PrismaClient();

const TRAINER_TYPE = 'z_image_diffsynth_trainer';
const ACTIONS_PATH = path.join(
  TOOLKIT_ROOT,
  'extensions_built_in',
  'diffusion_models',
  'z_image_diffsynth',
  'util_ui',
  'actions.json',
);

interface UiAction {
  id: string;
  icon: string;
  title: string;
}

class InvalidJobConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidJobConfigError';
  }
}

function loadActions(): UiAction[] {
  const loaded: unknown = JSON.parse(fs.readFileSync(ACTIONS_PATH, 'utf8'));
  if (!Array.isArray(loaded)) {
    throw new Error('actions.json must be an array');
  }
  return loaded.map((entry, index) => {
    if (
      typeof entry !== 'object' ||
      entry === null ||
      typeof (entry as { id?: unknown }).id !== 'string' ||
      typeof (entry as { icon?: unknown }).icon !== 'string' ||
      typeof (entry as { title?: unknown }).title !== 'string'
    ) {
      throw new Error(`actions.json[${index}] must have string id, icon, and title`);
    }
    const action = entry as { id: string; icon: string; title: string };
    return { id: action.id, icon: action.icon, title: action.title };
  });
}

function readProcessType(jobConfig: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jobConfig);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new InvalidJobConfigError(message);
  }
  if (typeof parsed !== 'object' || parsed === null) {
    throw new InvalidJobConfigError('job_config must be an object');
  }
  const process = (parsed as { config?: { process?: { type?: unknown }[] } }).config?.process;
  const type = process?.[0]?.type;
  if (typeof type !== 'string' || type.length === 0) {
    throw new InvalidJobConfigError('job_config process type is missing');
  }
  return type;
}

function jobConfigError(): NextResponse {
  return NextResponse.json({ error: 'job_config is invalid' }, { status: 400 });
}

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> },
) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  let processType: string;
  try {
    processType = readProcessType(job.job_config);
  } catch (err) {
    if (err instanceof InvalidJobConfigError) {
      return jobConfigError();
    }
    throw err;
  }
  if (processType !== TRAINER_TYPE) {
    return NextResponse.json({ actions: [] });
  }
  return NextResponse.json({ actions: loadActions() });
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> },
) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch (err) {
    if (err instanceof SyntaxError) {
      return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
    }
    throw err;
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const action = (body as { action?: unknown }).action;
  const known = loadActions();
  if (typeof action !== 'string' || !known.some(entry => entry.id === action)) {
    return NextResponse.json({ error: 'unknown ui action' }, { status: 400 });
  }

  let processType: string;
  try {
    processType = readProcessType(job.job_config);
  } catch (err) {
    if (err instanceof InvalidJobConfigError) {
      return jobConfigError();
    }
    throw err;
  }
  if (processType !== TRAINER_TYPE) {
    return NextResponse.json({ error: 'ui actions require z_image_diffsynth_trainer' }, { status: 400 });
  }
  if (job.status !== 'running' || job.stop) {
    return NextResponse.json({ error: 'job is not running' }, { status: 409 });
  }

  const created = await prisma.runtimeCommand.create({
    data: { jobId: jobID, action },
  });
  return NextResponse.json(created);
}
