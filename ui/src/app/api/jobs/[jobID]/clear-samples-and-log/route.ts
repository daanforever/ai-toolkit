import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';
import sqlite3 from 'sqlite3';

const prisma = new PrismaClient();

function runSql(db: sqlite3.Database, sql: string) {
  return new Promise<void>((resolve, reject) => {
    db.exec(sql, (err) => (err ? reject(err) : resolve()));
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

async function purgeLossLog(logPath: string) {
  const db = new sqlite3.Database(logPath);
  db.configure('busyTimeout', 30_000);
  try {
    await runSql(db, 'BEGIN IMMEDIATE;');
    try {
      await runSql(db, 'DELETE FROM metrics; DELETE FROM metric_keys; DELETE FROM steps;');
      await runSql(db, 'COMMIT;');
    } catch (error) {
      await runSql(db, 'ROLLBACK;');
      throw error;
    }
  } finally {
    await closeDb(db);
  }
}

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

  // Empty the samples directory, keep the directory itself
  const samplesPath = path.join(trainingFolder, 'samples');
  if (fs.existsSync(samplesPath) && fs.lstatSync(samplesPath).isDirectory()) {
    for (const entry of fs.readdirSync(samplesPath)) {
      fs.rmSync(path.join(samplesPath, entry), { recursive: true, force: true });
    }
  }

  // Purge loss log rows, keep loss_log.db file, tables and indexes
  const lossLogPath = path.join(trainingFolder, 'loss_log.db');
  if (fs.existsSync(lossLogPath)) {
    await purgeLossLog(lossLogPath);
  }

  // Truncate log.txt, the trainer opens it in append mode so it keeps writing at the new end
  const logTxtPath = path.join(trainingFolder, 'log.txt');
  if (fs.existsSync(logTxtPath)) {
    fs.truncateSync(logTxtPath, 0);
  }

  return NextResponse.json({ success: true });
}
