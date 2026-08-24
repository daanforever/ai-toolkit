'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, RotateCcw, Eraser, ChevronsUp, ChevronsDown } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import { startJob, stopJob, deleteJob, getAvaliableJobActions, markJobAsStopped, resetJob, clearSamplesAndLossLog, scaleJobLr } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  showLrScale?: boolean;
}

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
  showLrScale = false,
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue, canReset, canClear } = getAvaliableJobActions(job);
  const [lrScaleBusy, setLrScaleBusy] = useState(false);

  if (!afterDelete) afterDelete = onRefresh;

  const handleScaleLr = async (factor: number) => {
    if (lrScaleBusy || !canStop) return;
    setLrScaleBusy(true);
    try {
      await scaleJobLr(job, factor);
      if (onRefresh) onRefresh();
    } catch (e) {
      console.error('Error scaling job LR:', e);
    } finally {
      setLrScaleBusy(false);
    }
  };

  return (
    <div className={`${className}`}>
      {canStart && (
        <Button
          onClick={async () => {
            if (!canStart) return;
            await startJob(job.id);
            // start the queue as well
            if (autoStartQueue) {
              await startQueue(job.gpu_ids);
            }
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          onClick={async () => {
            if (!canRemoveFromQueue) return;
            await markJobAsStopped(job.id);
            if (onRefresh) onRefresh();
          }}
          className={`ml-2 opacity-100`}
        >
          <X />
        </Button>
      )}
      {canStop && (
        <Button
          onClick={() => {
            if (!canStop) return;
            openConfirm({
              title: 'Stop Job',
              message: `Are you sure you want to stop the job "${job.name}"? You CAN resume later.`,
              type: 'info',
              confirmText: 'Stop',
              onConfirm: async () => {
                await stopJob(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <Pause />
        </Button>
      )}
      {showLrScale && canStop && (
        <>
          <Button
            onClick={() => handleScaleLr(2)}
            disabled={lrScaleBusy}
            title="LR ×2"
            className={`ml-2 opacity-100`}
          >
            <ChevronsUp />
          </Button>
          <Button
            onClick={() => handleScaleLr(0.5)}
            disabled={lrScaleBusy}
            title="LR ÷2"
            className={`ml-2 opacity-100`}
          >
            <ChevronsDown />
          </Button>
        </>
      )}
      {canClear && (
        <Button
          onClick={() => {
            if (!canClear) return;
            openConfirm({
              title: 'Clear samples & loss log',
              message: `Are you sure you want to clear the samples folder and loss_log.* files for job "${job.name}"? This will permanently delete the samples directory and all loss_log files in the job output directory. The job will continue running.`,
              type: 'warning',
              confirmText: 'Clear',
              onConfirm: async () => {
                await clearSamplesAndLossLog(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <Eraser />
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye />
        </Link>
      )}
      {canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-2 hover:text-gray-100 inline-block">
          <Pen />
        </Link>
      )}
      {canReset && (
        <Button
          onClick={() => {
            openConfirm({
              title: 'Reset Job Output',
              message: `Are you sure you want to reset the output for job "${job.name}"? This will permanently delete all files and folders in the job output directory except for .job_config.json and config.yaml.`,
              type: 'warning',
              confirmText: 'Reset',
              onConfirm: async () => {
                await resetJob(job.id);
                if (onRefresh) onRefresh();
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <RotateCcw />
        </Button>
      )}
      <Button
        onClick={() => {
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: async () => {
              if (job.status === 'running') {
                try {
                  await stopJob(job.id);
                } catch (e) {
                  console.error('Error stopping job before deleting:', e);
                }
              }
              await deleteJob(job.id);
              if (afterDelete) afterDelete();
            },
          });
        }}
        className={`ml-2 opacity-100`}
      >
        <Trash2 />
      </Button>
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-2'}>
          <Cog />
        </MenuButton>
        <MenuItems anchor="bottom" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-4">
          <MenuItem>
            <Link href={`/jobs/new?cloneId=${job.id}`} className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block">
              Clone Job
            </Link>
          </MenuItem>
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded"
              onClick={() => {
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: async () => {
                    await markJobAsStopped(job.id);
                    onRefresh && onRefresh();
                  },
                });
              }}
            >
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
