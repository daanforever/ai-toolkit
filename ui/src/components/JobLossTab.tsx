'use client';

import { Job } from '@prisma/client';
import JobLossGraph from '@/components/JobLossGraph';
import JobScaleLrGraph from '@/components/JobScaleLrGraph';

export default function JobLossTab({ job }: { job: Job; onRefresh?: () => void }) {
  return (
    <div className="flex flex-col gap-4 pb-6">
      <JobLossGraph job={job} />
      <JobScaleLrGraph job={job} />
    </div>
  );
}
