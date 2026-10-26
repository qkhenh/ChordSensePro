import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { analyzeUrl, uploadFile } from '../api/analyze'
import { useApiClient } from '../api/context'
import { clearActiveJob, readActiveJob, writeActiveJob } from '../analyze/activeJob'
import { analyzeErrorText } from '../analyze/errors'
import { JobTracker } from '../analyze/JobTracker'
import { SourceSelector, type Source } from '../analyze/SourceSelector'
import { useJob } from '../analyze/useJob'
import { useI18n } from '../i18n/context'

export default function WorkspacePage() {
  const client = useApiClient()
  const { t } = useI18n()
  const [jobId, setJobId] = useState<string | null>(readActiveJob)
  const [source, setSource] = useState<Source | null>(null)
  // A remembered job that the server no longer knows (expired, other account) is dropped.
  const { job } = useJob(jobId, { onMissing: () => forget() })

  const start = useMutation({
    mutationFn: (next: Source) =>
      next.kind === 'file' ? uploadFile(client, next.file) : analyzeUrl(client, next.url),
    onSuccess: ({ job_id }, next) => {
      writeActiveJob(job_id)
      setSource(next)
      setJobId(job_id)
    },
  })

  function forget() {
    clearActiveJob()
    setJobId(null)
    setSource(null)
    start.reset()
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="font-display text-2xl font-bold text-white">{t.nav.workspace}</h2>
      <p className="mt-2 mb-6 text-slate-400">{t.pages.workspace}</p>

      {jobId === null && (
        <SourceSelector
          busy={start.isPending}
          serverError={start.isError ? analyzeErrorText(t, start.error) : null}
          onSubmit={(next) => start.mutate(next)}
        />
      )}
      {jobId !== null && job === null && (
        <p role="status" className="text-slate-400">
          {t.analyze.tracker.loading}
        </p>
      )}
      {jobId !== null && job !== null && (
        <JobTracker
          job={job}
          withFetch={source?.kind === 'url' || job.stage === 'fetching'}
          onRetry={source ? () => start.mutate(source) : undefined}
          onDismiss={forget}
        />
      )}
    </div>
  )
}
