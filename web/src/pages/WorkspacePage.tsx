import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { analyzeUrl, uploadFile } from '../api/analyze'
import { useApiClient } from '../api/context'
import { ApiError } from '../api/http'
import { clearActiveJob, readActiveJob, writeActiveJob } from '../analyze/activeJob'
import { analyzeErrorText } from '../analyze/errors'
import { JobTracker } from '../analyze/JobTracker'
import { QuotaBanner } from '../analyze/QuotaBanner'
import { SourceSelector, type Source } from '../analyze/SourceSelector'
import { useJob } from '../analyze/useJob'
import { useAuth } from '../auth/context'
import { useI18n } from '../i18n/context'
import { quotaKey } from '../shell/quota'

export default function WorkspacePage() {
  const client = useApiClient()
  const { t } = useI18n()
  const { user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState<string | null>(readActiveJob)
  const [source, setSource] = useState<Source | null>(null)
  // A remembered job that the server no longer knows (expired, other account) is dropped.
  const { job } = useJob(jobId, { onMissing: () => forget() })

  const refreshQuota = () => queryClient.invalidateQueries({ queryKey: quotaKey(user?.id) })

  const start = useMutation({
    mutationFn: (next: Source) =>
      next.kind === 'file' ? uploadFile(client, next.file) : analyzeUrl(client, next.url),
    onSuccess: ({ job_id }, next) => {
      writeActiveJob(job_id)
      setSource(next)
      setJobId(job_id)
    },
    // Accepted jobs spend a lookup, a refused one (429) must show the badge as it really is.
    onSettled: refreshQuota,
  })

  function forget() {
    clearActiveJob()
    setJobId(null)
    setSource(null)
    start.reset()
  }

  // When a job ends its quota may have changed (a failure gives it back); a finished one
  // opens its chart.
  const jobState = job?.state
  const chartId = job?.result_chart_id ?? null
  useEffect(() => {
    if (jobState !== 'done' && jobState !== 'failed') return
    void queryClient.invalidateQueries({ queryKey: quotaKey(user?.id) })
    if (jobState === 'done' && chartId) {
      clearActiveJob()
      navigate(`/charts/${chartId}`)
    }
  }, [jobState, chartId, navigate, queryClient, user?.id])

  const refusal = start.error instanceof ApiError && start.error.code === 'quota_exceeded' ? start.error : null

  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="font-display text-2xl font-bold text-white">{t.nav.workspace}</h2>
      <p className="mt-2 mb-6 text-slate-400">{t.pages.workspace}</p>

      {refusal && <QuotaBanner error={refusal} />}
      {jobId === null && (
        <SourceSelector
          busy={start.isPending}
          serverError={start.isError && !refusal ? analyzeErrorText(t, start.error) : null}
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
