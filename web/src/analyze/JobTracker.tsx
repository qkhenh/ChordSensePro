import type { ErrorCode, JobProgress, JobStage } from '../api/types'
import { useI18n } from '../i18n/context'

const STAGES: JobStage[] = ['fetching', 'normalizing', 'separating', 'beat_tracking', 'recognizing']
type StepState = 'done' | 'active' | 'failed' | 'pending'

interface Props {
  job: JobProgress
  /** Show the "fetching" step: only analyses of a link have one. */
  withFetch?: boolean
  /** Only offered when the source is still known and the failure is worth retrying. */
  onRetry?: () => void
  onDismiss: () => void
  /** A retry is on its way: do not offer to send it again. */
  busy?: boolean
}

function stepState(job: JobProgress, steps: JobStage[], index: number): StepState {
  if (job.state === 'done') return 'done'
  const current = job.stage ? steps.indexOf(job.stage) : -1
  if (job.state === 'queued' || current < 0 || index > current) return 'pending'
  if (index < current) return 'done'
  return job.state === 'failed' ? 'failed' : 'active'
}

const DOT: Record<StepState, string> = {
  done: 'bg-emerald-400',
  active: 'bg-brand-purple-light animate-pulse',
  failed: 'bg-red-400',
  pending: 'bg-white/20',
}

/** The stepper of an analysis: where it is, what went wrong, what the user can do next. */
export function JobTracker({ job, withFetch = false, onRetry, onDismiss, busy = false }: Props) {
  const { t } = useI18n()
  const steps = STAGES.filter((stage) => withFetch || stage !== 'fetching')
  const final = job.state === 'done' || job.state === 'failed'
  const headline =
    job.state === 'queued' ? t.analyze.tracker.queued : job.state === 'done' ? t.analyze.tracker.done : t.analyze.tracker.running
  const errors = t.analyze.errors as Record<string, string>
  const reason = job.error ? (errors[job.error.code as ErrorCode] ?? errors.unknown) : null

  return (
    <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
      {job.state !== 'failed' && <p className="mb-3 font-medium text-white">{headline}</p>}

      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(job.progress * 100)}
        className="mb-5 h-2 overflow-hidden rounded-full bg-white/10"
      >
        <div
          className={`h-full transition-all ${job.state === 'failed' ? 'bg-red-400' : 'bg-brand-purple'}`}
          style={{ width: `${Math.round(job.progress * 100)}%` }}
        />
      </div>

      <ol aria-label={t.analyze.tracker.steps} className="space-y-2">
        {steps.map((stage, index) => {
          const state = stepState(job, steps, index)
          return (
            <li
              key={stage}
              data-state={state}
              aria-current={state === 'active' ? 'step' : undefined}
              className={`flex items-center gap-3 text-sm ${state === 'pending' ? 'text-slate-500' : 'text-slate-200'}`}
            >
              <span aria-hidden="true" className={`h-2.5 w-2.5 shrink-0 rounded-full ${DOT[state]}`} />
              {t.analyze.stages[stage]}
            </li>
          )
        })}
      </ol>

      <div className="mt-5">
        {reason && (
          <p role="alert" className="mb-3 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {reason}
          </p>
        )}
        <div className="flex flex-wrap gap-3">
          {job.state === 'failed' && job.error?.retryable && onRetry && (
            <button
              type="button"
              onClick={onRetry}
              disabled={busy}
              className="rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white hover:bg-brand-purple-light disabled:opacity-60"
            >
              {busy ? t.analyze.sending : t.analyze.tracker.retry}
            </button>
          )}
          <button
            type="button"
            onClick={onDismiss}
            className="rounded-lg border border-white/20 px-4 py-2 text-sm text-white hover:bg-white/10"
          >
            {final ? t.analyze.tracker.close : t.analyze.tracker.stop}
          </button>
        </div>
      </div>
    </section>
  )
}
