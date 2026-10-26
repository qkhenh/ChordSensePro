import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { useApiClient } from '../api/context'
import { toastFor } from '../api/errors'
import type { Quota } from '../api/types'
import { useAuth } from '../auth/context'
import { useI18n } from '../i18n/context'
import { useToast } from './context'

const quotaKey = (userId: string | undefined) => ['quota', userId] as const

function useQuota() {
  const client = useApiClient()
  const { user } = useAuth()
  return useQuery({
    queryKey: quotaKey(user?.id),
    queryFn: () => client.request<Quota>('/users/me/quota'),
    enabled: !!user,
    staleTime: 30_000,
  })
}

/** Tier and remaining daily analyses (FR-18). */
export function QuotaBadge() {
  const { t, lang } = useI18n()
  const { show } = useToast()
  const { data, error, isPending } = useQuota()

  useEffect(() => {
    if (error) show(toastFor(error))
  }, [error, show])

  const base = 'inline-flex min-w-28 items-center rounded-full border px-3 py-1 text-sm font-medium'

  if (isPending) {
    return (
      <span data-testid="quota-badge" aria-busy="true" className={`${base} animate-pulse border-white/10 bg-white/5`}>
        <span className="h-4 w-20 rounded bg-white/10" />
      </span>
    )
  }
  if (!data) {
    return (
      <span data-testid="quota-badge" className={`${base} border-white/10 text-slate-400`}>
        {t.quota.unknown}
      </span>
    )
  }

  const exhausted = data.remaining === 0
  const detail =
    data.limit === null || data.remaining === null
      ? t.quota.unlimited
      : exhausted
        ? t.quota.exhausted
        : t.quota.left(data.remaining, data.limit)
  const resets = new Date(data.resets_at).toLocaleString(lang, { dateStyle: 'short', timeStyle: 'short' })

  return (
    <span
      data-testid="quota-badge"
      data-exhausted={exhausted}
      title={t.quota.resetsAt(resets)}
      className={`${base} ${exhausted ? 'border-amber-400/40 bg-amber-500/10 text-amber-200' : 'border-brand-purple/40 bg-brand-purple/10 text-brand-purple-light'}`}
    >
      {t.tier[data.tier]} · {detail}
    </span>
  )
}
