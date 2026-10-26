import type { ApiError } from '../api/http'
import { useI18n } from '../i18n/context'

/** EF2: the daily allowance is spent — say so, say when it comes back, offer the upgrade. */
export function QuotaBanner({ error }: { error: ApiError }) {
  const { t, lang } = useI18n()
  const when = error.resetsAt?.toLocaleString(lang, { dateStyle: 'short', timeStyle: 'short' })
  return (
    <div role="alert" className="mb-5 rounded-2xl border border-amber-400/40 bg-amber-500/10 p-5 text-amber-100">
      <p className="font-semibold">{t.analyze.quotaBanner.title(error.limit)}</p>
      {when && <p className="mt-1 text-sm">{t.analyze.quotaBanner.resets(when)}</p>}
      <a
        href="#upgrade"
        className="mt-3 inline-block rounded-lg bg-amber-400 px-4 py-2 text-sm font-semibold text-black hover:bg-amber-300"
      >
        {t.analyze.quotaBanner.upgrade}
      </a>
    </div>
  )
}
