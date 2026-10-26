import { useId, useState, type FormEvent } from 'react'
import { useI18n } from '../i18n/context'
import { validateUrl, type UrlProblem } from './validate'

interface Props {
  onSubmit(url: string): void
  onUseUpload(): void
  busy: boolean
}

/** Paste a link. A bad link is explained and the way out (AF1) is to upload the file instead. */
export function LinkForm({ onSubmit, onUseUpload, busy }: Props) {
  const { t } = useI18n()
  const inputId = useId()
  const [url, setUrl] = useState('')
  const [problem, setProblem] = useState<UrlProblem | null>(null)

  function submit(event: FormEvent) {
    event.preventDefault()
    const found = validateUrl(url)
    setProblem(found)
    if (!found) onSubmit(url.trim())
  }

  return (
    <form onSubmit={submit} noValidate>
      <label htmlFor={inputId} className="mb-1 block text-sm font-medium text-slate-300">
        {t.analyze.linkLabel}
      </label>
      <div className="flex flex-wrap gap-3">
        <input
          id={inputId}
          type="text"
          inputMode="url"
          autoComplete="off"
          placeholder="https://…"
          value={url}
          aria-invalid={problem ? true : undefined}
          onChange={(e) => {
            setUrl(e.target.value)
            setProblem(null)
          }}
          className="min-w-0 flex-1 rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-white outline-none focus:border-brand-purple-light"
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white transition hover:bg-brand-purple-light disabled:opacity-60"
        >
          {busy ? t.analyze.sending : t.analyze.submit}
        </button>
      </div>
      {problem && (
        <div role="alert" className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
          <p>{t.analyze.problems[problem]}</p>
          <button type="button" onClick={onUseUpload} className="mt-1 font-semibold underline">
            {t.analyze.uploadInstead}
          </button>
        </div>
      )}
    </form>
  )
}
