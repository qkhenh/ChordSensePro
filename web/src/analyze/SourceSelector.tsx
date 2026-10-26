import { useRef, useState, type KeyboardEvent } from 'react'
import { useI18n } from '../i18n/context'
import { Dropzone } from './Dropzone'
import { LinkForm } from './LinkForm'

export type Source = { kind: 'file'; file: File } | { kind: 'url'; url: string }
type TabId = 'upload' | 'record' | 'link'

const TABS: { id: TabId; enabled: boolean }[] = [
  { id: 'upload', enabled: true },
  { id: 'record', enabled: false }, // shown so people know it is coming (UC1 step 1)
  { id: 'link', enabled: true },
]

interface Props {
  onSubmit(source: Source): void
  busy?: boolean
  serverError?: string | null
}

/** Step 1 of UC1: choose where the music comes from. */
export function SourceSelector({ onSubmit, busy = false, serverError = null }: Props) {
  const { t } = useI18n()
  const [tab, setTab] = useState<TabId>('upload')
  const buttons = useRef(new Map<TabId, HTMLButtonElement>())
  const enabled = TABS.filter((entry) => entry.enabled).map((entry) => entry.id)

  function select(id: TabId) {
    setTab(id)
    buttons.current.get(id)?.focus()
  }

  function onKeyDown(event: KeyboardEvent) {
    const here = enabled.indexOf(tab)
    const move: Record<string, number | undefined> = {
      ArrowRight: (here + 1) % enabled.length,
      ArrowLeft: (here - 1 + enabled.length) % enabled.length,
      Home: 0,
      End: enabled.length - 1,
    }
    const next = move[event.key]
    if (next === undefined) return
    event.preventDefault()
    select(enabled[next])
  }

  return (
    <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
      <div role="tablist" aria-label={t.analyze.tabs.label} onKeyDown={onKeyDown} className="mb-5 flex gap-2">
        {TABS.map(({ id, enabled: usable }) => (
          <button
            key={id}
            ref={(el) => {
              if (el) buttons.current.set(id, el)
            }}
            type="button"
            role="tab"
            id={`source-tab-${id}`}
            aria-selected={tab === id}
            aria-controls={`source-panel-${id}`}
            aria-describedby={usable ? undefined : 'source-record-soon'}
            tabIndex={tab === id ? 0 : -1}
            disabled={!usable}
            onClick={() => setTab(id)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === id ? 'bg-brand-purple text-white' : 'text-slate-300 hover:bg-white/10'
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {t.analyze.tabs[id]}
          </button>
        ))}
      </div>
      <span id="source-record-soon" className="sr-only">
        {t.analyze.recordSoon}
      </span>

      <div role="tabpanel" id={`source-panel-${tab}`} aria-labelledby={`source-tab-${tab}`}>
        {tab === 'upload' && <Dropzone busy={busy} onSubmit={(file) => onSubmit({ kind: 'file', file })} />}
        {tab === 'link' && (
          <LinkForm
            busy={busy}
            onSubmit={(url) => onSubmit({ kind: 'url', url })}
            onUseUpload={() => select('upload')}
          />
        )}
      </div>

      {serverError && (
        <p role="alert" className="mt-4 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {serverError}
        </p>
      )}
    </section>
  )
}
