import type { ChordMode } from '../format/chord'
import { useI18n } from '../i18n/context'

const MODES: readonly ChordMode[] = ['basic', 'precise']

export function ChordModeToggle({ mode, onChange }: { mode: ChordMode; onChange: (mode: ChordMode) => void }) {
  const { t } = useI18n()
  return (
    <div role="group" aria-label={t.practice.chordMode} className="flex items-center gap-1">
      {MODES.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={m === mode}
          onClick={() => onChange(m)}
          className="rounded-md border border-white/10 px-2 py-1 text-sm text-slate-200 hover:bg-white/10 aria-pressed:border-brand-purple-light aria-pressed:bg-brand-purple/40"
        >
          {t.practice[m]}
        </button>
      ))}
    </div>
  )
}
