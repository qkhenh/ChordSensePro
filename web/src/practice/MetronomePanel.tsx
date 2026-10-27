import { useState } from 'react'
import { useI18n } from '../i18n/context'
import type { Timbre } from '../metronome/click'
import type { MetronomeMode, MetronomeState } from './useMetronome'

const MODES: readonly MetronomeMode[] = ['manual', 'song']
const TIMBRES: readonly Timbre[] = ['woodblock', 'beep']

/** A numeric text box that only accepts a value when it parses; anything else snaps back. */
function NumberField({
  label,
  value,
  onCommit,
  narrow,
}: {
  label: string
  value: number
  onCommit: (value: number) => void
  narrow?: boolean
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const commit = () => {
    if (draft === null) return
    const parsed = Number.parseFloat(draft.replace(',', '.'))
    if (Number.isFinite(parsed)) onCommit(parsed)
    setDraft(null)
  }
  return (
    <label className="flex items-center gap-2 text-sm text-slate-300">
      {label}
      <input
        type="text"
        inputMode="decimal"
        aria-label={label}
        value={draft ?? String(value)}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => event.key === 'Enter' && commit()}
        className={`rounded-md border border-white/10 bg-white/5 px-2 py-1 text-white ${narrow ? 'w-16' : 'w-20'}`}
      />
    </label>
  )
}

export function MetronomePanel({ metronome: m }: { metronome: MetronomeState }) {
  const { t } = useI18n()
  return (
    <section aria-label={t.practice.metronome.title} className="space-y-3 rounded-xl border border-white/10 p-4">
      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 font-semibold text-white">
          <input
            type="checkbox"
            role="switch"
            aria-label={t.practice.metronome.title}
            checked={m.enabled}
            onChange={(event) => void m.setEnabled(event.target.checked)}
          />
          {t.practice.metronome.title}
        </label>
        <div role="radiogroup" aria-label={t.practice.metronome.mode} className="flex gap-3 text-sm text-slate-300">
          {MODES.map((mode) => (
            <label key={mode} className="flex items-center gap-1">
              <input
                type="radio"
                name="metronome-mode"
                checked={m.mode === mode}
                onChange={() => m.setMode(mode)}
              />
              {t.practice.metronome[mode]}
            </label>
          ))}
        </div>
      </div>

      {m.mode === 'manual' && (
        <div className="flex flex-wrap items-center gap-4">
          <NumberField label="BPM" value={m.bpm} onCommit={m.setBpm} />
          <button
            type="button"
            onClick={m.tapTempo}
            className="rounded-md border border-white/10 px-3 py-1 text-sm text-slate-200 hover:bg-white/10"
          >
            {t.practice.metronome.tap}
          </button>
          <NumberField label={t.practice.metronome.beatsPerBar} value={m.beatsPerBar} onCommit={m.setBeatsPerBar} narrow />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4 text-sm text-slate-300">
        <label className="flex items-center gap-2">
          {t.practice.metronome.timbre}
          <select
            value={m.timbre}
            onChange={(event) => m.setTimbre(event.target.value as Timbre)}
            className="rounded-md border border-white/10 bg-[#0b0b1a] px-2 py-1 text-white"
          >
            {TIMBRES.map((timbre) => (
              <option key={timbre} value={timbre}>
                {t.practice.metronome[timbre]}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          {t.practice.metronome.volume}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={m.volume}
            onChange={(event) => m.setVolume(Number(event.target.value))}
          />
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={m.countInOn} onChange={(event) => m.setCountInOn(event.target.checked)} />
          {t.practice.metronome.countIn}
        </label>
      </div>
    </section>
  )
}
