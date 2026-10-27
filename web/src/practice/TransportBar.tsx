import { useEffect, useRef } from 'react'
import { minutesAndSeconds } from '../format/duration'
import { useI18n } from '../i18n/context'
import { RATES, type Rate, type Transport } from '../transport/types'

const CLOCK_MS = 250 // 4 Hz is plenty for a read-out

interface Props {
  transport: Transport | null
  playing: boolean
  canPlay: boolean
  durationMs: number
  rate: Rate
  onToggle: () => void
  onRate: (rate: Rate) => void
  /** Free plan: 0.5× and 0.75× are marked as Pro (choosing one is up to the page to refuse). */
  speedLocked?: boolean
}

/** `m:ss / m:ss`, written straight into the element: the clock never re-renders React. */
function useClock(transport: Transport | null, durationMs: number) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const total = minutesAndSeconds(durationMs)
    const write = () => {
      const text = `${minutesAndSeconds(transport?.currentTimeMs() ?? 0)} / ${total}`
      if (element.textContent !== text) element.textContent = text
    }
    write()
    const timer = setInterval(write, CLOCK_MS)
    return () => clearInterval(timer)
  }, [transport, durationMs])
  return ref
}

export function TransportBar({ transport, playing, canPlay, durationMs, rate, onToggle, onRate, speedLocked = false }: Props) {
  const { t } = useI18n()
  const clock = useClock(transport, durationMs)
  const canChangeRate = transport?.capabilities.rate ?? false

  return (
    <div className="flex flex-wrap items-center gap-4">
      <button
        type="button"
        disabled={!canPlay}
        onClick={onToggle}
        className="rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white hover:opacity-90 disabled:opacity-40"
      >
        {playing ? t.practice.pause : t.practice.play}
      </button>
      <span ref={clock} data-testid="clock" className="font-mono text-slate-300 tabular-nums" />
      <div role="group" aria-label={t.practice.speed} className="flex items-center gap-1">
        {RATES.map((r) => (
          <button
            key={r}
            type="button"
            disabled={!canChangeRate}
            aria-pressed={r === rate}
            onClick={() => onRate(r)}
            className="rounded-md border border-white/10 px-2 py-1 text-sm text-slate-200 hover:bg-white/10 disabled:opacity-40 aria-pressed:border-brand-purple-light aria-pressed:bg-brand-purple/40"
          >
            {r}×
            {speedLocked && r !== 1 && (
              <span aria-hidden="true" className="ml-1 text-xs text-amber-200">
                {t.practice.upgrade.badge}
              </span>
            )}
          </button>
        ))}
      </div>
      {transport && !canChangeRate && <p className="text-sm text-slate-500">{t.practice.rateUnsupported}</p>}
    </div>
  )
}
