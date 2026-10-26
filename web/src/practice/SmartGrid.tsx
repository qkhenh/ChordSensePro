import { memo, type Ref } from 'react'
import { formatChord, type ChordMode } from '../format/chord'
import type { Bar, Grid } from '../grid/grid'
import { useI18n } from '../i18n/context'

interface CellProps {
  bar: Bar
  mode: ChordMode
  label: (n: number, chords: string) => string
  onSeek: (ms: number) => void
}

/**
 * One bar. The "currently playing" look is applied from outside through `data-current` (see
 * usePlayhead), so a cell never re-renders as the song plays: it only changes when its own
 * chords, the chord mode or the language do.
 */
const BarCell = memo(function BarCell({ bar, mode, label, onSeek }: CellProps) {
  const names = bar.chords.map((c) => formatChord(c.event, mode))
  return (
    <button
      type="button"
      data-bar={bar.index}
      aria-label={label(bar.index + 1, names.join(', '))}
      onClick={() => onSeek(bar.startMs)}
      className="relative min-h-16 rounded-lg border border-white/10 bg-white/5 px-3 pt-5 pb-2 text-left text-slate-100 transition-colors duration-75 hover:bg-white/10 data-[current=true]:border-brand-purple-light data-[current=true]:bg-brand-purple/40"
    >
      <span aria-hidden="true" className="absolute top-1 left-2 text-xs text-slate-500">
        {bar.index + 1}
      </span>
      <span aria-hidden="true" className="flex flex-wrap gap-x-3 gap-y-1 font-display text-lg font-semibold">
        {bar.chords.map((chord, i) => (
          <span key={chord.event.event_id} data-carried={chord.startsHere ? undefined : 'true'} className="data-[carried=true]:opacity-50">
            {names[i]}
          </span>
        ))}
      </span>
    </button>
  )
})

interface Props {
  grid: Grid
  mode: ChordMode
  onSeek: (ms: number) => void
  /** Handed to `usePlayhead`, which lights the bar being played. */
  rootRef: Ref<HTMLDivElement>
}

export function SmartGrid({ grid, mode, onSeek, rootRef }: Props) {
  const { t } = useI18n()
  return (
    <div ref={rootRef} role="group" aria-label={t.practice.gridLabel} className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {grid.bars.map((bar) => (
        <BarCell key={bar.index} bar={bar} mode={mode} label={t.practice.bar} onSeek={onSeek} />
      ))}
    </div>
  )
}
