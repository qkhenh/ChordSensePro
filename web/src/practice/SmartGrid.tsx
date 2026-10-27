import { memo, useCallback, useEffect, useRef, type PointerEvent, type Ref } from 'react'
import { formatChord, type ChordMode } from '../format/chord'
import type { Bar, Grid } from '../grid/grid'
import { useI18n } from '../i18n/context'

interface CellProps {
  bar: Bar
  mode: ChordMode
  inLoop: boolean
  label: (n: number, chords: string) => string
  onActivate: (index: number, shift: boolean) => void
}

/**
 * One bar. The "currently playing" look is applied from outside through `data-current` (see
 * usePlayhead), so a cell never re-renders as the song plays: it only changes when its own
 * chords, the chord mode, the language or its membership of the loop do.
 */
const BarCell = memo(function BarCell({ bar, mode, inLoop, label, onActivate }: CellProps) {
  const names = bar.chords.map((c) => formatChord(c.event, mode))
  return (
    <button
      type="button"
      data-bar={bar.index}
      data-loop={inLoop ? 'true' : undefined}
      aria-label={label(bar.index + 1, names.join(', '))}
      onClick={(event) => onActivate(bar.index, event.shiftKey)}
      className="relative min-h-16 rounded-lg border border-white/10 bg-white/5 px-3 pt-5 pb-2 text-left text-slate-100 transition-colors duration-75 hover:bg-white/10 data-[loop=true]:border-brand-cyan/60 data-[current=true]:border-brand-purple-light data-[current=true]:bg-brand-purple/40"
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
  /** Two bar indexes in the order the person picked them (a drag, or two Shift+clicks). */
  onSelect?: (fromBar: number, toBar: number) => void
  /** Bars of the active loop, both ends included. */
  loop?: { from: number; to: number } | null
  /** Handed to `usePlayhead`, which lights the bar being played. */
  rootRef: Ref<HTMLDivElement>
}

const barOf = (target: EventTarget): number | null => {
  const cell = target instanceof Element ? target.closest<HTMLElement>('[data-bar]') : null
  return cell ? Number(cell.dataset.bar) : null
}

export function SmartGrid({ grid, mode, onSeek, onSelect, loop = null, rootRef }: Props) {
  const { t } = useI18n()
  // Cells get one stable callback, so a new `onSeek` identity never re-renders all of them.
  const latest = useRef({ grid, onSeek, onSelect })
  useEffect(() => {
    latest.current = { grid, onSeek, onSelect }
  })
  const anchor = useRef<number | null>(null)
  const dragStart = useRef<number | null>(null)
  const dragged = useRef(false)

  const activate = useCallback((index: number, shift: boolean) => {
    const { grid: g, onSeek: seek, onSelect: select } = latest.current
    if (dragged.current) {
      dragged.current = false // the click that ends a drag is not a seek
      return
    }
    if (shift) {
      if (anchor.current === null) anchor.current = index
      else {
        select?.(anchor.current, index)
        anchor.current = null
      }
      return
    }
    anchor.current = null
    seek(g.barStartMs(index))
  }, [])

  const onPointerDown = (event: PointerEvent) => {
    dragged.current = false
    dragStart.current = event.shiftKey ? null : barOf(event.target)
  }
  const onPointerUp = (event: PointerEvent) => {
    const from = dragStart.current
    dragStart.current = null
    const to = barOf(event.target)
    if (from === null || to === null || from === to) return
    dragged.current = true
    latest.current.onSelect?.(from, to)
  }

  return (
    <div
      ref={rootRef}
      role="group"
      aria-label={t.practice.gridLabel}
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      className="grid grid-cols-2 gap-2 sm:grid-cols-4"
    >
      {grid.bars.map((bar) => (
        <BarCell
          key={bar.index}
          bar={bar}
          mode={mode}
          inLoop={loop !== null && bar.index >= loop.from && bar.index <= loop.to}
          label={t.practice.bar}
          onActivate={activate}
        />
      ))}
    </div>
  )
}
