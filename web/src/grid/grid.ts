import type { AnalysisMetadata, ChordEvent } from '../api/types'

/** A chord, or the part of a chord, that falls inside one bar. */
export interface BarChord {
  event: ChordEvent
  fromMs: number
  toMs: number
  /** The chord begins in this bar (not carried over from the previous one). */
  startsHere: boolean
}

export interface Bar {
  index: number
  startMs: number
  endMs: number
  /** Where each beat of the bar falls; the first one is the downbeat. */
  beatsMs: number[]
  chords: BarChord[]
}

export interface Grid {
  bars: readonly Bar[]
  durationMs: number
  /** Start of the first bar; anything earlier is pre-roll and belongs to no bar. */
  firstBarMs: number
  /** Index of the bar containing `ms`; -1 before the first bar, the last bar from the end on. */
  barAtMs(ms: number): number
  barStartMs(index: number): number
  barEndMs(index: number): number
}

interface Span {
  startMs: number
  endMs: number
  /** Length a full bar would have here: beats are laid out on it even when the bar is cut short. */
  nominalMs: number
}

/** Bar spans, either interpolated from the tempo or taken from the explicit downbeats. */
function spansOf(meta: AnalysisMetadata): Span[] {
  const { beat_grid: grid, duration_ms: duration } = meta
  const barMs = (grid.beats_per_bar * 60_000) / grid.tempo_bpm
  const spans: Span[] = []

  const downbeats = (grid.downbeats_ms ?? []).filter((ms) => ms < duration)
  if (downbeats.length > 0) {
    let previous = barMs
    downbeats.forEach((start, i) => {
      const next = downbeats[i + 1]
      const nominal = next === undefined ? previous : next - start
      spans.push({ startMs: start, endMs: Math.min(next ?? duration, duration), nominalMs: nominal })
      previous = nominal
    })
    return spans
  }

  for (let k = 0; ; k++) {
    const startMs = Math.round(grid.first_beat_offset_ms + k * barMs)
    if (startMs >= duration) break
    const endMs = Math.min(Math.round(grid.first_beat_offset_ms + (k + 1) * barMs), duration)
    spans.push({ startMs, endMs, nominalMs: barMs })
  }
  return spans
}

export function buildGrid(meta: AnalysisMetadata, events: readonly ChordEvent[]): Grid {
  const perBar = meta.beat_grid.beats_per_bar
  const bars: Bar[] = spansOf(meta).map((span, index) => ({
    index,
    startMs: span.startMs,
    endMs: span.endMs,
    beatsMs: Array.from({ length: perBar }, (_, j) =>
      Math.round(span.startMs + (j * span.nominalMs) / perBar),
    ).filter((ms) => ms < span.endMs),
    chords: [],
  }))

  const firstBarMs = bars[0]?.startMs ?? 0
  const durationMs = meta.duration_ms

  /** First bar that ends after `ms` (the bar holding `ms`, or the next one when `ms` is in a gap). */
  const firstBarEndingAfter = (ms: number): number => {
    let lo = 0
    let hi = bars.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (bars[mid].endMs > ms) hi = mid
      else lo = mid + 1
    }
    return lo
  }

  for (const event of events) {
    for (let i = firstBarEndingAfter(event.start_ms); i < bars.length; i++) {
      const bar = bars[i]
      if (bar.startMs >= event.end_ms) break
      const fromMs = Math.max(event.start_ms, bar.startMs)
      const toMs = Math.min(event.end_ms, bar.endMs)
      if (toMs > fromMs) {
        bar.chords.push({ event, fromMs, toMs, startsHere: event.start_ms >= bar.startMs })
      }
    }
  }

  return {
    bars,
    durationMs,
    firstBarMs,
    barAtMs(ms) {
      if (bars.length === 0 || ms < firstBarMs) return -1
      const last = bars.length - 1
      return Math.min(firstBarEndingAfter(ms), last)
    },
    barStartMs: (index) => bars[index].startMs,
    barEndMs: (index) => bars[index].endMs,
  }
}
