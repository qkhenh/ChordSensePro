import type { Grid } from '../grid/grid'

export interface LoopRange {
  startMs: number
  endMs: number
}

/**
 * The A/B loop of the practice workspace, in song milliseconds (so playback speed is irrelevant).
 *
 * The player calls `tick` with the playhead position on every frame; a non-null answer is the
 * position to seek back to. Only a pass from inside the loop through its end wraps: seeking out
 * of the loop (`noteSeek`) never drags the user back.
 */
export class LoopController {
  private current: LoopRange | null = null
  private last: number | null = null
  private readonly durationMs: number

  constructor(durationMs: number = Number.POSITIVE_INFINITY) {
    this.durationMs = durationMs
  }

  get range(): LoopRange | null {
    return this.current
  }

  set(startMs: number, endMs: number): boolean {
    if (!Number.isFinite(startMs) || !(startMs >= 0) || Number.isNaN(endMs)) return false
    const end = Math.min(endMs, this.durationMs)
    if (!Number.isFinite(end) || !(end > startMs)) return false
    this.current = { startMs, endMs: end }
    return true
  }

  /** Bars are 0-based indexes, either way round; both ends are included. */
  setFromBars(grid: Grid, a: number, b: number): boolean {
    if (!Number.isInteger(a) || !Number.isInteger(b) || grid.bars.length === 0) return false
    const last = grid.bars.length - 1
    const first = Math.min(Math.max(Math.min(a, b), 0), last)
    const final = Math.min(Math.max(Math.max(a, b), 0), last)
    return this.set(grid.barStartMs(first), grid.barEndMs(final))
  }

  clear(): void {
    this.current = null
  }

  /** The playhead was moved by the user (or the wrap itself): do not treat it as playing through. */
  noteSeek(toMs: number): void {
    this.last = toMs
  }

  tick(positionMs: number): number | null {
    const previous = this.last
    this.last = positionMs
    const loop = this.current
    if (!loop || previous === null) return null
    const wasInside = previous >= loop.startMs && previous < loop.endMs
    if (wasInside && positionMs >= loop.endMs) {
      this.last = loop.startMs
      return loop.startMs
    }
    return null
  }
}
