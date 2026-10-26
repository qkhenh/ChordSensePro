import { clampBpm } from './beats'

const RESET_AFTER_MS = 2000
const KEEP = 8

/** Tap Tempo: the median gap of the latest taps, so one clumsy tap does not spoil the tempo. */
export class TapTempo {
  private taps: number[] = []

  /** Register a tap at `nowMs`; the BPM once there are at least two taps, else null. */
  tap(nowMs: number): number | null {
    const last = this.taps[this.taps.length - 1]
    if (last !== undefined && nowMs === last) return this.estimate()
    if (last !== undefined && nowMs - last > RESET_AFTER_MS) this.taps = []
    this.taps.push(nowMs)
    if (this.taps.length > KEEP) this.taps.shift()
    return this.estimate()
  }

  reset(): void {
    this.taps = []
  }

  private estimate(): number | null {
    if (this.taps.length < 2) return null
    const gaps = this.taps.slice(1).map((t, i) => t - this.taps[i])
    gaps.sort((a, b) => a - b)
    const middle = gaps.length >> 1
    const median = gaps.length % 2 ? gaps[middle] : (gaps[middle - 1] + gaps[middle]) / 2
    return Math.round(clampBpm(60_000 / median) * 10) / 10
  }
}
