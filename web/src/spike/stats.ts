import type { Grid } from '../grid/grid'

export interface Sample {
  /** Frame time (ms). */
  at: number
  /** Where the audio clock was at that frame. */
  audioMs: number
  /** The bar that was lit (-1 for none). */
  shown: number
  /** How far the lit bar was from the bar that should be lit, in ms (0 = right bar). */
  offset: number
}

export interface Stats {
  samples: number
  p50: number
  p95: number
  max: number
  meanFps: number
  /** Bar changes seen while playing forward. */
  flips: number
  /** How late those changes were, in ms of song (p95 and worst case). */
  lagP95: number
  lagMax: number
}

/**
 * NFR-01's measure: the distance, in milliseconds of song, between the bar that is lit and the
 * bar the audio clock says is playing. Zero when they agree.
 */
export function offsetMs(grid: Grid, audioMs: number, shown: number): number {
  const expected = grid.barAtMs(audioMs)
  if (shown === expected) return 0
  if (shown < 0) return Math.max(0, audioMs - grid.firstBarMs) // a bar should be lit but none is
  const bar = grid.bars[shown]
  return audioMs < bar.startMs ? bar.startMs - audioMs : Math.max(0, audioMs - bar.endMs)
}

/** Nearest-rank percentile. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.max(Math.ceil((p / 100) * sorted.length), 1) - 1]
}

/**
 * How late the light changed: for every step to the next bar while playing forward, the song time
 * at the first frame that showed it minus where that bar begins. This is the part of the delay
 * the page controls (at most about one frame); a jump back or over several bars is a seek, not
 * lateness, and is left out.
 */
export function flipLags(grid: Grid, samples: readonly Sample[]): number[] {
  const lags: number[] = []
  for (let i = 1; i < samples.length; i++) {
    const before = samples[i - 1].shown
    const now = samples[i].shown
    if (now >= 0 && now === before + 1) lags.push(samples[i].audioMs - grid.barStartMs(now))
  }
  return lags
}

export function summarize(samples: readonly Sample[], lags: readonly number[] = []): Stats {
  if (samples.length === 0) return { samples: 0, p50: 0, p95: 0, max: 0, meanFps: 0, flips: 0, lagP95: 0, lagMax: 0 }
  const offsets = samples.map((s) => s.offset)
  const seconds = (samples[samples.length - 1].at - samples[0].at) / 1000
  return {
    samples: samples.length,
    p50: percentile(offsets, 50),
    p95: percentile(offsets, 95),
    max: Math.max(...offsets),
    meanFps: seconds > 0 ? (samples.length - 1) / seconds : 0,
    flips: lags.length,
    lagP95: percentile(lags, 95),
    lagMax: lags.length > 0 ? Math.max(...lags) : 0,
  }
}
