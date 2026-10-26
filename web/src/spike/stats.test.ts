import { describe, expect, it } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { flipLags, offsetMs, percentile, summarize } from './stats'

const meta: AnalysisMetadata = {
  duration_ms: 20_000,
  key: { tonic: 'C', mode: 'major', confidence: 0.9 },
  beat_grid: {
    tempo_bpm: 120,
    tempo_confidence: 0.9,
    time_signature: { numerator: 4, denominator: 4 },
    beats_per_bar: 4,
    first_beat_offset_ms: 1000,
    downbeats_ms: null,
  },
  model_version: 'm',
  pipeline_version: 'p',
}
const grid = buildGrid(meta, []) // bars: 1000-3000, 3000-5000, …

describe('offsetMs: how far the lit bar is from the bar that should be lit', () => {
  it('is zero when the right bar is lit', () => {
    expect(offsetMs(grid, 3500, 1)).toBe(0)
  })

  it('is the distance to the lit bar when it is one behind', () => {
    expect(offsetMs(grid, 3040, 0)).toBe(40) // bar 0 ended at 3000
  })

  it('is the distance to the lit bar when it is one ahead', () => {
    expect(offsetMs(grid, 2960, 1)).toBe(40) // bar 1 starts at 3000
  })

  it('counts a bar lit in the pre-roll as zero off, since nothing should be lit there', () => {
    expect(offsetMs(grid, 400, -1)).toBe(0)
  })

  it('counts nothing lit when a bar should be as the time back to the first bar', () => {
    expect(offsetMs(grid, 1200, -1)).toBe(200)
  })

  it('is large when the light is far away', () => {
    expect(offsetMs(grid, 10_500, 0)).toBe(7500)
  })
})

describe('percentile', () => {
  it('reads the nearest rank of sorted values', () => {
    const values = Array.from({ length: 100 }, (_, i) => i + 1)
    expect(percentile(values, 50)).toBe(50)
    expect(percentile(values, 95)).toBe(95)
    expect(percentile(values, 100)).toBe(100)
  })

  it('does not depend on the order it was given', () => {
    expect(percentile([5, 1, 3, 2, 4], 80)).toBe(4)
  })

  it('is zero for no values', () => {
    expect(percentile([], 95)).toBe(0)
  })
})

describe('summarize', () => {
  it('reports the count, median, p95, worst case and frame rate', () => {
    const samples = Array.from({ length: 200 }, (_, i) => ({
      at: i * 16.6667,
      audioMs: 1500 + i * 16,
      shown: 0,
      offset: i < 190 ? 0 : 60,
    }))
    const stats = summarize(samples)
    expect(stats.samples).toBe(200)
    expect(stats.p50).toBe(0)
    expect(stats.p95).toBe(0)
    expect(stats.max).toBe(60)
    expect(stats.meanFps).toBeCloseTo(60, 0)
  })

  it('has p95 above zero when more than 5 percent of frames are off', () => {
    const samples = Array.from({ length: 100 }, (_, i) => ({ at: i * 16, audioMs: i, shown: 0, offset: i < 90 ? 0 : 30 }))
    expect(summarize(samples).p95).toBe(30)
  })

  it('copes with no samples', () => {
    expect(summarize([])).toEqual({ samples: 0, p50: 0, p95: 0, max: 0, meanFps: 0, flips: 0, lagP95: 0, lagMax: 0 })
  })
})

describe('flipLags: how late the light changes after the song crosses into the next bar', () => {
  const at = (audioMs: number, shown: number) => ({ at: 0, audioMs, shown, offset: 0 })

  it('is the song time at the first frame showing the new bar, minus where that bar begins', () => {
    const lags = flipLags(grid, [at(2970, 0), at(2988, 0), at(3012, 1), at(3030, 1), at(4990, 1), at(5004, 2)])
    expect(lags).toEqual([12, 4]) // bar 1 begins at 3000, bar 2 at 5000
  })

  it('ignores the very first frame, which has nothing before it to compare with', () => {
    expect(flipLags(grid, [at(3012, 1)])).toEqual([])
  })

  it('counts the light coming on for the first bar after the pre-roll', () => {
    expect(flipLags(grid, [at(990, -1), at(1008, 0)])).toEqual([8])
  })

  it('does not count a jump back (the user seeked) or a skip over several bars', () => {
    expect(flipLags(grid, [at(9000, 4), at(1200, 0), at(1216, 0), at(1300, 2)])).toEqual([])
  })

  it('reports no lag when the light went out', () => {
    expect(flipLags(grid, [at(1200, 0), at(400, -1)])).toEqual([])
  })

  it('feeds the summary: the p95 and the worst lag', () => {
    const samples = [at(2970, 0), at(3012, 1), at(4990, 1), at(5030, 2), at(6990, 2), at(7004, 3)]
    const stats = summarize(samples, flipLags(grid, samples))
    expect(stats.flips).toBe(3)
    expect(stats.lagMax).toBe(30)
    expect(stats.lagP95).toBe(30)
  })
})
