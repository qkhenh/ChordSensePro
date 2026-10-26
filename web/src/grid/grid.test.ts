import { describe, expect, it } from 'vitest'
import type { AnalysisMetadata, AnalysisResult, ChordEvent } from '../api/types'
import { buildGrid } from './grid'

const results = Object.values(
  import.meta.glob<AnalysisResult>('../../../contracts/fixtures/results/*.json', {
    eager: true,
    import: 'default',
  }),
)

interface Opts {
  tempo?: number
  offset?: number
  perBar?: number
  signature?: [number, number]
  duration?: number
  downbeats?: number[] | null
}

function meta({ tempo = 120, offset = 0, perBar = 4, signature = [4, 4], duration = 20000, downbeats = null }: Opts = {}): AnalysisMetadata {
  return {
    duration_ms: duration,
    key: { tonic: 'C', mode: 'major', confidence: 0.9 },
    beat_grid: {
      tempo_bpm: tempo,
      tempo_confidence: 0.9,
      time_signature: { numerator: signature[0], denominator: signature[1] as 4 },
      beats_per_bar: perBar,
      first_beat_offset_ms: offset,
      downbeats_ms: downbeats,
    },
    model_version: 'm',
    pipeline_version: 'p',
  }
}

const chord = (id: string, start: number, end: number, symbol = 'C'): ChordEvent => ({
  event_id: id,
  start_ms: start,
  end_ms: end,
  chord: { root: 'C', quality: 'major', extension: 'none', bass: null },
  symbol,
  confidence: null,
  is_user_edited: false,
})

describe('bar boundaries', () => {
  it('4/4 at 120 BPM with no offset: bar k starts at k·2000 ms with beats every 500 ms', () => {
    const grid = buildGrid(meta(), [])
    expect(grid.bars.slice(0, 3).map((b) => b.startMs)).toEqual([0, 2000, 4000])
    expect(grid.bars[1].beatsMs).toEqual([2000, 2500, 3000, 3500])
    expect(grid.bars).toHaveLength(10)
  })

  it('the first beat offset shifts everything and leaves a pre-roll before it', () => {
    const grid = buildGrid(meta({ offset: 350 }), [])
    expect(grid.firstBarMs).toBe(350)
    expect(grid.bars[0].startMs).toBe(350)
    expect(grid.bars[1].startMs).toBe(2350)
    expect(grid.barAtMs(349)).toBe(-1)
    expect(grid.barAtMs(350)).toBe(0)
  })

  it('3/4 at 90 BPM has 2000 ms bars of three beats', () => {
    const grid = buildGrid(meta({ tempo: 90, perBar: 3, signature: [3, 4] }), [])
    expect(grid.bars[0].endMs - grid.bars[0].startMs).toBe(2000)
    expect(grid.bars[0].beatsMs).toEqual([0, 667, 1333])
  })

  it('6/8 tracked in dotted quarters has two beats per bar', () => {
    const grid = buildGrid(meta({ tempo: 60, perBar: 2, signature: [6, 8] }), [])
    expect(grid.bars[0].beatsMs).toEqual([0, 1000])
    expect(grid.bars[1].startMs).toBe(2000)
  })

  it('a fractional tempo does not drift: bar 100 stays within a millisecond of the exact value', () => {
    const grid = buildGrid(meta({ tempo: 133.33, offset: 123, duration: 400_000 }), [])
    const exact = 123 + 100 * ((4 * 60000) / 133.33)
    expect(Math.abs(grid.bars[100].startMs - exact)).toBeLessThanOrEqual(1)
  })

  it('the last bar is cut at the end of the song', () => {
    const grid = buildGrid(meta({ duration: 9000 }), [])
    const last = grid.bars[grid.bars.length - 1]
    expect([last.startMs, last.endMs]).toEqual([8000, 9000])
    expect(last.beatsMs).toEqual([8000, 8500]) // only the beats that still fit
  })

  it('explicit downbeats win over the interpolated grid (rubato)', () => {
    const grid = buildGrid(meta({ downbeats: [0, 1900, 4100], duration: 6000 }), [])
    expect(grid.bars.map((b) => [b.startMs, b.endMs])).toEqual([
      [0, 1900],
      [1900, 4100],
      [4100, 6000],
    ])
    expect(grid.bars[0].beatsMs).toEqual([0, 475, 950, 1425])
    expect(grid.bars[1].beatsMs[1]).toBe(1900 + 550)
  })

  it('downbeats that start late leave a pre-roll, and ones past the end are dropped', () => {
    const grid = buildGrid(meta({ downbeats: [700, 2700, 99999], duration: 5000 }), [])
    expect(grid.firstBarMs).toBe(700)
    expect(grid.bars).toHaveLength(2)
    expect(grid.barAtMs(699)).toBe(-1)
  })

  it('an offset beyond the end of the song gives no bars at all', () => {
    const grid = buildGrid(meta({ offset: 30000, duration: 9000 }), [])
    expect(grid.bars).toHaveLength(0)
    expect(grid.barAtMs(5000)).toBe(-1)
  })
})

describe('looking a time up', () => {
  const grid = buildGrid(meta({ duration: 9000 }), [])

  it('finds the bar that contains a time, boundaries belonging to the later bar', () => {
    expect(grid.barAtMs(0)).toBe(0)
    expect(grid.barAtMs(1999)).toBe(0)
    expect(grid.barAtMs(2000)).toBe(1)
  })

  it('stays on the last bar at and after the end so the playhead does not vanish', () => {
    expect(grid.barAtMs(8999)).toBe(4)
    expect(grid.barAtMs(9000)).toBe(4)
    expect(grid.barAtMs(60000)).toBe(4)
  })

  it('reads a bar back as milliseconds', () => {
    expect(grid.barStartMs(2)).toBe(4000)
    expect(grid.barEndMs(2)).toBe(6000)
    expect(grid.barEndMs(4)).toBe(9000)
  })

  it('is exact on every bar of every fixture: a time inside a bar finds that bar', () => {
    for (const result of results) {
      const g = buildGrid(result.metadata, result.chord_events)
      for (const bar of g.bars) {
        expect(g.barAtMs(bar.startMs)).toBe(bar.index)
        expect(g.barAtMs(bar.endMs - 1)).toBe(bar.index)
      }
    }
  })
})

describe('placing chords in bars', () => {
  it('splits an event across the bars it spans and marks where it starts', () => {
    const grid = buildGrid(meta({ duration: 12000 }), [chord('e1', 5000, 9000)])
    const spans = grid.bars
      .filter((b) => b.chords.length > 0)
      .map((b) => [b.index, b.chords[0].fromMs, b.chords[0].toMs, b.chords[0].startsHere])
    expect(spans).toEqual([
      [2, 5000, 6000, true],
      [3, 6000, 8000, false],
      [4, 8000, 9000, false],
    ])
  })

  it('keeps two chords of one bar in time order', () => {
    const grid = buildGrid(meta(), [chord('a', 0, 1000, 'C'), chord('b', 1000, 2000, 'G')])
    expect(grid.bars[0].chords.map((c) => c.event.symbol)).toEqual(['C', 'G'])
  })

  it('leaves the bars before the first chord empty', () => {
    const grid = buildGrid(meta(), [chord('a', 4000, 6000)])
    expect(grid.bars[0].chords).toEqual([])
    expect(grid.bars[2].chords).toHaveLength(1)
  })

  it('clips an event that begins in the pre-roll', () => {
    const grid = buildGrid(meta({ offset: 1000 }), [chord('a', 0, 3000)])
    expect(grid.bars[0].chords[0]).toMatchObject({ fromMs: 1000, toMs: 3000, startsHere: false })
  })

  it('drops events entirely before the first bar or after the song', () => {
    const grid = buildGrid(meta({ offset: 1000, duration: 5000 }), [chord('a', 0, 800), chord('b', 6000, 7000)])
    expect(grid.bars.every((b) => b.chords.length === 0)).toBe(true)
  })

  it('still draws the bars when there are no events', () => {
    expect(buildGrid(meta({ duration: 9000 }), []).bars).toHaveLength(5)
  })

  it('covers every event of every fixture completely', () => {
    for (const result of results) {
      const g = buildGrid(result.metadata, result.chord_events)
      for (const e of result.chord_events) {
        const covered = g.bars
          .flatMap((b) => b.chords)
          .filter((c) => c.event.event_id === e.event_id)
          .reduce((sum, c) => sum + (c.toMs - c.fromMs), 0)
        expect(covered).toBe(Math.min(e.end_ms, g.durationMs) - Math.max(e.start_ms, g.firstBarMs))
      }
    }
  })
})

describe('properties', () => {
  // mulberry32: a small seeded generator so a failure can be replayed
  function random(seed: number) {
    let a = seed
    return () => {
      a |= 0
      a = (a + 0x6d2b79f5) | 0
      let t = Math.imul(a ^ (a >>> 15), 1 | a)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }

  it('bars tile the song and every time maps to the bar that contains it', () => {
    const rnd = random(20261026)
    for (let run = 0; run < 500; run++) {
      const grid = buildGrid(
        meta({
          tempo: Math.round((40 + rnd() * 200) * 10) / 10,
          offset: Math.floor(rnd() * 2000),
          perBar: 1 + Math.floor(rnd() * 12),
          duration: 10_000 + Math.floor(rnd() * 50_000),
        }),
        [],
      )
      for (let i = 0; i + 1 < grid.bars.length; i++) {
        expect(grid.bars[i].endMs).toBe(grid.bars[i + 1].startMs)
      }
      const ms = Math.floor(grid.firstBarMs + rnd() * (grid.durationMs - grid.firstBarMs))
      const index = grid.barAtMs(ms)
      expect(grid.barStartMs(index)).toBeLessThanOrEqual(ms)
      expect(ms).toBeLessThan(grid.barEndMs(index))
    }
  })
})
