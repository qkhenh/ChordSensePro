import { describe, expect, it } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { LoopController } from './loop'

const meta: AnalysisMetadata = {
  duration_ms: 40_000,
  key: { tonic: 'C', mode: 'major', confidence: 0.9 },
  beat_grid: {
    tempo_bpm: 120,
    tempo_confidence: 0.9,
    time_signature: { numerator: 4, denominator: 4 },
    beats_per_bar: 4,
    first_beat_offset_ms: 0,
    downbeats_ms: null,
  },
  model_version: 'm',
  pipeline_version: 'p',
}
const grid = buildGrid(meta, []) // 20 bars of 2000 ms

describe('choosing the range', () => {
  it('starts with no loop', () => {
    expect(new LoopController().range).toBeNull()
  })

  it('accepts a range whose end is after its start', () => {
    const loop = new LoopController()
    expect(loop.set(8000, 16000)).toBe(true)
    expect(loop.range).toEqual({ startMs: 8000, endMs: 16000 })
  })

  it.each([
    [8000, 8000],
    [16000, 8000],
    [-1, 5000],
    [Number.NaN, 5000],
    [1000, Number.POSITIVE_INFINITY],
  ])('rejects %f–%f and keeps what was set before', (start, end) => {
    const loop = new LoopController()
    loop.set(1000, 2000)
    expect(loop.set(start, end)).toBe(false)
    expect(loop.range).toEqual({ startMs: 1000, endMs: 2000 })
  })

  it('cuts a loop that runs past the end of the song at the end of the song', () => {
    const loop = new LoopController(20_000)
    expect(loop.set(8000, 30_000)).toBe(true)
    expect(loop.range).toEqual({ startMs: 8000, endMs: 20_000 })
  })

  it('refuses a loop that starts at or after the end of the song', () => {
    const loop = new LoopController(20_000)
    expect(loop.set(20_000, 25_000)).toBe(false)
    expect(loop.range).toBeNull()
  })

  it('turns bars 9 to 16 (indexes 8 to 15) into their whole span, whichever way they were dragged', () => {
    const forward = new LoopController()
    const backward = new LoopController()
    expect(forward.setFromBars(grid, 8, 15)).toBe(true)
    expect(backward.setFromBars(grid, 15, 8)).toBe(true)
    expect(forward.range).toEqual({ startMs: 16_000, endMs: 32_000 })
    expect(backward.range).toEqual(forward.range)
  })

  it('a single bar is a valid loop', () => {
    const loop = new LoopController()
    expect(loop.setFromBars(grid, 3, 3)).toBe(true)
    expect(loop.range).toEqual({ startMs: 6000, endMs: 8000 })
  })

  it('keeps bar numbers inside the song and refuses nonsense', () => {
    const loop = new LoopController()
    expect(loop.setFromBars(grid, 18, 99)).toBe(true)
    expect(loop.range).toEqual({ startMs: 36_000, endMs: 40_000 })
    expect(loop.setFromBars(grid, Number.NaN, 2)).toBe(false)
    const empty = buildGrid({ ...meta, beat_grid: { ...meta.beat_grid, first_beat_offset_ms: 99_999 } }, [])
    expect(loop.setFromBars(empty, 0, 1)).toBe(false) // a song with no bars has nothing to loop
  })

  it('can be cleared', () => {
    const loop = new LoopController()
    loop.set(1000, 2000)
    loop.clear()
    expect(loop.range).toBeNull()
  })
})

describe('wrapping around', () => {
  function playing(loop: LoopController, from: number, to: number, step = 16) {
    // Plays like the real player: after a wrap the position continues from where it was sent.
    const jumps: [number, number][] = []
    let wraps = 0
    for (let pos = from; pos <= to && wraps < 1000; pos += step) {
      const back = loop.tick(pos)
      if (back !== null) {
        jumps.push([pos, back])
        pos = back
        wraps += 1
        if (wraps >= 1) break // one pass is enough for these checks
      }
    }
    return jumps
  }

  it('sends the playhead back to the start the moment it passes the end', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(7990)
    expect(loop.tick(7990)).toBeNull()
    expect(loop.tick(8010)).toBeNull()
    expect(loop.tick(15_990)).toBeNull()
    expect(loop.tick(16_005)).toBe(8000)
  })

  it('does not wrap again right after wrapping', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(15_000)
    expect(loop.tick(16_010)).toBe(8000)
    expect(loop.tick(8020)).toBeNull()
  })

  it('goes round again and again while it keeps playing', () => {
    const loop = new LoopController()
    loop.set(0, 1000)
    loop.noteSeek(0)
    const jumps: number[] = []
    let pos = 0
    for (let i = 0; i < 400; i++) {
      pos += 16
      const back = loop.tick(pos)
      if (back !== null) {
        jumps.push(pos)
        pos = back
      }
    }
    expect(jumps.length).toBeGreaterThanOrEqual(5)
  })

  it('wraps even when a slow frame overshoots the end by a lot', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(15_900)
    expect(loop.tick(16_400)).toBe(8000)
  })

  it('does not drag the user back after they seek out of the loop', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(10_000)
    loop.tick(10_000)
    loop.noteSeek(20_000)
    expect(loop.tick(20_100)).toBeNull()
    expect(playing(loop, 20_100, 30_000)).toEqual([])
  })

  it('still loops after the user seeks back into it', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(20_000)
    loop.noteSeek(9000)
    expect(playing(loop, 9000, 16_100)).toEqual([[expect.any(Number), 8000]])
  })

  it('lets playback run into the loop from before it', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(0)
    expect(playing(loop, 0, 16_100, 100).map(([, back]) => back)).toEqual([8000])
  })

  it('stops looping as soon as it is cleared, even mid-pass', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(12_000)
    loop.tick(12_000)
    loop.clear()
    expect(loop.tick(16_050)).toBeNull()
  })

  it('knows nothing of speed: the same positions give the same answers', () => {
    const slow = new LoopController()
    const fast = new LoopController()
    for (const loop of [slow, fast]) {
      loop.set(8000, 16_000)
      loop.noteSeek(15_000)
    }
    const positions = [15_400, 15_900, 16_100, 8_200]
    expect(positions.map((p) => slow.tick(p))).toEqual(positions.map((p) => fast.tick(p)))
  })

  it('has nothing to do without a loop', () => {
    const loop = new LoopController()
    expect(loop.tick(1234)).toBeNull()
  })

  it('a new range chosen mid-play takes effect for the next pass', () => {
    const loop = new LoopController()
    loop.set(8000, 16_000)
    loop.noteSeek(12_000)
    loop.tick(12_000)
    loop.set(10_000, 14_000)
    expect(loop.tick(14_050)).toBe(10_000)
  })
})

describe('jumps made outside the app', () => {
  it('does not wrap when the playhead leaps past B in one step (a scrub in an embedded player)', () => {
    const loop = new LoopController()
    loop.set(8000, 16000)
    loop.tick(10_000)
    expect(loop.tick(30_000)).toBeNull()
    expect(loop.tick(30_016)).toBeNull()
  })

  it('still wraps after a throttled gap of a second or two', () => {
    const loop = new LoopController()
    loop.set(8000, 16000)
    loop.tick(15_000)
    expect(loop.tick(16_900)).toBe(8000)
  })
})
