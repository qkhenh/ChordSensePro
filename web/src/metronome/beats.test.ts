import { describe, expect, it } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { clampBpm, manualBeats, MAX_BPM, MIN_BPM, songBeats } from './beats'

function meta(over: Partial<AnalysisMetadata['beat_grid']> = {}, duration = 20_000): AnalysisMetadata {
  return {
    duration_ms: duration,
    key: { tonic: 'C', mode: 'major', confidence: 0.9 },
    beat_grid: {
      tempo_bpm: 120,
      tempo_confidence: 0.9,
      time_signature: { numerator: 4, denominator: 4 },
      beats_per_bar: 4,
      first_beat_offset_ms: 0,
      downbeats_ms: null,
      ...over,
    },
    model_version: 'm',
    pipeline_version: 'p',
  }
}

describe('songBeats', () => {
  it('4/4 at 120 BPM: a beat every 500 ms, the first of each bar accented', () => {
    const beats = songBeats(buildGrid(meta(), [])).beatsBetween(0, 4000)
    expect(beats.map((b) => b.ms)).toEqual([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    expect(beats.map((b) => b.accent)).toEqual([true, false, false, false, true, false, false, false])
  })

  it('windows are half open, so consecutive windows neither repeat nor skip a beat', () => {
    const source = songBeats(buildGrid(meta(), []))
    const edges = [0, 700, 1500, 1501, 3300, 4000]
    const joined = edges.slice(0, -1).flatMap((from, i) => source.beatsBetween(from, edges[i + 1]))
    expect(joined).toEqual(source.beatsBetween(0, 4000))
  })

  it('6/8 tracked in dotted quarters has two beats per bar', () => {
    const grid = buildGrid(
      meta({ tempo_bpm: 60, beats_per_bar: 2, time_signature: { numerator: 6, denominator: 8 } }),
      [],
    )
    const beats = songBeats(grid).beatsBetween(0, 4000)
    expect(beats.map((b) => [b.ms, b.accent])).toEqual([
      [0, true],
      [1000, false],
      [2000, true],
      [3000, false],
    ])
  })

  it('irregular downbeats spread the beats evenly inside each bar', () => {
    const grid = buildGrid(meta({ downbeats_ms: [0, 1900, 4100] }, 6000), [])
    const beats = songBeats(grid).beatsBetween(1900, 4100)
    expect(beats.map((b) => b.ms)).toEqual([1900, 2450, 3000, 3550])
    expect(beats[0].accent).toBe(true)
  })

  it('makes no click before the first beat', () => {
    const source = songBeats(buildGrid(meta({ first_beat_offset_ms: 700 }), []))
    expect(source.beatsBetween(0, 700)).toEqual([])
    expect(source.beatsBetween(0, 701)[0]).toEqual({ ms: 700, accent: true })
  })

  it('an empty or reversed window has no beats', () => {
    const source = songBeats(buildGrid(meta(), []))
    expect(source.beatsBetween(500, 500)).toEqual([])
    expect(source.beatsBetween(900, 100)).toEqual([])
  })

  it('a song with no bars has no beats', () => {
    const empty = buildGrid(meta({ first_beat_offset_ms: 99_999 }), [])
    expect(songBeats(empty).beatsBetween(0, 50_000)).toEqual([])
  })
})

describe('manualBeats', () => {
  it('60 BPM in 4: a click every second, accent on every fourth', () => {
    const beats = manualBeats(60, 4).beatsBetween(0, 5000)
    expect(beats.map((b) => b.ms)).toEqual([0, 1000, 2000, 3000, 4000])
    expect(beats.map((b) => b.accent)).toEqual([true, false, false, false, true])
  })

  it('finds the beats in a window far from the start without drifting', () => {
    const beats = manualBeats(90, 3).beatsBetween(600_000, 601_000)
    expect(beats).toHaveLength(2)
    const k = beats[0].ms / (60_000 / 90)
    expect(k).toBeCloseTo(Math.round(k), 6) // on the beat grid, not between beats
    expect(beats[1].ms - beats[0].ms).toBeCloseTo(60_000 / 90, 6)
  })

  it('accents the downbeat by counting from zero in every window', () => {
    const source = manualBeats(120, 4)
    expect(source.beatsBetween(1900, 2100)).toEqual([{ ms: 2000, accent: true }])
    expect(source.beatsBetween(2400, 2600)).toEqual([{ ms: 2500, accent: false }])
  })
})

describe('clampBpm', () => {
  it.each([
    [10, MIN_BPM],
    [29.9, MIN_BPM],
    [30, 30],
    [120, 120],
    [280, 280],
    [999, MAX_BPM],
  ])('%f becomes %f', (input, expected) => {
    expect(clampBpm(input)).toBe(expected)
  })

  it('keeps manualBeats inside the range too', () => {
    expect(manualBeats(10, 4).beatsBetween(0, 2001).map((b) => b.ms)).toEqual([0, 2000])
    expect(manualBeats(999, 4).beatsBetween(0, 215).length).toBe(2) // 280 BPM: every ~214.3 ms
  })
})
