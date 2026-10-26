import { describe, expect, it } from 'vitest'
import { TapTempo } from './tap'

function tapAt(tap: TapTempo, times: number[]): (number | null)[] {
  return times.map((t) => tap.tap(t))
}

describe('TapTempo', () => {
  it('needs two taps before it can say anything', () => {
    const tap = new TapTempo()
    expect(tap.tap(0)).toBeNull()
    expect(tap.tap(500)).toBe(120)
  })

  it('four taps 500 ms apart make 120 BPM', () => {
    expect(tapAt(new TapTempo(), [0, 500, 1000, 1500]).at(-1)).toBe(120)
  })

  it('shrugs off one clumsy tap thanks to the median', () => {
    const result = tapAt(new TapTempo(), [0, 500, 1020, 1500, 2000, 2500]).at(-1) as number
    expect(result).toBeGreaterThan(118)
    expect(result).toBeLessThan(122)
  })

  it('only listens to the most recent taps, so a tempo change takes over', () => {
    const times = [0, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500]
    expect(tapAt(new TapTempo(), times).at(-1)).toBe(120)
  })

  it('starts again after a pause of more than two seconds', () => {
    const tap = new TapTempo()
    tapAt(tap, [0, 500, 1000])
    expect(tap.tap(4000)).toBeNull() // a fresh first tap
    expect(tap.tap(4600)).toBe(100)
  })

  it('a gap of exactly two seconds still counts as the same run', () => {
    const tap = new TapTempo()
    tap.tap(0)
    expect(tap.tap(2000)).toBe(30)
  })

  it('never answers outside 30 to 280 BPM', () => {
    expect(tapAt(new TapTempo(), [0, 10, 20, 30]).at(-1)).toBe(280)
    expect(tapAt(new TapTempo(), [0, 2000]).at(-1)).toBe(30)
  })

  it('reset forgets everything', () => {
    const tap = new TapTempo()
    tapAt(tap, [0, 500])
    tap.reset()
    expect(tap.tap(600)).toBeNull()
  })

  it('ignores taps that arrive at the same instant', () => {
    const tap = new TapTempo()
    tap.tap(100)
    expect(tap.tap(100)).toBeNull()
    expect(tap.tap(600)).toBe(120)
  })
})
