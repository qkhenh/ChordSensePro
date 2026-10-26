import { describe, expect, it } from 'vitest'
import type { Chart, ChordEvent, EventOverride } from '../api/types'
import { applyOverrides, OverlapError } from './effective'

const edited = Object.values(
  import.meta.glob<Chart>('../../../contracts/fixtures/charts/autumn_leaves_gm.edited.chart.json', {
    eager: true,
    import: 'default',
  }),
)[0]
const golden = Object.values(
  import.meta.glob<ChordEvent[]>('../../../contracts/fixtures/charts/autumn_leaves_gm.effective.json', {
    eager: true,
    import: 'default',
  }),
)[0]

const event = (id: string, start: number, end: number, symbol = 'C'): ChordEvent => ({
  event_id: id,
  start_ms: start,
  end_ms: end,
  chord: { root: 'C', quality: 'major', extension: 'none', bass: null },
  symbol,
  confidence: null,
  is_user_edited: false,
})

const base = [event('e1', 0, 1000), event('e2', 1000, 2000), event('e3', 2000, 3000)]

describe('applyOverrides', () => {
  it('returns the baseline untouched when there is nothing to apply', () => {
    expect(applyOverrides(base, [])).toEqual(base)
  })

  it('replaces an event with set and marks it as edited', () => {
    const changed = event('e2', 1000, 2000, 'Am')
    const out = applyOverrides(base, [{ event_id: 'e2', op: 'set', event: changed }])
    expect(out[1]).toMatchObject({ event_id: 'e2', symbol: 'Am', is_user_edited: true })
    expect(out[0]).toEqual(base[0])
  })

  it('removes an event with delete', () => {
    const out = applyOverrides(base, [{ event_id: 'e2', op: 'delete', event: null }])
    expect(out.map((e) => e.event_id)).toEqual(['e1', 'e3'])
  })

  it('adds an event with insert and keeps the timeline sorted', () => {
    const gap = [event('e1', 0, 1000), event('e3', 2000, 3000)]
    const out = applyOverrides(gap, [{ event_id: 'e2', op: 'insert', event: event('e2', 1000, 2000, 'F') }])
    expect(out.map((e) => e.event_id)).toEqual(['e1', 'e2', 'e3'])
    expect(out[1].is_user_edited).toBe(true)
  })

  it('ignores a delete of something that is not there', () => {
    expect(applyOverrides(base, [{ event_id: 'nope', op: 'delete', event: null }])).toEqual(base)
  })

  it('applies several overrides in order, the last one for an id winning', () => {
    const overrides: EventOverride[] = [
      { event_id: 'e2', op: 'set', event: event('e2', 1000, 2000, 'Dm') },
      { event_id: 'e2', op: 'set', event: event('e2', 1000, 2000, 'G7') },
    ]
    expect(applyOverrides(base, overrides)[1].symbol).toBe('G7')
  })

  it('refuses a result with overlapping events', () => {
    const overrides: EventOverride[] = [{ event_id: 'e2', op: 'set', event: event('e2', 500, 2000) }]
    expect(() => applyOverrides(base, overrides)).toThrow(OverlapError)
  })

  it('never changes the baseline it was given', () => {
    const copy = structuredClone(base)
    applyOverrides(base, [{ event_id: 'e1', op: 'delete', event: null }])
    expect(base).toEqual(copy)
  })

  it('is the same function as the Python one on the shared fixture', () => {
    expect(applyOverrides(edited.ai_events, edited.overrides)).toEqual(golden)
  })
})
