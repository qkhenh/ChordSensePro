import { afterEach, describe, expect, it, vi } from 'vitest'
import { manualBeats } from './beats'
import { MetronomeScheduler, type PlaybackClock } from './scheduler'

interface Click {
  when: number
  accent: boolean
}

function setup(over: { playing?: boolean; rate?: number } = {}) {
  const cancelled: number[] = []
  const state = { songMs: 0, ctxTime: 10, rate: over.rate ?? 1, playing: over.playing ?? true }
  const clock: PlaybackClock = { sample: () => ({ ...state }) }
  const clicks: Click[] = []
  const scheduler = new MetronomeScheduler({} as BaseAudioContext, manualBeats(120, 4), clock, {
    lookaheadMs: 1200,
    tickMs: 25,
    schedule: (_ctx, when, opts) => {
      clicks.push({ when, accent: opts.accent })
      return () => cancelled.push(when)
    },
  })
  return { state, scheduler, clicks, cancelled }
}

afterEach(() => vi.useRealTimers())

describe('what a tick schedules', () => {
  it('the beats inside the look-ahead window, at the right audio-clock times', () => {
    const { scheduler, clicks } = setup()
    scheduler.tick()
    expect(clicks.map((c) => c.when)).toEqual([10, 10.5, 11])
    expect(clicks.map((c) => c.accent)).toEqual([true, false, false])
  })

  it('never schedules the same beat twice, tick after tick', () => {
    const { scheduler, state, clicks } = setup()
    scheduler.tick()
    state.songMs = 100
    state.ctxTime = 10.1
    scheduler.tick()
    state.songMs = 400
    state.ctxTime = 10.4
    scheduler.tick()
    expect(clicks.map((c) => Math.round(c.when * 1000))).toEqual([10_000, 10_500, 11_000, 11_500])
  })

  it('plays beats further apart when the song plays slower (0.75x)', () => {
    const { scheduler, clicks } = setup({ rate: 0.75 })
    scheduler.tick()
    const gaps = clicks.slice(1).map((c, i) => c.when - clicks[i].when)
    expect(gaps.length).toBeGreaterThan(0)
    for (const gap of gaps) expect(gap).toBeCloseTo(0.5 / 0.75, 6)
  })

  it('looks ahead by real time: less song at half speed, more at double speed', () => {
    const slow = setup({ rate: 0.5 })
    slow.scheduler.tick()
    const fast = setup({ rate: 2 })
    fast.scheduler.tick()
    expect(slow.clicks.length).toBe(2) // 600 ms of song: beats at 0 and 500
    expect(fast.clicks.length).toBe(5) // 2400 ms of song: 0, 500, 1000, 1500, 2000
  })

  it('aims at the beat that is just ahead of the playhead', () => {
    const { scheduler, state, clicks } = setup()
    state.songMs = 1990
    scheduler.tick()
    expect(clicks[0].when).toBeCloseTo(10.01, 6)
    expect(clicks[0].accent).toBe(true) // 2000 ms is a downbeat
  })
})

describe('when time does not flow smoothly', () => {
  it('does not play the beats it missed while the tab was hidden', () => {
    const { scheduler, state, clicks } = setup()
    scheduler.tick() // beats 0, 500, 1000
    clicks.length = 0
    state.songMs = 5100 // the page was throttled for 5 s
    state.ctxTime = 15.1
    scheduler.tick()
    expect(clicks.every((c) => c.when >= 15.1 - 1e-9)).toBe(true)
    expect(clicks.length).toBeLessThanOrEqual(3) // only what is ahead, not 8 queued-up clicks
  })

  it('starts clean after a seek', () => {
    const { scheduler, state, clicks } = setup()
    state.songMs = 3000
    scheduler.tick()
    clicks.length = 0
    state.songMs = 0 // the user went back to the start
    state.ctxTime = 10.2
    scheduler.reset()
    scheduler.tick()
    expect(clicks.length).toBe(3)
    expect(clicks[0].when).toBeCloseTo(10.2, 6)
  })

  it('does not repeat a window already scheduled when nothing was reset', () => {
    const { scheduler, state, clicks } = setup()
    scheduler.tick()
    clicks.length = 0
    state.songMs = 0
    scheduler.tick()
    expect(clicks).toEqual([])
  })

  it('stays silent while the song is paused and picks up when it resumes', () => {
    const { scheduler, state, clicks } = setup({ playing: false })
    scheduler.tick()
    expect(clicks).toEqual([])
    state.playing = true
    scheduler.tick()
    expect(clicks.length).toBe(3)
  })

  it('swapping the beat source starts afresh', () => {
    const { scheduler, clicks } = setup()
    scheduler.tick()
    clicks.length = 0
    scheduler.setSource(manualBeats(60, 4))
    scheduler.tick()
    expect(clicks.map((c) => c.when)).toEqual([10, 11])
  })
})

describe('clicks that are already queued', () => {
  it('are cancelled when the song is paused, so nothing sounds after the pause', () => {
    const { scheduler, state, cancelled } = setup()
    scheduler.tick() // queues 10, 10.5, 11
    state.songMs = 100
    state.ctxTime = 10.1
    state.playing = false
    scheduler.tick()
    expect(cancelled.sort()).toEqual([10.5, 11]) // 10 is already in the past
  })

  it('are cancelled on a seek, so no stray click plays at the old position', () => {
    const { scheduler, state, cancelled } = setup()
    scheduler.tick()
    state.songMs = 9000
    state.ctxTime = 10.2
    scheduler.reset()
    expect(cancelled.sort()).toEqual([10.5, 11])
  })

  it('are left alone while the song just keeps playing', () => {
    const { scheduler, state, cancelled } = setup()
    scheduler.tick()
    state.songMs = 100
    state.ctxTime = 10.1
    scheduler.tick()
    expect(cancelled).toEqual([])
  })

  it('a schedule function that returns nothing is fine', () => {
    const state = { songMs: 0, ctxTime: 1, rate: 1, playing: true }
    const scheduler = new MetronomeScheduler({} as BaseAudioContext, manualBeats(120, 4), { sample: () => ({ ...state }) }, {
      schedule: () => undefined,
    })
    scheduler.tick()
    expect(() => scheduler.reset()).not.toThrow()
  })
})

describe('running on a timer', () => {
  it('ticks right away and then every tick interval until stopped', () => {
    vi.useFakeTimers()
    const { scheduler, state, clicks } = setup()
    scheduler.start()
    expect(clicks.length).toBe(3)
    state.songMs = 600
    state.ctxTime = 10.6
    vi.advanceTimersByTime(25)
    expect(clicks.length).toBe(4) // the beat at 1500 entered the window
    scheduler.stop()
    state.songMs = 5000
    vi.advanceTimersByTime(1000)
    expect(clicks.length).toBe(4)
  })

  it('starting twice does not double the timer', () => {
    vi.useFakeTimers()
    const { scheduler, state, clicks } = setup()
    scheduler.start()
    scheduler.start()
    state.songMs = 600
    state.ctxTime = 10.6
    vi.advanceTimersByTime(25)
    expect(clicks.length).toBe(4)
    scheduler.stop()
  })

  it('can be restarted after a stop', () => {
    vi.useFakeTimers()
    const { scheduler, clicks } = setup()
    scheduler.start()
    scheduler.stop()
    clicks.length = 0
    scheduler.start()
    expect(clicks.length).toBe(3)
    scheduler.stop()
  })
})
