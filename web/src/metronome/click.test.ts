import { describe, expect, it } from 'vitest'
import { scheduleClick } from './click'

interface Entry {
  freq: number
  type: string
  start: number
  stop: number
  peak: number
}

/** Records what a click asks of the Web Audio graph. */
function fakeContext() {
  const log: Entry[] = []
  const ctx = {
    destination: {},
    createOscillator() {
      const entry: Entry = { freq: 0, type: '', start: -1, stop: -1, peak: 0 }
      log.push(entry)
      return {
        set type(v: string) {
          entry.type = v
        },
        frequency: {
          set value(v: number) {
            entry.freq = v
          },
        },
        connect: () => {},
        start: (t: number) => {
          entry.start = t
        },
        stop: (t: number) => {
          entry.stop = t
        },
      }
    },
    createGain() {
      const entry = () => log[log.length - 1]
      return {
        connect: () => {},
        gain: {
          setValueAtTime: () => {},
          linearRampToValueAtTime: (v: number) => {
            entry().peak = v
          },
          exponentialRampToValueAtTime: () => {},
        },
      }
    },
  }
  return { ctx: ctx as unknown as BaseAudioContext, log }
}

describe('scheduleClick', () => {
  it('starts the sound exactly at the requested time and stops it shortly after', () => {
    const { ctx, log } = fakeContext()
    scheduleClick(ctx, 12.5, { accent: false, timbre: 'woodblock', volume: 0.8 })
    expect(log).toHaveLength(1)
    expect(log[0].start).toBe(12.5)
    expect(log[0].stop).toBeGreaterThan(12.5)
    expect(log[0].stop - 12.5).toBeLessThan(0.2)
  })

  it('makes the accented beat higher and louder than the others', () => {
    const { ctx, log } = fakeContext()
    scheduleClick(ctx, 1, { accent: true, timbre: 'beep', volume: 1 })
    scheduleClick(ctx, 2, { accent: false, timbre: 'beep', volume: 1 })
    expect(log[0].freq).toBeGreaterThan(log[1].freq)
    expect(log[0].peak).toBeGreaterThan(log[1].peak)
  })

  it('never goes louder than the chosen volume', () => {
    const { ctx, log } = fakeContext()
    scheduleClick(ctx, 1, { accent: true, timbre: 'woodblock', volume: 0.3 })
    expect(log[0].peak).toBeLessThanOrEqual(0.3)
  })

  it('has two different voices', () => {
    const { ctx, log } = fakeContext()
    scheduleClick(ctx, 1, { accent: false, timbre: 'woodblock', volume: 1 })
    scheduleClick(ctx, 1, { accent: false, timbre: 'beep', volume: 1 })
    expect([log[0].type, log[0].freq]).not.toEqual([log[1].type, log[1].freq])
  })

  it('a silent click schedules nothing audible', () => {
    const { ctx, log } = fakeContext()
    scheduleClick(ctx, 1, { accent: true, timbre: 'beep', volume: 0 })
    expect(log.length === 0 || log[0].peak === 0).toBe(true)
  })
})
