import { describe, expect, it } from 'vitest'
import type { Transport, TransportError, TransportState } from '../transport/types'

/** What the shared tests need from an adapter's fake backend. */
export interface Driver {
  /** The backend has loaded the media and knows its length. */
  finishLoading(durationMs: number): void
  /** Something unrecoverable happens to the backend. */
  breakDown(): void
  /** The playback speed the backend was actually told to use. */
  rate(): number
}

export type Factory = () => { transport: Transport; driver: Driver }

/** A real backend answers after the call returns, never inside it. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

const CODES: TransportError['code'][] = ['autoplay_blocked', 'unavailable', 'embed_blocked', 'network']

async function ready(factory: Factory, durationMs = 60_000) {
  const made = factory()
  const loading = made.transport.load()
  await settle()
  made.driver.finishLoading(durationMs)
  await loading
  return made
}

/** The behaviour every transport must have, whatever sits behind it. */
export function runTransportContract(name: string, factory: Factory, opts: { rate: boolean }) {
  describe(`transport contract: ${name}`, () => {
    it('starts idle, becomes ready once loaded and knows the length', async () => {
      const { transport, driver } = factory()
      expect(transport.state).toBe('idle')
      expect(transport.durationMs()).toBeNull()
      const loading = transport.load()
      expect(transport.state).toBe('loading')
      await settle()
      driver.finishLoading(61_234)
      await loading
      expect(transport.state).toBe('ready')
      expect(transport.durationMs()).toBe(61_234)
    })

    it('plays and pauses', async () => {
      const { transport } = await ready(factory)
      await transport.play()
      expect(transport.state).toBe('playing')
      transport.pause()
      expect(transport.state).toBe('paused')
    })

    it('seeks to a position, clamped to the song', async () => {
      const { transport } = await ready(factory, 60_000)
      transport.seek(5000)
      expect(Math.abs(transport.currentTimeMs() - 5000)).toBeLessThanOrEqual(1)
      transport.seek(-300)
      expect(transport.currentTimeMs()).toBe(0)
      transport.seek(999_999)
      expect(transport.currentTimeMs()).toBe(60_000)
    })

    it('reports whole milliseconds', async () => {
      const { transport } = await ready(factory)
      transport.seek(1234.567)
      expect(Number.isInteger(transport.currentTimeMs())).toBe(true)
    })

    it('applies a supported speed when the source can, and refuses unsupported ones', async () => {
      const { transport, driver } = await ready(factory)
      expect(transport.capabilities.rate).toBe(opts.rate)
      transport.setRate(0.75)
      if (opts.rate) expect(driver.rate()).toBe(0.75)
      for (const bad of [0.8, 2, 0, -1, Number.NaN]) {
        expect(() => transport.setRate(bad as 0.5)).toThrow(RangeError)
      }
    })

    it('tells listeners about every state change, until they unsubscribe', async () => {
      const { transport, driver } = factory()
      const seen: TransportState[] = []
      const off = transport.on('state', () => seen.push(transport.state))
      const loading = transport.load()
      await settle()
      driver.finishLoading(60_000)
      await loading
      await transport.play()
      transport.pause()
      expect(seen).toEqual(['loading', 'ready', 'playing', 'paused'])
      off()
      await transport.play()
      expect(seen).toHaveLength(4)
    })

    it('stops and goes quiet when destroyed', async () => {
      const { transport } = await ready(factory)
      await transport.play()
      const seen: TransportState[] = []
      transport.on('state', () => seen.push(transport.state))
      transport.destroy()
      expect(() => {
        transport.pause()
        transport.seek(1000)
        transport.currentTimeMs()
        transport.durationMs()
      }).not.toThrow()
      await expect(transport.play()).resolves.toBeUndefined()
      expect(transport.state).not.toBe('playing')
      expect(seen.filter((s) => s === 'playing')).toEqual([])
    })

    it('reports an unrecoverable failure as an error with a known code', async () => {
      const { transport, driver } = await ready(factory)
      const errors: TransportError[] = []
      transport.on('error', (e) => errors.push(e))
      driver.breakDown()
      await new Promise((r) => setTimeout(r, 0))
      expect(errors.length).toBeGreaterThan(0)
      expect(CODES).toContain(errors[0].code)
      expect(errors[0].message).not.toBe('')
      expect(transport.state).toBe('error')
    })
  })
}
