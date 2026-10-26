import { describe, expect, it } from 'vitest'
import { createFakeSoundCloud } from '../test/fakeEmbeds'
import { runTransportContract } from '../test/transportContract'
import { SoundCloudTransport } from './soundcloud'
import type { TransportError } from './types'

const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

function setup(options: { now?: () => number; playTimeoutMs?: number } = {}) {
  const fake = createFakeSoundCloud()
  const frame = document.createElement('iframe')
  const transport = new SoundCloudTransport(frame, async () => fake.api, options)
  const errors: TransportError[] = []
  transport.on('error', (e) => errors.push(e))
  return { ...fake, frame, transport, errors }
}

async function loaded(options: Parameters<typeof setup>[0] = {}, durationMs = 60_000) {
  const made = setup(options)
  const loading = made.transport.load()
  await settle()
  made.driver.finishLoading(durationMs)
  await loading
  return made
}

runTransportContract(
  'soundcloud',
  () => {
    const { transport, driver } = setup()
    return { transport, driver }
  },
  { rate: false },
)

describe('what the widget cannot do', () => {
  it('has no speed control, but still refuses nonsense speeds', async () => {
    const { transport } = await loaded()
    expect(transport.capabilities.rate).toBe(false)
    expect(() => transport.setRate(0.75)).not.toThrow()
    expect(() => transport.setRate(3 as 1)).toThrow(RangeError)
  })
})

describe('following the widget', () => {
  it('maps its events to transport states', async () => {
    const { driver, transport } = await loaded()
    await transport.play()
    expect(transport.state).toBe('playing')
    transport.pause()
    expect(transport.state).toBe('paused')
    await transport.play()
    driver.finish()
    expect(transport.state).toBe('ended')
  })

  it('learns the length from the widget', async () => {
    expect((await loaded({}, 187_000)).transport.durationMs()).toBe(187_000)
  })

  it('an error before it is ready rejects the load', async () => {
    const made = setup()
    const loading = made.transport.load()
    await settle()
    made.driver.breakDown()
    await expect(loading).rejects.toMatchObject({ code: 'unavailable' })
  })

  it('a widget that never starts is reported as blocked autoplay', async () => {
    const { state, transport, errors } = await loaded({ playTimeoutMs: 20 })
    state.blockAutoplay = true
    await transport.play()
    expect(errors.map((e) => e.code)).toEqual(['autoplay_blocked'])
  })

  it('stops listening to the widget when destroyed', async () => {
    const { transport, state } = await loaded()
    transport.destroy()
    expect(state.unbound.length).toBeGreaterThan(0)
  })
})

describe('reading the position smoothly', () => {
  it('fills the gap between progress events while playing', async () => {
    let now = 0
    const { driver, transport } = await loaded({ now: () => now })
    await transport.play()
    driver.progress(10_000)
    expect(transport.currentTimeMs()).toBe(10_000)
    now += 120
    expect(transport.currentTimeMs()).toBe(10_120)
    now += 130
    driver.progress(10_260) // the next event: back on the widget's own number
    expect(transport.currentTimeMs()).toBe(10_260)
  })

  it('never runs more than a second past the last event', async () => {
    let now = 0
    const { driver, transport } = await loaded({ now: () => now })
    await transport.play()
    driver.progress(10_000)
    now += 9000
    expect(transport.currentTimeMs()).toBe(11_000)
  })

  it('a seek moves the position straight away', async () => {
    let now = 0
    const { transport } = await loaded({ now: () => now })
    await transport.play()
    transport.seek(30_000)
    now += 40
    expect(transport.currentTimeMs()).toBe(30_040)
  })

  it('stands still when paused', async () => {
    let now = 0
    const { driver, transport } = await loaded({ now: () => now })
    await transport.play()
    driver.progress(5000)
    transport.pause()
    now += 3000
    expect(transport.currentTimeMs()).toBe(5000)
  })
})
