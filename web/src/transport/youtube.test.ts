import { describe, expect, it } from 'vitest'
import { createFakeYouTube } from '../test/fakeEmbeds'
import { runTransportContract } from '../test/transportContract'
import { YouTubeTransport } from './youtube'
import type { TransportError } from './types'

const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

function setup(options: { now?: () => number; playTimeoutMs?: number } = {}) {
  const fake = createFakeYouTube()
  const host = document.createElement('div')
  const transport = new YouTubeTransport(host, 'dQw4w9WgXcQ', async () => fake.api, options)
  const errors: TransportError[] = []
  transport.on('error', (e) => errors.push(e))
  return { ...fake, host, transport, errors }
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
  'youtube',
  () => {
    const { transport, driver } = setup()
    return { transport, driver }
  },
  { rate: true },
)

describe('creating the player', () => {
  it('embeds the right video in the host element', async () => {
    const { created, host } = await loaded()
    expect(created).toEqual([expect.objectContaining({ host, videoId: 'dQw4w9WgXcQ' })])
  })

  it('cannot even start when the API script is blocked', async () => {
    const host = document.createElement('div')
    const transport = new YouTubeTransport(host, 'x', async () => {
      throw new Error('script blocked')
    })
    await expect(transport.load()).rejects.toMatchObject({ code: 'network' })
    expect(transport.state).toBe('error')
  })

  it('destroys the player with the transport', async () => {
    const { transport, state } = await loaded()
    transport.destroy()
    expect(state.destroyed).toBe(true)
  })
})

describe('when the video cannot be played', () => {
  it.each([
    [101, 'embed_blocked'],
    [150, 'embed_blocked'],
    [100, 'unavailable'],
    [2, 'unavailable'],
    [5, 'unavailable'],
  ])('error %i becomes %s', async (code, expected) => {
    const { driver, errors, transport } = await loaded()
    driver.emitError(code)
    expect(errors.map((e) => e.code)).toEqual([expected])
    expect(transport.state).toBe('error')
  })

  it('rejects the load when the video is refused before it is ready', async () => {
    const made = setup()
    const loading = made.transport.load()
    await settle()
    made.driver.emitError(101)
    await expect(loading).rejects.toMatchObject({ code: 'embed_blocked' })
  })
})

describe('following the player', () => {
  it('maps the player states to transport states', async () => {
    const { driver, transport } = await loaded()
    driver.emitState(1)
    expect(transport.state).toBe('playing')
    driver.emitState(2)
    expect(transport.state).toBe('paused')
    driver.emitState(3)
    expect(transport.state).toBe('loading') // buffering
    driver.emitState(0)
    expect(transport.state).toBe('ended')
    driver.emitState(5)
    expect(transport.state).toBe('ready') // cued again
  })

  it('does not change state for "unstarted" (-1)', async () => {
    const { driver, transport } = await loaded()
    driver.emitState(-1)
    expect(transport.state).toBe('ready')
  })

  it('seeks in seconds and lets the player fetch from further ahead', async () => {
    const { transport, state } = await loaded()
    transport.seek(12_500)
    expect(state.seeks).toEqual([[12.5, true]])
  })
})

describe('reading the position smoothly', () => {
  it('fills the gap between the player\'s coarse updates while playing', async () => {
    let now = 1000
    const { driver, state, transport } = await loaded({ now: () => now })
    state.time = 10
    driver.emitState(1)
    expect(transport.currentTimeMs()).toBe(10_000)
    now += 200
    expect(transport.currentTimeMs()).toBe(10_200)
    state.time = 10.25 // the player caught up
    expect(transport.currentTimeMs()).toBe(10_250)
  })

  it('moves slower when the song plays slower', async () => {
    let now = 0
    const { driver, state, transport } = await loaded({ now: () => now })
    transport.setRate(0.75)
    state.time = 10
    driver.emitState(1)
    transport.currentTimeMs()
    now += 200
    expect(transport.currentTimeMs()).toBe(10_150)
  })

  it('trusts the player alone while buffering, so a stall does not run ahead', async () => {
    let now = 0
    const { driver, state, transport } = await loaded({ now: () => now })
    state.time = 10
    driver.emitState(1)
    transport.currentTimeMs()
    driver.emitState(3)
    now += 800
    expect(transport.currentTimeMs()).toBe(10_000)
    driver.emitState(1)
    state.time = 10.4
    expect(transport.currentTimeMs()).toBe(10_400) // resynchronised from the player
  })

  it('never runs more than half a second past the last real reading', async () => {
    let now = 0
    const { driver, state, transport } = await loaded({ now: () => now })
    state.time = 10
    driver.emitState(1)
    transport.currentTimeMs()
    now += 5000
    expect(transport.currentTimeMs()).toBe(10_500)
  })

  it('stands still when paused', async () => {
    let now = 0
    const { driver, state, transport } = await loaded({ now: () => now })
    state.time = 10
    driver.emitState(2)
    now += 5000
    expect(transport.currentTimeMs()).toBe(10_000)
  })
})

describe('speed', () => {
  it('is available when the player offers 0.5 and 0.75', async () => {
    expect((await loaded()).transport.capabilities.rate).toBe(true)
  })

  it('is switched off, and ignored, when the player does not offer it', async () => {
    const made = setup()
    made.state.rates = [1]
    const loading = made.transport.load()
    await settle()
    made.driver.finishLoading(1000)
    await loading
    expect(made.transport.capabilities.rate).toBe(false)
    made.transport.setRate(0.75)
    expect(made.state.rate).toBe(1)
  })
})

describe('autoplay', () => {
  it('a player that silently refuses to start is reported once the wait runs out', async () => {
    const { state, transport, errors } = await loaded({ playTimeoutMs: 20 })
    state.blockAutoplay = true
    await transport.play()
    expect(errors.map((e) => e.code)).toEqual(['autoplay_blocked'])
    expect(transport.state).toBe('paused')
  })
})

describe('the player UI', () => {
  it('hides the native controls and keyboard shortcuts, so speed, seeking and the loop stay in the app', async () => {
    const { created } = await loaded()
    expect(created[0].playerVars).toMatchObject({ controls: 0, disablekb: 1 })
  })
})
