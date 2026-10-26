import { describe, expect, it, vi } from 'vitest'
import { FakeAudio } from '../test/fakeMedia'
import { HtmlAudioTransport } from './html'
import type { TransportError } from './types'

function setup(urls: (string | Error | { status: number })[] = ['/a?ticket=1', '/a?ticket=2', '/a?ticket=3']) {
  const audio = new FakeAudio()
  let call = 0
  const source = vi.fn(async () => {
    const next = urls[Math.min(call++, urls.length - 1)]
    if (typeof next === 'string') return next
    throw next
  })
  const transport = new HtmlAudioTransport(audio as unknown as HTMLAudioElement, source)
  const errors: TransportError[] = []
  transport.on('error', (e) => errors.push(e))
  return { audio, source, transport, errors }
}

async function loaded(durationMs = 60_000) {
  const made = setup()
  const loading = made.transport.load()
  await Promise.resolve()
  await Promise.resolve()
  made.audio.finishLoading(durationMs)
  await loading
  return made
}

const tick = () => new Promise((r) => setTimeout(r, 0))

describe('pitch', () => {
  it('asks the browser to keep the pitch when the speed changes', async () => {
    const { audio, transport } = await loaded()
    expect(audio.preservesPitch).toBe(true)
    transport.setRate(0.75)
    expect(audio.playbackRate).toBe(0.75)
    expect(audio.preservesPitch).toBe(true)
  })

  it('also sets the vendor-prefixed flags that older browsers use', () => {
    const audio = new FakeAudio() as FakeAudio & Record<string, unknown>
    new HtmlAudioTransport(audio as unknown as HTMLAudioElement, async () => '/a')
    expect(audio.mozPreservesPitch).toBe(true)
    expect(audio.webkitPreservesPitch).toBe(true)
  })
})

describe('starting', () => {
  it('uses the url the source gives and starts loading it', async () => {
    const { audio, transport } = setup()
    const loading = transport.load()
    await tick()
    expect(audio.src).toBe('/a?ticket=1')
    expect(audio.loads).toBe(1)
    audio.finishLoading(1000)
    await loading
  })

  it('blocked autoplay is reported, not thrown, and leaves the player paused', async () => {
    const { audio, transport, errors } = await loaded()
    audio.blockAutoplay = true
    await expect(transport.play()).resolves.toBeUndefined()
    expect(errors.map((e) => e.code)).toEqual(['autoplay_blocked'])
    expect(transport.state).toBe('paused')
    audio.blockAutoplay = false
    await transport.play()
    expect(transport.state).toBe('playing')
  })

  it('remembers a seek made before the media was ready and applies it once it is', async () => {
    const { audio, transport } = setup()
    const loading = transport.load()
    await tick()
    transport.seek(7000)
    expect(audio.currentTime).toBe(0)
    audio.finishLoading(60_000)
    await loading
    expect(transport.currentTimeMs()).toBe(7000)
  })

  it('a source that cannot give a url fails the load with a useful error', async () => {
    const { transport, errors } = setup([new Error('offline')])
    await expect(transport.load()).rejects.toMatchObject({ code: 'network' })
    expect(errors[0].code).toBe('network')
    expect(transport.state).toBe('error')
  })

  it('a chart without audio on the server is "unavailable", not a network problem', async () => {
    const { transport } = setup([{ status: 404 }])
    await expect(transport.load()).rejects.toMatchObject({ code: 'unavailable' })
  })
})

describe('the end of the song', () => {
  it('reaches the ended state and plays again from the start', async () => {
    const { audio, transport } = await loaded()
    await transport.play()
    audio.end()
    expect(transport.state).toBe('ended')
    await transport.play()
    expect(audio.currentTime).toBe(0)
    expect(transport.state).toBe('playing')
  })
})

describe('a ticket that stops working', () => {
  it('gets a fresh url and carries on from the same spot', async () => {
    const { audio, source, transport } = await loaded()
    await transport.play()
    audio.currentTime = 12.345
    audio.fail(2)
    await tick()
    expect(source).toHaveBeenCalledTimes(2)
    expect(audio.src).toBe('/a?ticket=2')
    audio.finishLoading(60_000)
    await tick()
    expect(transport.currentTimeMs()).toBe(12_345)
    expect(transport.state).toBe('playing')
  })

  it('stays paused after the refresh if it was paused', async () => {
    const { audio, transport } = await loaded()
    audio.currentTime = 3
    audio.fail(2)
    await tick()
    audio.finishLoading(60_000)
    await tick()
    expect(transport.currentTimeMs()).toBe(3000)
    expect(transport.state).not.toBe('playing')
  })

  it('gives up if the fresh url fails as well', async () => {
    const { audio, transport, errors } = await loaded()
    audio.fail(2)
    await tick()
    audio.fail(2)
    await tick()
    expect(errors.map((e) => e.code)).toEqual(['network'])
    expect(transport.state).toBe('error')
  })

  it('can refresh again later, once a refreshed stream has played', async () => {
    const { audio, source, transport, errors } = await loaded()
    await transport.play()
    audio.fail(2)
    await tick()
    audio.finishLoading(60_000)
    await tick() // playing again: the budget is back
    audio.fail(2)
    await tick()
    expect(source).toHaveBeenCalledTimes(3)
    expect(errors).toEqual([])
  })

  it('says "unavailable" when the refresh finds the audio gone', async () => {
    const { audio, transport, errors } = setup(['/a?ticket=1', { status: 404 }])
    const loading = transport.load()
    await tick()
    audio.finishLoading(1000)
    await loading
    audio.fail(4)
    await tick()
    expect(errors.map((e) => e.code)).toEqual(['unavailable'])
  })
})

describe('numbers', () => {
  it('has no length before the media loads, then whole milliseconds', async () => {
    const { audio, transport } = setup()
    expect(transport.durationMs()).toBeNull()
    expect(transport.currentTimeMs()).toBe(0)
    const loading = transport.load()
    await tick()
    audio.finishLoading(12_345.6)
    await loading
    expect(transport.durationMs()).toBe(12_346)
  })
})
