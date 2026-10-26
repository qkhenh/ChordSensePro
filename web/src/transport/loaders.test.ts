import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { loadSoundCloudApi, loadYouTubeApi, resetLoaders } from './loaders'

const YT_SRC = 'https://www.youtube.com/iframe_api'
const SC_SRC = 'https://w.soundcloud.com/player/api.js'

const scripts = (src: string) =>
  [...document.querySelectorAll('script')].filter((s) => s.getAttribute('src') === src)

type Win = typeof window & { YT?: unknown; SC?: unknown; onYouTubeIframeAPIReady?: () => void }
const win = window as Win

beforeEach(() => resetLoaders())
afterEach(() => {
  for (const s of document.querySelectorAll('script')) s.remove()
  delete win.YT
  delete win.SC
  delete win.onYouTubeIframeAPIReady
})

describe('YouTube', () => {
  it('adds the official script once and resolves when YouTube says it is ready', async () => {
    const first = loadYouTubeApi()
    const second = loadYouTubeApi()
    expect(scripts(YT_SRC)).toHaveLength(1)
    const api = { Player: class {} }
    win.YT = api
    win.onYouTubeIframeAPIReady?.()
    await expect(first).resolves.toBe(api)
    await expect(second).resolves.toBe(api)
  })

  it('does not touch the network when the API is already on the page', async () => {
    const api = { Player: class {} }
    win.YT = api
    await expect(loadYouTubeApi()).resolves.toBe(api)
    expect(scripts(YT_SRC)).toHaveLength(0)
  })

  it('rejects when the script cannot be loaded, and tries again next time', async () => {
    const failing = loadYouTubeApi()
    scripts(YT_SRC)[0].dispatchEvent(new Event('error'))
    await expect(failing).rejects.toThrow(/YouTube/)
    expect(scripts(YT_SRC)).toHaveLength(0) // the broken tag does not stay on the page

    const retry = loadYouTubeApi()
    expect(scripts(YT_SRC)).toHaveLength(1) // a fresh attempt, not the cached failure
    const api = { Player: class {} }
    win.YT = api
    win.onYouTubeIframeAPIReady?.()
    await expect(retry).resolves.toBe(api)
  })

  it('gives up after a timeout instead of waiting forever', async () => {
    resetLoaders({ timeoutMs: 20 })
    await expect(loadYouTubeApi()).rejects.toThrow(/timed out/i)
  })
})

describe('SoundCloud', () => {
  it('adds the widget script once and resolves when it has loaded', async () => {
    const first = loadSoundCloudApi()
    const second = loadSoundCloudApi()
    expect(scripts(SC_SRC)).toHaveLength(1)
    const api = { Widget: () => ({}) }
    win.SC = api
    scripts(SC_SRC)[0].dispatchEvent(new Event('load'))
    await expect(first).resolves.toBe(api)
    await expect(second).resolves.toBe(api)
  })

  it('does not touch the network when the API is already on the page', async () => {
    const api = { Widget: () => ({}) }
    win.SC = api
    await expect(loadSoundCloudApi()).resolves.toBe(api)
    expect(scripts(SC_SRC)).toHaveLength(0)
  })

  it('rejects when the script cannot be loaded, and tries again next time', async () => {
    const failing = loadSoundCloudApi()
    scripts(SC_SRC)[0].dispatchEvent(new Event('error'))
    await expect(failing).rejects.toThrow(/SoundCloud/)
    expect(scripts(SC_SRC)).toHaveLength(0)
    const retry = loadSoundCloudApi()
    expect(scripts(SC_SRC)).toHaveLength(1)
    win.SC = { Widget: () => ({}) }
    scripts(SC_SRC)[0].dispatchEvent(new Event('load'))
    await expect(retry).resolves.toBe(win.SC)
  })
})
