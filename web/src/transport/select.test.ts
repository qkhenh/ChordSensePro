import { describe, expect, it, vi } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { PlaybackSource } from '../api/types'
import { createFakeSoundCloud, createFakeYouTube } from '../test/fakeEmbeds'
import { authResponse, apiError, http, HttpResponse, server } from '../test/server'
import { createTransport } from './select'

const API = '*/api/v1'

async function client() {
  server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
  const c = createApiClient({ storage: memoryStorage() })
  await c.login('ann@example.com', 'secret-pass')
  return c
}

function deps(c: Awaited<ReturnType<typeof client>>) {
  const mount = document.createElement('div')
  const yt = createFakeYouTube()
  const sc = createFakeSoundCloud()
  return { mount, yt, sc, deps: { chartId: 'chart-1', client: c, mount, loaders: { youtube: async () => yt.api, soundcloud: async () => sc.api } } }
}

describe('picking the transport for a chart', () => {
  it('a stored file plays through an <audio> element with a ticket for that chart', async () => {
    const c = await client()
    let asked = ''
    server.use(
      http.post(`${API}/charts/:id/audio-ticket`, ({ params }) => {
        asked = String(params.id)
        return HttpResponse.json({ url: '/api/v1/charts/chart-1/audio?ticket=t', expires_at: '2026-10-26T12:30:00Z' })
      }),
    )
    const made = deps(c)
    const playback: PlaybackSource = { kind: 'file', audio_url: '/api/v1/charts/chart-1/audio', embed_id: null }
    const transport = createTransport(playback, made.deps)
    expect(transport.kind).toBe('html')
    const audioSrc = vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    void transport.load().catch(() => {})
    await vi.waitFor(() => expect(asked).toBe('chart-1'))
    audioSrc.mockRestore()
    transport.destroy()
    expect(made.mount.children).toHaveLength(0) // an <audio> needs no place on the page
  })

  it('a direct link plays from where it is, without asking the server for anything', async () => {
    const c = await client()
    const made = deps(c)
    const transport = createTransport({ kind: 'direct', audio_url: 'https://cdn.example.com/a.mp3', embed_id: null }, made.deps)
    expect(transport.kind).toBe('html')
    const load = vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
    void transport.load().catch(() => {})
    await new Promise((r) => setTimeout(r, 0))
    load.mockRestore()
    transport.destroy()
  })

  it('YouTube gets a host element in the page and the right video', async () => {
    const made = deps(await client())
    const transport = createTransport({ kind: 'youtube', audio_url: null, embed_id: 'dQw4w9WgXcQ' }, made.deps)
    expect(transport.kind).toBe('youtube')
    expect(made.mount.children).toHaveLength(1)
    void transport.load()
    await new Promise((r) => setTimeout(r, 0))
    expect(made.yt.created[0].videoId).toBe('dQw4w9WgXcQ')
    expect(made.yt.created[0].host).toBe(made.mount.children[0])
    transport.destroy()
  })

  it('SoundCloud gets its widget iframe pointed at the track', async () => {
    const made = deps(await client())
    const track = 'https://soundcloud.com/artist/track'
    const transport = createTransport({ kind: 'soundcloud', audio_url: null, embed_id: track }, made.deps)
    expect(transport.kind).toBe('soundcloud')
    const frame = made.mount.querySelector('iframe')
    expect(frame?.getAttribute('src')).toBe(`https://w.soundcloud.com/player/?url=${encodeURIComponent(track)}`)
    transport.destroy()
  })

  it('removes what it added to the page when destroyed', async () => {
    const made = deps(await client())
    createTransport({ kind: 'youtube', audio_url: null, embed_id: 'x'.repeat(11) }, made.deps).destroy()
    createTransport({ kind: 'soundcloud', audio_url: null, embed_id: 'https://soundcloud.com/a/b' }, made.deps).destroy()
    expect(made.mount.children).toHaveLength(0)
  })

  it('a chart whose audio the server no longer has fails with "unavailable"', async () => {
    const c = await client()
    server.use(http.post(`${API}/charts/:id/audio-ticket`, () => apiError(404, 'not_found')))
    const made = deps(c)
    const transport = createTransport({ kind: 'file', audio_url: '/x', embed_id: null }, made.deps)
    await expect(transport.load()).rejects.toMatchObject({ code: 'unavailable' })
  })
})
