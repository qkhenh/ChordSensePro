import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { ApiClientContext } from '../api/context'
import { createApiClient, memoryStorage } from '../api/http'
import type { PlaybackSource } from '../api/types'
import { createFakeYouTube } from '../test/fakeEmbeds'
import { authResponse, http, HttpResponse, server } from '../test/server'
import { useTransport } from './useTransport'

const API = '*/api/v1'

async function setup(playback: PlaybackSource) {
  server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage() })
  await client.login('ann@example.com', 'secret-pass')
  const fake = createFakeYouTube()
  const mount = document.createElement('div')
  document.body.appendChild(mount)
  const wrapper = ({ children }: { children: ReactNode }) => (
    <ApiClientContext.Provider value={client}>{children}</ApiClientContext.Provider>
  )
  const hook = renderHook(
    ({ source, el }: { source: PlaybackSource; el: HTMLElement | null }) =>
      useTransport({ id: 'chart-1', playback: source }, el, { youtube: async () => fake.api }),
    { wrapper, initialProps: { source: playback, el: mount as HTMLElement | null } },
  )
  return { ...hook, fake, mount }
}

const youtube = (id: string): PlaybackSource => ({ kind: 'youtube', audio_url: null, embed_id: id })

describe('useTransport', () => {
  it('creates and loads the transport once there is somewhere to put it', async () => {
    const { result, fake, mount } = await setup(youtube('aaaaaaaaaaa'))
    await waitFor(() => expect(result.current.transport?.kind).toBe('youtube'))
    expect(mount.children).toHaveLength(1)
    await waitFor(() => expect(fake.created).toHaveLength(1))
    act(() => fake.driver.finishLoading(60_000))
    await waitFor(() => expect(result.current.state).toBe('ready'))
  })

  it('waits while the mount point does not exist yet', async () => {
    const { result, rerender, fake, mount } = await setup(youtube('aaaaaaaaaaa'))
    rerender({ source: youtube('aaaaaaaaaaa'), el: null })
    await waitFor(() => expect(result.current.transport).toBeNull())
    rerender({ source: youtube('aaaaaaaaaaa'), el: mount })
    await waitFor(() => expect(result.current.transport).not.toBeNull())
    expect(fake.created.length).toBeGreaterThan(0)
  })

  it('exposes the state the transport reports', async () => {
    const { result, fake } = await setup(youtube('aaaaaaaaaaa'))
    await waitFor(() => expect(fake.created).toHaveLength(1))
    act(() => fake.driver.finishLoading(60_000))
    await waitFor(() => expect(result.current.state).toBe('ready'))
    act(() => fake.driver.emitState(1))
    expect(result.current.state).toBe('playing')
  })

  it('exposes a failure as an error the page can explain', async () => {
    const { result, fake } = await setup(youtube('aaaaaaaaaaa'))
    await waitFor(() => expect(fake.created).toHaveLength(1))
    act(() => fake.driver.finishLoading(60_000))
    act(() => fake.driver.emitError(150))
    await waitFor(() => expect(result.current.error?.code).toBe('embed_blocked'))
    expect(result.current.state).toBe('error')
  })

  it('tears everything down on unmount', async () => {
    const { unmount, fake, mount } = await setup(youtube('aaaaaaaaaaa'))
    await waitFor(() => expect(fake.created).toHaveLength(1))
    unmount()
    expect(fake.state.destroyed).toBe(true)
    expect(mount.children).toHaveLength(0)
  })

  it('starts afresh for a different chart, destroying the old player', async () => {
    const { result, rerender, fake, mount } = await setup(youtube('aaaaaaaaaaa'))
    await waitFor(() => expect(fake.created).toHaveLength(1))
    rerender({ source: youtube('bbbbbbbbbbb'), el: mount })
    await waitFor(() => expect(fake.created).toHaveLength(2))
    expect(fake.created[1].videoId).toBe('bbbbbbbbbbb')
    expect(fake.state.destroyed).toBe(true)
    expect(result.current.error).toBeNull()
    expect(mount.children).toHaveLength(1)
  })
})
