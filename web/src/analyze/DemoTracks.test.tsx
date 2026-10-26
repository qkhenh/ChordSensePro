import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { ApiClientContext } from '../api/context'
import { createApiClient, memoryStorage } from '../api/http'
import type { DemoTrack } from '../api/types'
import { ToastProvider } from '../shell/toast'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'
import { DemoTracks } from './DemoTracks'

const API = '*/api/v1'
const JOB = '5ae8d1b8-ce75-5e37-b3fd-0e987715e3f1'
const tracks: DemoTrack[] = [
  { id: 'ii_v_i_c', title: 'ii–V–I in C', artist: 'ChordSense demo', duration_ms: 9000, tempo_bpm: 120, key: 'C major' },
  { id: 'blues_f', title: '12-bar blues in F', artist: 'ChordSense demo', duration_ms: 30150, tempo_bpm: 100, key: 'F major' },
  { id: 'waltz_g_3_4', title: 'Waltz in G (3/4)', artist: 'ChordSense demo', duration_ms: 17000, tempo_bpm: 90, key: 'G major' },
]

async function setup(onStart = vi.fn()) {
  server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage() })
  await client.login('ann@example.com', 'secret-pass')
  const wrapper = ({ children }: { children: ReactNode }) => (
    <ApiClientContext.Provider value={client}>
      <ToastProvider>{children}</ToastProvider>
    </ApiClientContext.Provider>
  )
  // The component fetches through TanStack Query, so it needs a client of its own.
  const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
  const queries = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queries}>
      <DemoTracks onStart={onStart} />
    </QueryClientProvider>,
    { wrapper },
  )
  return { onStart, user: userEvent.setup() }
}

describe('DemoTracks', () => {
  it('shows placeholders while the catalog loads, then every song', async () => {
    server.use(
      http.get(`${API}/demo-tracks`, async () => {
        await new Promise((r) => setTimeout(r, 30))
        return HttpResponse.json(tracks)
      }),
    )
    await setup()
    expect(screen.getByTestId('demo-skeleton')).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /ii–V–I in C/ })).toBeInTheDocument()
    expect(screen.getAllByRole('button')).toHaveLength(3)
    expect(screen.queryByTestId('demo-skeleton')).not.toBeInTheDocument()
  })

  it('describes each song by key and tempo', async () => {
    server.use(http.get(`${API}/demo-tracks`, () => HttpResponse.json(tracks)))
    await setup()
    const first = await screen.findByRole('button', { name: /ii–V–I in C/ })
    expect(first).toHaveTextContent('C major')
    expect(first).toHaveTextContent('120 BPM')
  })

  it('starts the chosen song and hands the job to the page', async () => {
    let asked = ''
    server.use(
      http.get(`${API}/demo-tracks`, () => HttpResponse.json(tracks)),
      http.post(`${API}/analyze/demo/:id`, ({ params }) => {
        asked = String(params.id)
        return HttpResponse.json({ job_id: JOB }, { status: 202 })
      }),
    )
    const { user, onStart } = await setup()
    await user.click(await screen.findByRole('button', { name: /12-bar blues/ }))
    await waitFor(() => expect(onStart).toHaveBeenCalledWith(JOB))
    expect(asked).toBe('blues_f')
  })

  it('does not depend on the daily quota: nothing here is disabled by it', async () => {
    server.use(http.get(`${API}/demo-tracks`, () => HttpResponse.json(tracks)))
    await setup()
    await screen.findAllByRole('button')
    for (const button of screen.getAllByRole('button')) expect(button).toBeEnabled()
  })

  it('locks the list while a song is being started', async () => {
    server.use(
      http.get(`${API}/demo-tracks`, () => HttpResponse.json(tracks)),
      http.post(`${API}/analyze/demo/:id`, async () => {
        await new Promise((r) => setTimeout(r, 60))
        return HttpResponse.json({ job_id: JOB }, { status: 202 })
      }),
    )
    const { user } = await setup()
    await user.click(await screen.findByRole('button', { name: /ii–V–I in C/ }))
    for (const button of screen.getAllByRole('button')) expect(button).toBeDisabled()
  })

  it('tells the user when the catalog cannot be loaded', async () => {
    server.use(http.get(`${API}/demo-tracks`, () => apiError(500, 'internal_error')))
    await setup()
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(screen.queryByTestId('demo-skeleton')).not.toBeInTheDocument()
  })

  it('explains a demo whose audio has not been generated', async () => {
    server.use(
      http.get(`${API}/demo-tracks`, () => HttpResponse.json(tracks)),
      http.post(`${API}/analyze/demo/:id`, () => apiError(503, 'demo_unavailable')),
    )
    const { user, onStart } = await setup()
    await user.click(await screen.findByRole('button', { name: /ii–V–I in C/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/demo/i)
    expect(onStart).not.toHaveBeenCalled()
  })
})
