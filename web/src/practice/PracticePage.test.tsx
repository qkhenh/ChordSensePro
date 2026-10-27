import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { Chart } from '../api/types'
import { formatChord } from '../format/chord'
import { applyOverrides } from '../grid/effective'
import { buildGrid } from '../grid/grid'
import { buildRoutes } from '../routes'
import { createFakeYouTube } from '../test/fakeEmbeds'
import { FakeAudio } from '../test/fakeMedia'
import { resetAudioContext } from './audio'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'

const youtube = vi.hoisted(() => ({ api: null as unknown }))
vi.mock('../transport/loaders', async (original) => {
  const real = await original<typeof import('../transport/loaders')>()
  return { ...real, loadYouTubeApi: async () => youtube.api }
})

const API = '*/api/v1'
const fixtures = import.meta.glob<Chart>('../../../contracts/fixtures/charts/*.chart.json', {
  eager: true,
  import: 'default',
})
const byName = (name: string) => Object.entries(fixtures).find(([path]) => path.endsWith(name))![1]
const plain = byName('/ii_v_i_c.chart.json')
const edited = byName('/autumn_leaves_gm.edited.chart.json')

let audios: FakeAudio[] = []
let tickets = 0
let resume: ReturnType<typeof vi.fn>

beforeEach(() => {
  audios = []
  tickets = 0
  resume = vi.fn(async () => {})
  vi.stubGlobal(
    'Audio',
    class extends FakeAudio {
      constructor() {
        super()
        audios.push(this)
      }
    },
  )
  vi.stubGlobal(
    'AudioContext',
    class {
      state = 'suspended'
      resume = resume
      close = vi.fn(async () => {})
    },
  )
  server.use(
    http.post(`${API}/charts/:id/audio-ticket`, ({ params }) => {
      tickets += 1
      return HttpResponse.json({
        url: `/api/v1/charts/${params.id}/audio?ticket=t${tickets}`,
        expires_at: '2099-01-01T00:00:00Z',
      })
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
  resetAudioContext()
})

function renderChart(id: string) {
  server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: [`/charts/${id}`] })
  const view = render(<RouterProvider router={router} />)
  return { router, ...view }
}

const serve = (chart: Chart) => server.use(http.get(`${API}/charts/${chart.id}`, () => HttpResponse.json(chart)))
const bars = (chart: Chart) => buildGrid(chart.metadata, applyOverrides(chart.ai_events, chart.overrides)).bars

async function loadedAudio(durationMs = 9000) {
  await waitFor(() => expect(audios[0]?.src).toBeTruthy())
  act(() => audios[0].finishLoading(durationMs))
  return audios[0]
}

describe('practice page', () => {
  it('shows the song, its key and tempo and one cell per bar of the grid', async () => {
    serve(plain)
    renderChart(plain.id)
    expect(await screen.findByRole('heading', { level: 2, name: plain.title })).toBeInTheDocument()
    expect(screen.getByText(/C major/)).toBeInTheDocument()
    expect(screen.getByText(/120/)).toBeInTheDocument()
    const grid = await screen.findByRole('group', { name: 'Lưới hợp âm' })
    expect(within(grid).getAllByRole('button')).toHaveLength(bars(plain).length)
    expect(within(grid).getByRole('button', { name: /Ô nhịp 1:/ })).toBeInTheDocument()
  })

  it('shows the chords as the person edited them, not the AI baseline', async () => {
    serve(edited)
    renderChart(edited.id)
    const grid = await screen.findByRole('group', { name: 'Lưới hợp âm' })
    const override = edited.overrides.find((o) => o.op === 'set')!.event!
    const deleted = edited.overrides.find((o) => o.op === 'delete')!.event_id
    const text = within(grid).getAllByRole('button').map((b) => b.getAttribute('aria-label')).join('|')
    expect(text).toContain(formatChord(override, 'precise'))
    const effective = applyOverrides(edited.ai_events, edited.overrides)
    expect(effective.some((e) => e.event_id === deleted)).toBe(false)
  })

  it('says when the chart does not exist and leads back to the library', async () => {
    server.use(http.get(`${API}/charts/nope`, () => apiError(404, 'not_found')))
    renderChart('nope')
    expect(await screen.findByText('Không tìm thấy bài này')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Về Thư viện' })).toHaveAttribute('href', '/library')
  })

  it('reports a server problem without crashing', async () => {
    server.use(http.get(`${API}/charts/boom`, () => apiError(500, 'internal_error')))
    renderChart('boom')
    expect(await screen.findByRole('alert')).toHaveTextContent(/không tải được/i)
  })

  it('asks the person to enable sound when the browser blocks autoplay, then plays on that click', async () => {
    serve(plain)
    renderChart(plain.id)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    const audio = await loadedAudio()
    audio.blockAutoplay = true
    await userEvent.click(await screen.findByRole('button', { name: 'Phát' }))
    const dialog = await screen.findByRole('dialog')
    audio.blockAutoplay = false
    await userEvent.click(within(dialog).getByRole('button', { name: 'Bật âm thanh' }))
    expect(resume).toHaveBeenCalled()
    await waitFor(() => expect(audio.paused).toBe(false))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('shows the video frame for a YouTube chart without asking for an audio ticket', async () => {
    const fake = createFakeYouTube()
    youtube.api = fake.api
    const chart: Chart = {
      ...plain,
      id: '11111111-aaaa-4aaa-8aaa-111111111111',
      source: { ...plain.source, kind: 'url', playback: { kind: 'youtube', embed_id: 'abc123', audio_url: null } },
    }
    serve(chart)
    renderChart(chart.id)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    await waitFor(() => expect(fake.created).toHaveLength(1))
    expect(fake.created[0].videoId).toBe('abc123')
    expect(screen.getByTestId('media-frame')).toBeVisible()
    expect(tickets).toBe(0)
  })

  it('offers an upload instead when the video cannot be embedded', async () => {
    const fake = createFakeYouTube()
    youtube.api = fake.api
    const chart: Chart = {
      ...plain,
      id: '22222222-aaaa-4aaa-8aaa-222222222222',
      source: { ...plain.source, kind: 'url', playback: { kind: 'youtube', embed_id: 'blocked', audio_url: null } },
    }
    serve(chart)
    renderChart(chart.id)
    await waitFor(() => expect(fake.created).toHaveLength(1))
    act(() => fake.driver.finishLoading(9000))
    act(() => fake.driver.breakDown())
    expect(await screen.findByText(/video không khả dụng/i)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Tải tệp lên thay thế' })).toHaveAttribute('href', '/workspace')
  })

  it('stops playback and the animation loop when the page is left', async () => {
    serve(plain)
    const { router } = renderChart(plain.id)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    const audio = await loadedAudio()
    await userEvent.click(await screen.findByRole('button', { name: 'Phát' }))
    await waitFor(() => expect(audio.paused).toBe(false))
    const frames = vi.spyOn(window, 'requestAnimationFrame')
    await act(async () => {
      await router.navigate('/library')
    })
    server.use(http.get(`${API}/charts`, () => HttpResponse.json({ items: [], next_cursor: null })))
    await screen.findByRole('heading', { level: 2, name: 'Thư viện' })
    frames.mockClear()
    await new Promise((r) => setTimeout(r, 80))
    expect(frames).not.toHaveBeenCalled()
    expect(audio.src).toBe('')
  })

  it('does not keep the previous chart grid when another chart is opened', async () => {
    const other: Chart = { ...edited, id: '33333333-aaaa-4aaa-8aaa-333333333333' }
    serve(plain)
    serve(other)
    const { router } = renderChart(plain.id)
    const first = await screen.findByRole('group', { name: 'Lưới hợp âm' })
    expect(within(first).getAllByRole('button')).toHaveLength(bars(plain).length)
    await act(async () => {
      await router.navigate(`/charts/${other.id}`)
    })
    await waitFor(() => expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(other.title))
    const second = screen.getByRole('group', { name: 'Lưới hợp âm' })
    expect(within(second).getAllByRole('button')).toHaveLength(bars(other).length)
  })
})
