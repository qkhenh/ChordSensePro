import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { Chart, Role } from '../api/types'
import { formatChord } from '../format/chord'
import { applyOverrides } from '../grid/effective'
import { buildGrid } from '../grid/grid'
import { buildRoutes } from '../routes'
import { createFakeYouTube } from '../test/fakeEmbeds'
import { FakeAudio } from '../test/fakeMedia'
import { resetAudioContext } from './audio'
import { apiError, authResponse, http, HttpResponse, server, TEST_USER } from '../test/server'

const youtube = vi.hoisted(() => ({ api: null as unknown, fail: false }))
vi.mock('../transport/loaders', async (original) => {
  const real = await original<typeof import('../transport/loaders')>()
  return {
    ...real,
    loadYouTubeApi: async () => {
      if (youtube.fail) throw new Error('script blocked')
      return youtube.api
    },
  }
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
  youtube.fail = false
  vi.unstubAllGlobals()
  resetAudioContext()
})

function renderChart(id: string, role: Role = 'free') {
  server.use(
    http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse({ user: { ...TEST_USER, role } }))),
  )
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

describe('practice controls', () => {
  const grid = buildGrid(edited.metadata, applyOverrides(edited.ai_events, edited.overrides))
  const open = async (role: Role = 'free') => {
    serve(edited)
    renderChart(edited.id, role)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    return loadedAudio(edited.metadata.duration_ms)
  }
  const cell = (n: number) => screen.getByRole('button', { name: new RegExp(`^Ô nhịp ${n}(:|$)`) })

  it('jumps to the start of a bar that is clicked', async () => {
    const audio = await open()
    await userEvent.click(cell(9))
    expect(audio.currentTime * 1000).toBeCloseTo(grid.barStartMs(8), 0)
  })

  it('plays and pauses with Space', async () => {
    const audio = await open()
    await userEvent.keyboard(' ')
    await waitFor(() => expect(audio.paused).toBe(false))
    await userEvent.keyboard(' ')
    await waitFor(() => expect(audio.paused).toBe(true))
  })

  it('changes the speed and keeps the pitch', async () => {
    const audio = await open('paid')
    await userEvent.click(screen.getByRole('button', { name: '0.75×' }))
    expect(audio.playbackRate).toBe(0.75)
    expect(audio.preservesPitch).toBe(true)
    expect(screen.getByRole('button', { name: '0.75×' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('loops a dragged selection: past its end the playhead returns to its start', async () => {
    const audio = await open()
    const user = userEvent.setup()
    await user.pointer([{ keys: '[MouseLeft>]', target: cell(3) }, { target: cell(5) }, { keys: '[/MouseLeft]', target: cell(5) }])
    expect(await screen.findByText(/ô nhịp 3–5/i)).toBeInTheDocument()
    const start = grid.barStartMs(2)
    const end = grid.barEndMs(4)
    audio.currentTime = (end - 30) / 1000
    await new Promise((r) => setTimeout(r, 60)) // a few frames see the playhead inside the loop
    audio.currentTime = (end + 10) / 1000
    await waitFor(() => expect(audio.currentTime * 1000).toBeCloseTo(start, 0))
  })

  it('leaves the loop and plays on when it is cleared', async () => {
    const audio = await open()
    const user = userEvent.setup()
    await user.pointer([{ keys: '[MouseLeft>]', target: cell(3) }, { target: cell(5) }, { keys: '[/MouseLeft]', target: cell(5) }])
    await user.click(await screen.findByRole('button', { name: 'Xoá loop' }))
    expect(screen.queryByText(/ô nhịp 3–5/i)).not.toBeInTheDocument()
    const end = grid.barEndMs(4)
    audio.currentTime = (end - 30) / 1000
    await new Promise((r) => setTimeout(r, 60))
    audio.currentTime = (end + 10) / 1000
    await new Promise((r) => setTimeout(r, 60))
    expect(audio.currentTime * 1000).toBeCloseTo(end + 10, 0)
  })
})

describe('chord style and metronome', () => {
  it('shows simpler names in Basic and makes no request to do it', async () => {
    serve(edited)
    renderChart(edited.id)
    const grid = await screen.findByRole('group', { name: 'Lưới hợp âm' })
    const override = edited.overrides.find((o) => o.op === 'set')!.event!
    const names = () => within(grid).getAllByRole('button').map((b) => b.getAttribute('aria-label')).join('|')
    expect(names()).toContain(formatChord(override, 'precise'))
    const requests: string[] = []
    const record = ({ request }: { request: Request }) => void requests.push(request.url)
    server.events.on('request:start', record)
    await userEvent.click(screen.getByRole('button', { name: 'Cơ bản' }))
    server.events.removeListener('request:start', record)
    expect(names()).toContain(formatChord(override, 'basic'))
    expect(names()).not.toContain(formatChord(override, 'precise'))
    expect(requests).toEqual([])
  })

  it('offers the metronome panel next to the player', async () => {
    serve(plain)
    renderChart(plain.id)
    expect(await screen.findByRole('switch', { name: 'Metronome' })).not.toBeChecked()
  })
})

describe('plans', () => {
  const open = async (role: Role) => {
    serve(edited)
    renderChart(edited.id, role)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    return loadedAudio(edited.metadata.duration_ms)
  }
  const upgrade = () => screen.findByRole('link', { name: 'Nâng cấp lên Pro' })

  it('keeps speed behind Pro for a free user and says so', async () => {
    const audio = await open('free')
    await userEvent.click(screen.getByRole('button', { name: '0.75×' }))
    expect(await upgrade()).toHaveAttribute('href', '#upgrade')
    expect(audio.playbackRate).toBe(1)
    expect(screen.getByRole('button', { name: '1×' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('keeps the song-synced metronome and the count-in behind Pro, but not the manual one', async () => {
    await open('free')
    await userEvent.click(screen.getByRole('radio', { name: 'Đồng bộ bài' }))
    expect(await upgrade()).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Thủ công' })).toBeChecked()
    await userEvent.click(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' }))
    expect(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' })).not.toBeChecked()
    await userEvent.click(screen.getByRole('switch', { name: 'Metronome' }))
    expect(screen.getByRole('switch', { name: 'Metronome' })).toBeChecked()
  })

  it('still lets a free user loop a section at normal speed', async () => {
    await open('free')
    const user = userEvent.setup()
    const cell = (n: number) => screen.getByRole('button', { name: new RegExp(`^Ô nhịp ${n}(:|$)`) })
    await user.pointer([{ keys: '[MouseLeft>]', target: cell(3) }, { target: cell(5) }, { keys: '[/MouseLeft]', target: cell(5) }])
    expect(await screen.findByText(/ô nhịp 3–5/i)).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Nâng cấp lên Pro' })).not.toBeInTheDocument()
  })

  it.each<Role>(['paid', 'teacher', 'student'])('opens everything for %s, and writes nothing to the server', async (role) => {
    const writes: string[] = []
    const record = ({ request }: { request: Request }) => {
      if (request.method !== 'GET' && !request.url.includes('/audio-ticket') && !request.url.includes('/auth/')) {
        writes.push(`${request.method} ${request.url}`)
      }
    }
    server.events.on('request:start', record)
    const audio = await open(role)
    await userEvent.click(screen.getByRole('button', { name: '0.75×' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Đồng bộ bài' }))
    await userEvent.click(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' }))
    server.events.removeListener('request:start', record)
    expect(audio.playbackRate).toBe(0.75)
    expect(screen.getByRole('radio', { name: 'Đồng bộ bài' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Đếm nhịp vào bài' })).toBeChecked()
    expect(screen.queryByRole('link', { name: 'Nâng cấp lên Pro' })).not.toBeInTheDocument()
    expect(writes).toEqual([])
  })
})

describe('embedded players that cannot start', () => {
  it('suggests uploading a file when the YouTube player cannot be loaded at all', async () => {
    youtube.fail = true
    const chart: Chart = {
      ...plain,
      id: '44444444-aaaa-4aaa-8aaa-444444444444',
      source: { ...plain.source, kind: 'url', playback: { kind: 'youtube', embed_id: 'abc123', audio_url: null } },
    }
    serve(chart)
    renderChart(chart.id)
    expect(await screen.findByRole('link', { name: 'Tải tệp lên thay thế' })).toHaveAttribute('href', '/workspace')
  })

  it('shows a plain message, without the upload suggestion, when a stored file fails to play', async () => {
    serve(plain)
    renderChart(plain.id)
    await screen.findByRole('group', { name: 'Lưới hợp âm' })
    await waitFor(() => expect(audios[0]?.src).toBeTruthy())
    act(() => audios[0].fail()) // the first failure only asks for a fresh ticket
    await waitFor(() => expect(audios[0].src).toContain('ticket=t2'))
    act(() => audios[0].fail())
    expect(await screen.findByText(/không phát được bài này/i)).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Tải tệp lên thay thế' })).not.toBeInTheDocument()
  })
})
