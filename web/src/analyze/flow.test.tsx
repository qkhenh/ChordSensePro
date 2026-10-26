/** The whole user journey of UC1, from choosing a file to the chart opening. */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { Chart, JobProgress } from '../api/types'
import { buildRoutes } from '../routes'
import { FakeSocket } from '../test/fakeSocket'
import { apiError, authResponse, http, HttpResponse, quotaOut, server } from '../test/server'
import { readActiveJob } from './activeJob'

const API = '*/api/v1'
const JOB = '5ae8d1b8-ce75-5e37-b3fd-0e987715e3f1'
const chart = Object.values(
  import.meta.glob<Chart>('../../../contracts/fixtures/charts/ii_v_i_c.chart.json', {
    eager: true,
    import: 'default',
  }),
)[0]
const finished: JobProgress = {
  job_id: JOB,
  state: 'done',
  stage: 'recognizing',
  progress: 1,
  error: null,
  result_chart_id: chart.id,
}

function renderWorkspace() {
  server.use(
    http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())),
    http.get(`${API}/charts/${chart.id}`, () => HttpResponse.json(chart)),
  )
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: ['/workspace'] })
  render(<RouterProvider router={router} />)
  return { router, user: userEvent.setup({ applyAccept: false }) }
}

const song = (name = 'Song.mp3') => new File([new Uint8Array(64)], name, { type: 'audio/mpeg' })

async function pickAndAnalyze(user: ReturnType<typeof userEvent.setup>, file = song()) {
  await user.upload(await screen.findByLabelText(/chọn tệp/i), file)
  await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
}

function countQuotaCalls(replies: (() => ReturnType<typeof quotaOut>)[] = []) {
  const calls = { n: 0 }
  server.use(
    http.get(`${API}/users/me/quota`, () => {
      const reply = replies[Math.min(calls.n, replies.length - 1)]
      calls.n += 1
      return HttpResponse.json(reply ? reply() : quotaOut())
    }),
  )
  return calls
}

beforeEach(() => {
  FakeSocket.reset()
  FakeSocket.failToConstruct = true // follow jobs by polling
  vi.stubGlobal('WebSocket', FakeSocket)
})
afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('success', () => {
  it('uploads, follows the job and opens the finished chart', async () => {
    server.use(
      http.post(`${API}/analyze/file`, () => HttpResponse.json({ job_id: JOB }, { status: 202 })),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(finished)),
    )
    const { router, user } = renderWorkspace()
    await pickAndAnalyze(user)

    await waitFor(() => expect(router.state.location.pathname).toBe(`/charts/${chart.id}`))
    expect(await screen.findByRole('heading', { level: 2, name: chart.title })).toBeInTheDocument()
    expect(readActiveJob()).toBeNull() // nothing left to resume
  })

  it('refreshes the quota badge after the job is accepted and finished', async () => {
    const calls = countQuotaCalls([
      () => quotaOut({ used: 2, remaining: 3 }),
      () => quotaOut({ used: 3, remaining: 2 }),
    ])
    server.use(
      http.post(`${API}/analyze/file`, () => HttpResponse.json({ job_id: JOB }, { status: 202 })),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(finished)),
    )
    const { user } = renderWorkspace()
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('còn 3/5'))
    await pickAndAnalyze(user)
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('còn 2/5'))
    expect(calls.n).toBeGreaterThanOrEqual(2)
  })
})

describe('refusals', () => {
  it('shows the daily limit and the reset time on a 429, with no job started (EF2)', async () => {
    countQuotaCalls([() => quotaOut({ used: 4, remaining: 1 }), () => quotaOut({ used: 5, remaining: 0 })])
    server.use(
      http.post(`${API}/analyze/file`, () =>
        HttpResponse.json(
          { code: 'quota_exceeded', message: 'used', limit: 5, resets_at: '2026-10-27T00:00:00Z' },
          { status: 429 },
        ),
      ),
    )
    const { user } = renderWorkspace()
    await pickAndAnalyze(user)

    const banner = await screen.findByRole('alert')
    expect(banner).toHaveTextContent('Bạn đã dùng hết 5 lượt phân tích hôm nay')
    expect(screen.getByRole('link', { name: 'Nâng cấp Pro' })).toBeInTheDocument()
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(readActiveJob()).toBeNull()
    // the badge catches up with reality
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('hết lượt'))
  })

  it.each([
    [413, 'file_too_large', /lớn hơn 100 MB/],
    [415, 'unsupported_format', /không phải âm thanh hợp lệ/i],
    [413, 'duration_too_long', /dài hơn 10 phút/],
    [422, 'ssrf_blocked', /an toàn/i],
    [422, 'url_unresolvable', /không lấy được âm thanh/i],
  ])('explains %i %s and leaves the quota alone', async (status, code, text) => {
    countQuotaCalls([() => quotaOut({ used: 2, remaining: 3 })])
    server.use(http.post(`${API}/analyze/file`, () => apiError(status, code)))
    const { user } = renderWorkspace()
    await pickAndAnalyze(user)

    expect(await screen.findByRole('alert')).toHaveTextContent(text)
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('còn 3/5'))
    expect(screen.getByRole('tab', { name: 'Tải lên' })).toBeInTheDocument() // can try something else
  })
})

describe('retry', () => {
  it('sends the same file again after a retryable failure, without asking for it again', async () => {
    const sizes: number[] = []
    const second = '9c2ed1b8-ce75-5e37-b3fd-0e987715e3f9'
    let posts = 0
    server.use(
      http.post(`${API}/analyze/file`, async ({ request }) => {
        posts += 1
        sizes.push(((await request.formData()).get('file') as File).size)
        return HttpResponse.json({ job_id: posts === 1 ? JOB : second }, { status: 202 })
      }),
      http.get(`${API}/analyze/status/${JOB}`, () =>
        HttpResponse.json({
          ...finished,
          state: 'failed',
          stage: 'recognizing',
          progress: 0.9,
          result_chart_id: null,
          error: { code: 'ai_failure', message: 'x', retryable: true },
        }),
      ),
      http.get(`${API}/analyze/status/${second}`, () =>
        HttpResponse.json({ ...finished, job_id: second }),
      ),
    )
    const { router, user } = renderWorkspace()
    await pickAndAnalyze(user)
    await user.click(await screen.findByRole('button', { name: 'Thử lại' }))

    await waitFor(() => expect(router.state.location.pathname).toBe(`/charts/${chart.id}`))
    expect(posts).toBe(2)
    expect(sizes).toEqual([64, 64])
  })
})


describe('retry safety', () => {
  const failedJob: JobProgress = {
    ...finished,
    state: 'failed',
    stage: 'recognizing',
    progress: 0.9,
    result_chart_id: null,
    error: { code: 'ai_failure', message: 'x', retryable: true },
  }

  it('a double click on "Thử lại" sends the file once, not twice', async () => {
    let posts = 0
    server.use(
      http.post(`${API}/analyze/file`, async () => {
        posts += 1
        await new Promise((r) => setTimeout(r, 40))
        return HttpResponse.json({ job_id: JOB }, { status: 202 })
      }),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(failedJob)),
    )
    const { user } = renderWorkspace()
    await pickAndAnalyze(user)
    await user.dblClick(await screen.findByRole('button', { name: 'Thử lại' }))
    await new Promise((r) => setTimeout(r, 150))
    expect(posts).toBe(2) // the first attempt plus ONE retry
  })

  it('shows why a retry could not even be sent', async () => {
    let posts = 0
    server.use(
      http.post(`${API}/analyze/file`, () => {
        posts += 1
        return posts === 1
          ? HttpResponse.json({ job_id: JOB }, { status: 202 })
          : apiError(413, 'file_too_large')
      }),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(failedJob)),
    )
    const { user } = renderWorkspace()
    await pickAndAnalyze(user)
    await user.click(await screen.findByRole('button', { name: 'Thử lại' }))
    await waitFor(() => expect(posts).toBe(2))
    const alerts = await screen.findAllByRole('alert')
    expect(alerts.some((a) => /lớn hơn 100 MB/.test(a.textContent ?? ''))).toBe(true)
    expect(screen.getByRole('button', { name: 'Thử lại' })).toBeEnabled() // and it can be tried again
  })

  it('shows the quota banner when a retry is refused for quota', async () => {
    let posts = 0
    server.use(
      http.post(`${API}/analyze/file`, () => {
        posts += 1
        return posts === 1
          ? HttpResponse.json({ job_id: JOB }, { status: 202 })
          : HttpResponse.json(
              { code: 'quota_exceeded', message: 'x', limit: 5, resets_at: '2026-10-27T00:00:00Z' },
              { status: 429 },
            )
      }),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(failedJob)),
    )
    const { user } = renderWorkspace()
    await pickAndAnalyze(user)
    await user.click(await screen.findByRole('button', { name: 'Thử lại' }))
    expect(await screen.findByText('Bạn đã dùng hết 5 lượt phân tích hôm nay')).toBeInTheDocument()
  })
})

describe('demo song', () => {
  it('opens its chart and does not spend the quota', async () => {
    const calls = countQuotaCalls([() => quotaOut({ used: 5, remaining: 0 })])
    server.use(
      http.get(`${API}/demo-tracks`, () =>
        HttpResponse.json([
          { id: 'ii_v_i_c', title: 'ii–V–I in C', artist: 'ChordSense demo', duration_ms: 9000, tempo_bpm: 120, key: 'C major' },
        ]),
      ),
      http.post(`${API}/analyze/demo/ii_v_i_c`, () => HttpResponse.json({ job_id: JOB }, { status: 202 })),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(finished)),
    )
    const { router, user } = renderWorkspace()
    await waitFor(() => expect(screen.getByTestId('quota-badge')).toHaveTextContent('hết lượt'))
    await user.click(await screen.findByRole('button', { name: /ii–V–I in C/ }))

    await waitFor(() => expect(router.state.location.pathname).toBe(`/charts/${chart.id}`))
    expect(calls.n).toBeGreaterThanOrEqual(1)
    expect(screen.getByTestId('quota-badge')).toHaveTextContent('hết lượt') // still the same allowance
  })
})
