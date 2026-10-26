import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { JobProgress } from '../api/types'
import { readActiveJob, writeActiveJob } from '../analyze/activeJob'
import { buildRoutes } from '../routes'
import { FakeSocket } from '../test/fakeSocket'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'

const API = '*/api/v1'
const JOB = '5ae8d1b8-ce75-5e37-b3fd-0e987715e3f1'

const running: JobProgress = {
  job_id: JOB,
  state: 'processing',
  stage: 'separating',
  progress: 0.45,
  error: null,
  result_chart_id: null,
}

function renderWorkspace() {
  server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: ['/workspace'] })
  render(<RouterProvider router={router} />)
  return { router, user: userEvent.setup({ applyAccept: false }) }
}

beforeEach(() => {
  FakeSocket.reset()
  FakeSocket.failToConstruct = true // these tests follow jobs by polling; useJob.test covers the socket
  vi.stubGlobal('WebSocket', FakeSocket)
})
afterEach(() => {
  vi.unstubAllGlobals()
  sessionStorage.clear()
})

describe('starting an analysis', () => {
  it('offers the source selector when nothing is running', async () => {
    renderWorkspace()
    expect(await screen.findByRole('tab', { name: 'Tải lên' })).toBeInTheDocument()
  })

  it('uploads a file, remembers the job and starts tracking it', async () => {
    let form: FormData | undefined
    server.use(
      http.post(`${API}/analyze/file`, async ({ request }) => {
        form = await request.formData()
        return HttpResponse.json({ job_id: JOB }, { status: 202 })
      }),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(running)),
    )
    const { user } = renderWorkspace()
    await user.upload(
      await screen.findByLabelText(/chọn tệp/i),
      new File([new Uint8Array(64)], 'Song.mp3', { type: 'audio/mpeg' }),
    )
    await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))

    expect(await screen.findByRole('progressbar')).toBeInTheDocument()
    const sent = form?.get('file') as File | null | undefined
    expect(sent?.size).toBe(64)
    expect(readActiveJob()).toBe(JOB)
  })

  it('submits a pasted link', async () => {
    let body: unknown
    server.use(
      http.post(`${API}/analyze/url`, async ({ request }) => {
        body = await request.json()
        return HttpResponse.json({ job_id: JOB }, { status: 202 })
      }),
      http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(running)),
    )
    const { user } = renderWorkspace()
    await user.click(await screen.findByRole('tab', { name: 'Liên kết' }))
    await user.type(screen.getByLabelText(/dán liên kết/i), 'https://youtu.be/dQw4w9WgXcQ{Enter}')
    expect(await screen.findByRole('progressbar')).toBeInTheDocument()
    expect(body).toEqual({ url: 'https://youtu.be/dQw4w9WgXcQ' })
  })
})

describe('reloading the page during a job (F5)', () => {
  it('goes on watching the job that was running', async () => {
    writeActiveJob(JOB)
    server.use(http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(running)))
    renderWorkspace()
    expect(await screen.findByRole('progressbar')).toHaveAttribute('aria-valuenow', '45')
    expect(screen.queryByRole('tab', { name: 'Tải lên' })).not.toBeInTheDocument()
  })

  it('forgets a job that no longer exists and offers the selector again', async () => {
    writeActiveJob(JOB)
    server.use(http.get(`${API}/analyze/status/${JOB}`, () => apiError(404, 'not_found')))
    renderWorkspace()
    expect(await screen.findByRole('tab', { name: 'Tải lên' })).toBeInTheDocument()
    await waitFor(() => expect(readActiveJob()).toBeNull())
  })

  it('lets the user stop watching and start something else', async () => {
    writeActiveJob(JOB)
    server.use(http.get(`${API}/analyze/status/${JOB}`, () => HttpResponse.json(running)))
    const { user } = renderWorkspace()
    await user.click(await screen.findByRole('button', { name: 'Dừng theo dõi' }))
    expect(await screen.findByRole('tab', { name: 'Tải lên' })).toBeInTheDocument()
    expect(readActiveJob()).toBeNull()
  })
})

describe('failures', () => {
  it('shows a failed job with its reason and lets the user close it', async () => {
    writeActiveJob(JOB)
    server.use(
      http.get(`${API}/analyze/status/${JOB}`, () =>
        HttpResponse.json({
          ...running,
          state: 'failed',
          error: { code: 'analysis_timeout', message: 'slow', retryable: true },
        }),
      ),
    )
    const { user } = renderWorkspace()
    expect(await screen.findByRole('alert')).toHaveTextContent(/quá nhiều thời gian/i)
    await user.click(within(screen.getByRole('alert').parentElement as HTMLElement).getByRole('button', { name: 'Đóng' }))
    expect(await screen.findByRole('tab', { name: 'Tải lên' })).toBeInTheDocument()
  })
})
