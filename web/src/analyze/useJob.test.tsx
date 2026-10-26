import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiClientContext } from '../api/context'
import { createApiClient, memoryStorage } from '../api/http'
import type { JobProgress } from '../api/types'
import { FakeSocket } from '../test/fakeSocket'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'
import { POLL_INTERVAL_MS, useJob } from './useJob'

const API = '*/api/v1'
const JOB = '5ae8d1b8-ce75-5e37-b3fd-0e987715e3f1'
const FAST = 15

const progress = (over: Partial<JobProgress> = {}): JobProgress => ({
  job_id: JOB,
  state: 'processing',
  stage: 'normalizing',
  progress: 0.15,
  error: null,
  result_chart_id: null,
  ...over,
})
const done = () => progress({ state: 'done', stage: 'recognizing', progress: 1, result_chart_id: 'c1' })

async function signedInClient() {
  server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage() })
  await client.login('ann@example.com', 'secret-pass')
  return client
}

async function render(jobId: string | null, client?: Awaited<ReturnType<typeof signedInClient>>) {
  const api = client ?? (await signedInClient())
  const wrapper = ({ children }: { children: ReactNode }) => (
    <ApiClientContext.Provider value={api}>{children}</ApiClientContext.Provider>
  )
  return renderHook(({ id }: { id: string | null }) => useJob(id, { pollMs: FAST, connectMs: 40 }), {
    wrapper,
    initialProps: { id: jobId },
  })
}

function pollStatus(...replies: (JobProgress | 'network' | 404)[]) {
  const seen: number[] = []
  server.use(
    http.get(`${API}/analyze/status/${JOB}`, () => {
      seen.push(Date.now())
      const reply = replies[Math.min(seen.length, replies.length) - 1]
      if (reply === 'network') return HttpResponse.error()
      if (reply === 404) return apiError(404, 'not_found')
      return HttpResponse.json(reply)
    }),
  )
  return seen
}

beforeEach(() => {
  FakeSocket.reset()
  vi.stubGlobal('WebSocket', FakeSocket)
})
afterEach(() => vi.unstubAllGlobals())

describe('constants', () => {
  it('polls every two seconds by default (AF2)', () => {
    expect(POLL_INTERVAL_MS).toBe(2000)
  })
})

describe('idle', () => {
  it('does nothing without a job', async () => {
    const { result } = await render(null)
    expect(result.current).toMatchObject({ job: null, transport: 'idle', error: null })
    expect(FakeSocket.instances).toHaveLength(0)
  })
})

describe('over the WebSocket', () => {
  it('connects with the access token and follows the pushed progress', async () => {
    const { result } = await render(JOB)
    expect(FakeSocket.last.url).toContain(`/ws/jobs/${JOB}`)
    expect(FakeSocket.last.protocols).toEqual(['bearer', 'access-1'])

    act(() => FakeSocket.last.open())
    expect(result.current.transport).toBe('ws')
    act(() => FakeSocket.last.message(progress()))
    expect(result.current.job?.stage).toBe('normalizing')
    act(() => FakeSocket.last.message(progress({ stage: 'separating', progress: 0.45 })))
    expect(result.current.job?.progress).toBe(0.45)
  })

  it('ignores a stale update that arrives after a newer one', async () => {
    const { result } = await render(JOB)
    act(() => FakeSocket.last.open())
    act(() => FakeSocket.last.message(progress({ stage: 'separating', progress: 0.45 })))
    act(() => FakeSocket.last.message(progress({ stage: 'normalizing', progress: 0.15 })))
    expect(result.current.job?.progress).toBe(0.45)
  })

  it('stops listening once the job is final', async () => {
    const { result } = await render(JOB)
    act(() => FakeSocket.last.open())
    act(() => FakeSocket.last.message(done()))
    expect(result.current.job?.state).toBe('done')
    expect(FakeSocket.last.closed).toBe(true)
  })

  it('does not poll while the socket is healthy', async () => {
    const seen = pollStatus(progress())
    await render(JOB)
    act(() => FakeSocket.last.open())
    await new Promise((r) => setTimeout(r, FAST * 5))
    expect(seen).toHaveLength(0)
  })

  it('ignores malformed messages', async () => {
    const { result } = await render(JOB)
    act(() => FakeSocket.last.open())
    act(() => FakeSocket.last.message('not json'))
    act(() => FakeSocket.last.message(progress()))
    expect(result.current.job?.progress).toBe(0.15)
  })
})

describe('falling back to polling', () => {
  it('polls when the socket drops before the job finishes', async () => {
    const seen = pollStatus(progress({ stage: 'separating', progress: 0.45 }), progress({ stage: 'recognizing', progress: 0.9 }), done())
    const { result } = await render(JOB)
    act(() => FakeSocket.last.open())
    act(() => FakeSocket.last.drop(1006))

    await waitFor(() => expect(result.current.job?.state).toBe('done'))
    expect(result.current.transport).toBe('polling')
    expect(seen.length).toBeGreaterThanOrEqual(3)
  })

  it('polls when the socket never opens', async () => {
    pollStatus(done())
    const { result } = await render(JOB)
    expect(result.current.transport).toBe('idle')
    await waitFor(() => expect(result.current.job?.state).toBe('done'))
    expect(result.current.transport).toBe('polling')
    expect(FakeSocket.last.closed).toBe(true) // the stalled attempt is abandoned
  })

  it('does not abandon a socket that opens in time', async () => {
    const seen = pollStatus(done())
    await render(JOB)
    act(() => FakeSocket.last.open())
    await new Promise((r) => setTimeout(r, 120))
    expect(seen).toHaveLength(0)
    expect(FakeSocket.last.closed).toBe(false)
  })

  it('polls when the server refuses the token (close code 4401)', async () => {
    pollStatus(done())
    const { result } = await render(JOB)
    act(() => FakeSocket.last.drop(4401))
    await waitFor(() => expect(result.current.job?.state).toBe('done'))
  })

  it('polls straight away when a socket cannot even be created', async () => {
    FakeSocket.failToConstruct = true
    pollStatus(done())
    const { result } = await render(JOB)
    await waitFor(() => expect(result.current.job?.state).toBe('done'))
    expect(result.current.transport).toBe('polling')
  })

  it('polls straight away when nobody is signed in on the client, and reports the refusal', async () => {
    const seen = pollStatus(404)
    server.use(http.get(`${API}/analyze/status/${JOB}`, () => apiError(401, 'not_authenticated')))
    const { result } = await render(JOB, createApiClient({ storage: memoryStorage() }))
    expect(FakeSocket.instances).toHaveLength(0)
    await waitFor(() => expect(result.current.error?.code).toBe('not_authenticated'))
    expect(seen).toHaveLength(0)
  })

  it('stops polling after the final state', async () => {
    const seen = pollStatus(done())
    const { result } = await render(JOB)
    act(() => FakeSocket.last.drop())
    await waitFor(() => expect(result.current.job?.state).toBe('done'))
    const count = seen.length
    await new Promise((r) => setTimeout(r, FAST * 5))
    expect(seen).toHaveLength(count)
  })

  it('keeps going through a transient network failure', async () => {
    pollStatus('network', progress({ stage: 'separating', progress: 0.45 }), done())
    const { result } = await render(JOB)
    act(() => FakeSocket.last.drop())
    await waitFor(() => expect(result.current.job?.state).toBe('done'))
    expect(result.current.error).toBeNull()
  })

  it('reports a job that does not exist and gives up', async () => {
    const seen = pollStatus(404)
    const { result } = await render(JOB)
    act(() => FakeSocket.last.drop())
    await waitFor(() => expect(result.current.error?.code).toBe('not_found'))
    const count = seen.length
    await new Promise((r) => setTimeout(r, FAST * 5))
    expect(seen).toHaveLength(count)
  })
})

describe('cleanup', () => {
  it('closes the socket and stops polling on unmount', async () => {
    const seen = pollStatus(progress())
    const first = await render(JOB)
    act(() => FakeSocket.last.drop())
    await waitFor(() => expect(seen.length).toBeGreaterThan(0))
    first.unmount()
    const count = seen.length
    await new Promise((r) => setTimeout(r, FAST * 5))
    expect(seen).toHaveLength(count)

    const second = await render(JOB)
    const socket = FakeSocket.last
    second.unmount()
    expect(socket.closed).toBe(true)
  })

  it('switches to the new job when the id changes', async () => {
    const { result, rerender } = await render(JOB)
    const first = FakeSocket.last
    act(() => first.message(progress()))
    expect(result.current.job).not.toBeNull()

    rerender({ id: 'other-job' })
    expect(first.closed).toBe(true)
    expect(FakeSocket.last).not.toBe(first)
    expect(FakeSocket.last.url).toContain('/ws/jobs/other-job')
    expect(result.current.job).toBeNull() // the old job's progress is not shown for the new one
  })

  it('goes idle when the id is cleared', async () => {
    const { result, rerender } = await render(JOB)
    const socket = FakeSocket.last
    rerender({ id: null })
    expect(socket.closed).toBe(true)
    expect(result.current.transport).toBe('idle')
  })
})
