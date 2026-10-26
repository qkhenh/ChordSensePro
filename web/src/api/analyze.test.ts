import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  analyzeDemo,
  analyzeUrl,
  getChart,
  getJob,
  listDemoTracks,
  openJobSocket,
  uploadFile,
} from './analyze'
import { createApiClient, memoryStorage } from './http'
import type { Chart, JobProgress } from './types'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'

const API = '*/api/v1'
const JOB_ID = '5ae8d1b8-ce75-5e37-b3fd-0e987715e3f1'

const sequence = Object.values(
  import.meta.glob<JobProgress[]>('../../../contracts/fixtures/jobs/success.sequence.json', {
    eager: true,
    import: 'default',
  }),
)[0]
const charts = Object.values(
  import.meta.glob<Chart>('../../../contracts/fixtures/charts/ii_v_i_c.chart.json', {
    eager: true,
    import: 'default',
  }),
)

async function signedIn() {
  server.use(http.post(`${API}/auth/login`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage() })
  await client.login('ann@example.com', 'secret-pass')
  return client
}

const song = () => new File([new Uint8Array([82, 73, 70, 70])], 'Song One.wav', { type: 'audio/wav' })

describe('uploadFile', () => {
  it('sends the file as multipart under the field "file" and lets the browser set the boundary', async () => {
    const client = await signedIn()
    // jsdom's FormData reaches Node's fetch without the file name, so check the name where we set it.
    const append = vi.spyOn(FormData.prototype, 'append')
    let seen: { type: string | null; size: number; auth: string | null } | undefined
    server.use(
      http.post(`${API}/analyze/file`, async ({ request }) => {
        const form = await request.formData()
        const file = form.get('file') as File
        seen = {
          type: request.headers.get('content-type'),
          size: file.size,
          auth: request.headers.get('authorization'),
        }
        return HttpResponse.json({ job_id: JOB_ID }, { status: 202 })
      }),
    )
    const accepted = await uploadFile(client, song())
    expect(accepted).toEqual({ job_id: JOB_ID })
    expect(seen?.type).toMatch(/^multipart\/form-data; boundary=/)
    expect(seen).toMatchObject({ size: 4, auth: 'Bearer access-1' })
    expect(append).toHaveBeenCalledWith('file', expect.any(File), 'Song One.wav')
    append.mockRestore()
  })

  it('sends the same file again after a transparent token refresh', async () => {
    const client = await signedIn()
    const sizes: number[] = []
    let calls = 0
    server.use(
      http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse({ access_token: 'access-2' }))),
      http.post(`${API}/analyze/file`, async ({ request }) => {
        calls += 1
        if (calls === 1) return apiError(401, 'token_expired')
        sizes.push(((await request.formData()).get('file') as File).size)
        return HttpResponse.json({ job_id: JOB_ID }, { status: 202 })
      }),
    )
    await uploadFile(client, song())
    expect(calls).toBe(2)
    expect(sizes).toEqual([4])
  })

  it('surfaces the server refusal with its code', async () => {
    const client = await signedIn()
    server.use(http.post(`${API}/analyze/file`, () => apiError(415, 'unsupported_format')))
    await expect(uploadFile(client, song())).rejects.toMatchObject({ status: 415, code: 'unsupported_format' })
  })
})

describe('other calls', () => {
  it('analyzeUrl posts the link as JSON', async () => {
    const client = await signedIn()
    let body: unknown
    server.use(
      http.post(`${API}/analyze/url`, async ({ request }) => {
        body = await request.json()
        return HttpResponse.json({ job_id: JOB_ID }, { status: 202 })
      }),
    )
    expect(await analyzeUrl(client, 'https://example.com/a.mp3')).toEqual({ job_id: JOB_ID })
    expect(body).toEqual({ url: 'https://example.com/a.mp3' })
  })

  it('analyzeDemo posts to the demo endpoint without a body', async () => {
    const client = await signedIn()
    let path = ''
    server.use(
      http.post(`${API}/analyze/demo/:id`, ({ params }) => {
        path = String(params.id)
        return HttpResponse.json({ job_id: JOB_ID }, { status: 202 })
      }),
    )
    await analyzeDemo(client, 'ii_v_i_c')
    expect(path).toBe('ii_v_i_c')
  })

  it('getJob returns the contract JobProgress', async () => {
    const client = await signedIn()
    server.use(http.get(`${API}/analyze/status/${JOB_ID}`, () => HttpResponse.json(sequence[1])))
    const job: JobProgress = await getJob(client, JOB_ID)
    expect(job).toMatchObject({ state: 'processing', stage: 'normalizing' })
  })

  it('getChart returns the contract Chart', async () => {
    const client = await signedIn()
    server.use(http.get(`${API}/charts/${charts[0].id}`, () => HttpResponse.json(charts[0])))
    const chart: Chart = await getChart(client, charts[0].id)
    expect(chart.ai_events.length).toBeGreaterThan(0)
  })

  it('listDemoTracks returns the catalog', async () => {
    const client = await signedIn()
    const track = { id: 'ii_v_i_c', title: 'ii–V–I', artist: 'ChordSense demo', duration_ms: 9000, tempo_bpm: 120, key: 'C major' }
    server.use(http.get(`${API}/demo-tracks`, () => HttpResponse.json([track])))
    expect(await listDemoTracks(client)).toEqual([track])
  })
})

describe('openJobSocket', () => {
  class FakeSocket {
    static last: FakeSocket
    url: string
    protocols: string[]
    constructor(url: string, protocols: string[]) {
      this.url = url
      this.protocols = protocols
      FakeSocket.last = this
    }
  }

  afterEach(() => vi.unstubAllGlobals())

  it('connects to the job channel with the access token as a subprotocol', async () => {
    vi.stubGlobal('WebSocket', FakeSocket)
    const client = await signedIn()
    openJobSocket(client, JOB_ID)
    expect(FakeSocket.last.url).toBe(`ws://${location.host}/api/v1/ws/jobs/${JOB_ID}`)
    expect(FakeSocket.last.protocols).toEqual(['bearer', 'access-1'])
  })

  it('refuses to connect when nobody is signed in', () => {
    vi.stubGlobal('WebSocket', FakeSocket)
    const client = createApiClient({ storage: memoryStorage() })
    expect(() => openJobSocket(client, JOB_ID)).toThrowError(/signed in/i)
  })
})
