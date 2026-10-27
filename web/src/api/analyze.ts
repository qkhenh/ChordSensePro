import { ApiError, type ApiClient } from './http'
import type { AudioTicketWire, Chart, ChartPage, DemoTrack, JobAccepted, JobProgress } from './types'

export function uploadFile(client: ApiClient, file: File): Promise<JobAccepted> {
  const body = new FormData()
  body.append('file', file, file.name)
  return client.request<JobAccepted>('/analyze/file', { method: 'POST', body })
}

export function analyzeUrl(client: ApiClient, url: string): Promise<JobAccepted> {
  return client.request<JobAccepted>('/analyze/url', { method: 'POST', body: { url } })
}

export function analyzeDemo(client: ApiClient, id: string): Promise<JobAccepted> {
  return client.request<JobAccepted>(`/analyze/demo/${encodeURIComponent(id)}`, { method: 'POST' })
}

export function getJob(client: ApiClient, id: string): Promise<JobProgress> {
  return client.request<JobProgress>(`/analyze/status/${encodeURIComponent(id)}`)
}

const chartCache = new WeakMap<ApiClient, Map<string, { etag: string; chart: Chart }>>()

/** Revalidates with the ETag of the last copy; a 304 hands that copy back unchanged. */
export async function getChart(client: ApiClient, id: string): Promise<Chart> {
  let cache = chartCache.get(client)
  if (!cache) chartCache.set(client, (cache = new Map()))
  const cached = cache.get(id)
  const { data, etag } = await client.requestConditional<Chart>(
    `/charts/${encodeURIComponent(id)}`,
    cached?.etag ?? null,
  )
  if (data === undefined) {
    if (cached) return cached.chart
    throw new ApiError(502, 'bad_response', 'Unexpected 304 without a cached chart')
  }
  if (etag) cache.set(id, { etag, chart: data })
  return data
}

export function listCharts(
  client: ApiClient,
  { cursor, limit }: { cursor?: string | null; limit?: number } = {},
): Promise<ChartPage> {
  const query = new URLSearchParams()
  if (cursor) query.set('cursor', cursor)
  if (limit) query.set('limit', String(limit))
  const qs = query.size ? `?${query}` : ''
  return client.request<ChartPage>(`/charts${qs}`)
}

export function listDemoTracks(client: ApiClient): Promise<DemoTrack[]> {
  return client.request<DemoTrack[]>('/demo-tracks')
}

/** Live progress of a job. Browsers cannot set headers on a WebSocket, so the access token
 * travels as a subprotocol: the server answers 4401 when it is missing or expired. */
export function openJobSocket(client: ApiClient, id: string): WebSocket {
  const token = client.getAccessToken()
  if (!token) throw new ApiError(401, 'not_authenticated', 'Not signed in')
  return new WebSocket(client.socketUrl(`/ws/jobs/${encodeURIComponent(id)}`), ['bearer', token])
}

export interface AudioTicket {
  /** A URL an `<audio>` element can load without credentials. */
  url: string
  expiresAt: Date
}

export async function getAudioTicket(client: ApiClient, chartId: string): Promise<AudioTicket> {
  const wire = await client.request<AudioTicketWire>(
    `/charts/${encodeURIComponent(chartId)}/audio-ticket`,
    { method: 'POST' },
  )
  return { url: client.resolveUrl(wire.url), expiresAt: new Date(wire.expires_at) }
}
