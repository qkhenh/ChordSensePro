import { ApiError, type ApiClient } from './http'
import type { Chart, DemoTrack, JobAccepted, JobProgress } from './types'

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

export function getChart(client: ApiClient, id: string): Promise<Chart> {
  return client.request<Chart>(`/charts/${encodeURIComponent(id)}`)
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
