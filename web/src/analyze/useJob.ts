import { useEffect, useRef, useState } from 'react'
import { getJob, openJobSocket } from '../api/analyze'
import { useApiClient } from '../api/context'
import { ApiError } from '../api/http'
import type { JobProgress } from '../api/types'

export const POLL_INTERVAL_MS = 2000
const CONNECT_TIMEOUT_MS = 5000
// An answer that will not change by asking again.
const FATAL_STATUSES = new Set([401, 403, 404])

export type Transport = 'idle' | 'ws' | 'polling'

interface JobWatch {
  job: JobProgress | null
  transport: Transport
  error: ApiError | null
}

const IDLE: JobWatch = { job: null, transport: 'idle', error: null }

/** What is known about one job; kept with its id so a different job never inherits it. */
type Tracked = JobWatch & { id: string | null }
const isFinal = (job: JobProgress) => job.state === 'done' || job.state === 'failed'

function parse(data: unknown): JobProgress | null {
  try {
    const value = JSON.parse(String(data))
    return value && typeof value.state === 'string' && typeof value.progress === 'number' ? value : null
  } catch {
    return null
  }
}

/**
 * Follow a job: WebSocket first (updates are pushed), polling every two seconds when the socket
 * cannot be had, never opens, or drops before the job is over (UC1 AF2).
 */
export function useJob(
  jobId: string | null,
  {
    pollMs = POLL_INTERVAL_MS,
    connectMs = CONNECT_TIMEOUT_MS,
    onMissing,
  }: { pollMs?: number; connectMs?: number; onMissing?: () => void } = {},
): JobWatch {
  const client = useApiClient()
  const [tracked, setTracked] = useState<Tracked>({ ...IDLE, id: null })
  const missing = useRef(onMissing)
  useEffect(() => {
    missing.current = onMissing
  })

  useEffect(() => {
    if (!jobId) return
    const setWatch = (update: (prev: JobWatch) => JobWatch) =>
      setTracked((prev) => ({ ...update(prev.id === jobId ? prev : IDLE), id: jobId }))

    let stopped = false
    let finished = false
    let polling = false
    let socket: WebSocket | null = null
    let pollTimer: ReturnType<typeof setTimeout> | undefined
    let connectTimer: ReturnType<typeof setTimeout> | undefined

    const apply = (next: JobProgress) => {
      if (isFinal(next)) finished = true
      // A late message must not move the display backwards.
      setWatch((prev) =>
        prev.job && !isFinal(next) && next.progress < prev.job.progress
          ? prev
          : { ...prev, job: next, error: null },
      )
    }

    const startPolling = () => {
      if (stopped || finished || polling) return
      polling = true
      clearTimeout(connectTimer)
      setWatch((prev) => ({ ...prev, transport: 'polling' }))
      const tick = async () => {
        try {
          const next = await getJob(client, jobId)
          if (stopped) return
          apply(next)
          if (isFinal(next)) return
        } catch (error) {
          if (stopped) return
          if (error instanceof ApiError && FATAL_STATUSES.has(error.status)) {
            setWatch((prev) => ({ ...prev, error }))
            if (error.status === 404) missing.current?.()
            return
          }
          // network trouble or a server hiccup: ask again on the next round
        }
        pollTimer = setTimeout(tick, pollMs)
      }
      void tick()
    }

    try {
      socket = openJobSocket(client, jobId)
    } catch {
      startPolling()
    }
    if (socket) {
      const live = socket
      connectTimer = setTimeout(() => {
        live.close()
        startPolling()
      }, connectMs)
      live.onopen = () => {
        clearTimeout(connectTimer)
        setWatch((prev) => ({ ...prev, transport: 'ws' }))
      }
      live.onmessage = (event) => {
        const next = parse(event.data)
        if (!next) return
        apply(next)
        if (isFinal(next)) live.close()
      }
      live.onclose = startPolling
    }

    return () => {
      stopped = true
      clearTimeout(pollTimer)
      clearTimeout(connectTimer)
      socket?.close()
    }
  }, [client, jobId, pollMs, connectMs])

  return tracked.id === jobId && jobId !== null ? tracked : IDLE
}
