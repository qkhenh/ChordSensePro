import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { useApiClient } from '../api/context'
import type { Chart } from '../api/types'
import { createTransport, type TransportDeps } from './select'
import type { Transport, TransportError, TransportState } from './types'

export interface TransportView {
  transport: Transport | null
  state: TransportState
  error: TransportError | null
}

const IDLE: TransportView = { transport: null, state: 'idle', error: null }

/** A tiny store so the transport's own events can feed React without setState in an effect. */
class ViewStore {
  snapshot: TransportView = IDLE
  private listeners = new Set<() => void>()

  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getSnapshot = () => this.snapshot

  set(next: TransportView) {
    this.snapshot = next
    for (const listener of [...this.listeners]) listener()
  }
}

/**
 * Create the transport for a chart once there is an element to host embedded players, load it,
 * and expose its state. A different chart, or leaving the page, destroys it.
 */
export function useTransport(
  chart: { id: string; playback: Chart['source']['playback'] },
  mount: HTMLElement | null,
  loaders?: TransportDeps['loaders'],
  /** Bump to throw the transport away and build a new one (a retry after a failure). */
  attempt = 0,
): TransportView {
  const client = useApiClient()
  const [store] = useState(() => new ViewStore())
  const loadersRef = useRef(loaders)
  useEffect(() => {
    loadersRef.current = loaders
  })

  const { kind, embed_id: embedId, audio_url: audioUrl } = chart.playback
  const chartId = chart.id

  useEffect(() => {
    if (!mount) return
    const transport = createTransport(
      { kind, embed_id: embedId, audio_url: audioUrl },
      { chartId, client, mount, loaders: loadersRef.current },
    )
    const publish = (error: TransportError | null) => {
      // A failure stays on show until the transport is healthy again.
      const healthy = transport.state === 'playing' || transport.state === 'ready'
      store.set({ transport, state: transport.state, error: error ?? (healthy ? null : store.snapshot.error) })
    }
    store.set({ transport, state: transport.state, error: null })
    const offState = transport.on('state', () => publish(null))
    const offError = transport.on('error', (error) => publish(error))
    transport.load().catch(() => {
      /* already reported through the error event */
    })
    return () => {
      offState()
      offError()
      transport.destroy()
      store.set(IDLE)
    }
  }, [store, client, mount, chartId, kind, embedId, audioUrl, attempt])

  return useSyncExternalStore(store.subscribe, store.getSnapshot)
}
