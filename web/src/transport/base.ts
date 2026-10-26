import type { TransportError, TransportState } from './types'
import { TransportFailure } from './types'

const PLAY_TIMEOUT_MS = 3000

/** What the embedded-player transports share: state, listeners, failure and "did play start?". */
export abstract class EmbedTransport {
  private current: TransportState = 'idle'
  protected destroyed = false
  private readonly stateListeners = new Set<() => void>()
  private readonly errorListeners = new Set<(error: TransportError) => void>()
  /** Settles the pending `load()`; set while a load is in flight. */
  protected waiter: { resolve: () => void; reject: (error: TransportFailure) => void } | null = null
  protected readonly now: () => number
  private readonly playTimeoutMs: number

  protected constructor(options: { now?: () => number; playTimeoutMs?: number }) {
    this.now = options.now ?? (() => performance.now())
    this.playTimeoutMs = options.playTimeoutMs ?? PLAY_TIMEOUT_MS
  }

  get state(): TransportState {
    return this.current
  }

  on(event: 'state', fn: () => void): () => void
  on(event: 'error', fn: (error: TransportError) => void): () => void
  on(event: 'state' | 'error', fn: (() => void) | ((error: TransportError) => void)): () => void {
    const set = (event === 'state' ? this.stateListeners : this.errorListeners) as Set<typeof fn>
    set.add(fn)
    return () => {
      set.delete(fn)
    }
  }

  protected setState(next: TransportState): void {
    if (this.destroyed || this.current === next) return
    this.current = next
    for (const fn of [...this.stateListeners]) fn()
  }

  protected fail(failure: TransportFailure): void {
    if (this.destroyed) return
    this.setState('error')
    for (const fn of [...this.errorListeners]) fn(failure)
    this.waiter?.reject(failure)
    this.waiter = null
  }

  protected markDestroyed(): void {
    this.destroyed = true
    this.stateListeners.clear()
    this.errorListeners.clear()
    this.waiter = null
    this.current = 'idle'
  }

  /**
   * Start playback and wait for the player to say it is playing. An embedded player that is
   * refused autoplay just stays silent, so after a while that is reported as blocked.
   */
  protected async startAndWait(start: () => void): Promise<void> {
    const reached = new Promise<void>((resolve) => {
      const off = this.on('state', () => {
        if (this.current === 'playing') {
          off()
          clearTimeout(timer)
          resolve()
        }
      })
      const timer = setTimeout(() => {
        off()
        resolve()
      }, this.playTimeoutMs)
    })
    start()
    if (this.current !== 'playing') await reached
    if (this.destroyed || this.current === 'playing' || this.current === 'error') return
    for (const fn of [...this.errorListeners]) {
      fn(new TransportFailure('autoplay_blocked', 'The player did not start. Press play to try again.'))
    }
    this.setState('paused')
  }
}
