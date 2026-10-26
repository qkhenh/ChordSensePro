import { assertRate, TransportFailure, type Rate, type Transport, type TransportError, type TransportState } from './types'

const HAVE_METADATA = 1

function classify(reason: unknown): TransportFailure {
  const gone = typeof reason === 'object' && reason !== null && (reason as { status?: number }).status === 404
  return gone
    ? new TransportFailure('unavailable', "This song's audio is no longer available.")
    : new TransportFailure('network', 'The audio could not be loaded. Check your connection and try again.')
}

interface Resume {
  atMs: number
  after: TransportState
}

/**
 * Plays a chart's own audio through an `<audio>` element. The element cannot send credentials, so
 * `source` hands over a URL that carries a short-lived ticket; when the ticket stops working
 * mid-song the transport asks for a new one once and carries on from the same spot.
 */
export class HtmlAudioTransport implements Transport {
  readonly kind = 'html' as const
  readonly capabilities = { rate: true }

  private current: TransportState = 'idle'
  private readonly audio: HTMLAudioElement
  private readonly source: () => Promise<string>
  private readonly stateListeners = new Set<() => void>()
  private readonly errorListeners = new Set<(error: TransportError) => void>()
  private readonly cleanups: (() => void)[] = []
  private pendingSeekMs: number | null = null
  private resume: Resume | null = null
  private refreshed = false // a ticket refresh has been spent and nothing has played since
  private waiter: { resolve: () => void; reject: (error: TransportFailure) => void } | null = null
  private destroyed = false
  private rate: Rate = 1

  constructor(audio: HTMLAudioElement, source: () => Promise<string>) {
    this.audio = audio
    this.source = source
    audio.preload = 'auto'
    this.keepPitch()
    const on = (type: string, fn: () => void) => {
      audio.addEventListener(type, fn)
      this.cleanups.push(() => audio.removeEventListener(type, fn))
    }
    on('loadedmetadata', () => this.onMetadata())
    on('playing', () => {
      this.refreshed = false
      this.setState('playing')
    })
    on('pause', () => {
      if (!this.resume && !audio.ended && this.current === 'playing') this.setState('paused')
    })
    on('ended', () => this.setState('ended'))
    on('error', () => void this.onMediaError())
  }

  get state(): TransportState {
    return this.current
  }

  async load(): Promise<void> {
    this.setState('loading')
    let url: string
    try {
      url = await this.source()
    } catch (reason) {
      const failure = classify(reason)
      this.fail(failure)
      throw failure
    }
    if (this.destroyed) return
    await new Promise<void>((resolve, reject) => {
      this.waiter = { resolve, reject }
      this.audio.src = url
      this.audio.load()
    })
  }

  async play(): Promise<void> {
    if (this.destroyed) return
    if (this.current === 'ended') this.audio.currentTime = 0
    try {
      await this.audio.play()
    } catch (reason) {
      if (this.destroyed) return
      if (reason instanceof DOMException && reason.name === 'NotAllowedError') {
        this.emitError(
          new TransportFailure('autoplay_blocked', 'Your browser blocked automatic playback. Press play to start.'),
        )
        this.setState('paused')
        return
      }
      this.fail(classify(reason))
    }
  }

  pause(): void {
    if (!this.destroyed) this.audio.pause()
  }

  seek(ms: number): void {
    if (this.destroyed) return
    const length = this.durationMs()
    const target = Math.max(0, length === null ? ms : Math.min(ms, length))
    if (this.audio.readyState < HAVE_METADATA) {
      this.pendingSeekMs = target
      return
    }
    this.audio.currentTime = target / 1000
    if (this.current === 'ended') this.setState('paused')
  }

  setRate(rate: Rate): void {
    assertRate(rate)
    this.rate = rate
    this.audio.playbackRate = rate
    this.keepPitch()
  }

  currentTimeMs(): number {
    return Math.round(this.audio.currentTime * 1000)
  }

  durationMs(): number | null {
    return Number.isFinite(this.audio.duration) ? Math.round(this.audio.duration * 1000) : null
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

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    for (const undo of this.cleanups) undo()
    this.audio.pause()
    this.audio.removeAttribute('src')
    this.stateListeners.clear()
    this.errorListeners.clear()
    this.waiter = null
    this.current = 'idle'
  }

  // --- internals -----------------------------------------------------------------

  /** Time-stretch without changing pitch; some browsers reset the flag when the source changes. */
  private keepPitch(): void {
    const audio = this.audio as HTMLAudioElement & {
      mozPreservesPitch?: boolean
      webkitPreservesPitch?: boolean
    }
    audio.preservesPitch = true
    audio.mozPreservesPitch = true
    audio.webkitPreservesPitch = true
  }

  private setState(next: TransportState): void {
    if (this.destroyed || this.current === next) return
    this.current = next
    for (const fn of [...this.stateListeners]) fn()
  }

  private emitError(error: TransportError): void {
    for (const fn of [...this.errorListeners]) fn(error)
  }

  private fail(failure: TransportFailure): void {
    if (this.destroyed) return
    this.setState('error')
    this.emitError(failure)
    this.waiter?.reject(failure)
    this.waiter = null
  }

  private onMetadata(): void {
    const audio = this.audio
    const wanted = this.resume?.atMs ?? this.pendingSeekMs
    if (wanted !== null) {
      const length = this.durationMs()
      audio.currentTime = (length === null ? wanted : Math.min(wanted, length)) / 1000
      this.pendingSeekMs = null
    }
    audio.playbackRate = this.rate
    this.keepPitch()

    const resumed = this.resume
    this.resume = null
    this.setState(resumed && resumed.after !== 'playing' ? resumed.after : resumed ? 'loading' : 'ready')
    this.waiter?.resolve()
    this.waiter = null
    if (resumed?.after === 'playing') void this.play()
  }

  private async onMediaError(): Promise<void> {
    if (this.destroyed || this.current === 'error') return
    if (this.refreshed) {
      this.fail(new TransportFailure('network', 'The audio stopped working and could not be reloaded.'))
      return
    }
    this.refreshed = true
    const before = this.current
    this.resume = {
      atMs: this.currentTimeMs(),
      after: before === 'playing' ? 'playing' : before === 'loading' || before === 'ready' ? 'ready' : 'paused',
    }
    this.setState('loading')
    try {
      const url = await this.source()
      if (this.destroyed) return
      this.audio.src = url
      this.audio.load()
    } catch (reason) {
      this.resume = null
      this.fail(classify(reason))
    }
  }
}
