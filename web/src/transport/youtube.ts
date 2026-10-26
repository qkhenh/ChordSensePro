import { EmbedTransport } from './base'
import type { YTNamespace, YTPlayer } from './embeds'
import { assertRate, TransportFailure, type Rate, type Transport } from './types'

// YouTube's own error numbers (IFrame Player API).
const NOT_EMBEDDABLE = new Set([101, 150])
const MAX_ESTIMATE_MS = 500

function failureFor(code: number): TransportFailure {
  return NOT_EMBEDDABLE.has(code)
    ? new TransportFailure('embed_blocked', 'The owner of this video does not allow it to be played here.')
    : new TransportFailure('unavailable', 'This video cannot be played (removed, private or invalid).')
}

/**
 * Drives a YouTube IFrame player. The player reports its position only every few hundred
 * milliseconds, so between reports the position is estimated from the clock (never more than
 * half a second ahead) to keep a 60 fps playhead smooth; while buffering it is not estimated.
 */
export class YouTubeTransport extends EmbedTransport implements Transport {
  readonly kind = 'youtube' as const
  private readonly host: HTMLElement
  private readonly videoId: string
  private readonly loadApi: () => Promise<YTNamespace>
  private player: YTPlayer | null = null
  private length: number | null = null
  private rateOk = true
  private rate: Rate = 1
  private sample: { raw: number; at: number } | null = null

  constructor(
    host: HTMLElement,
    videoId: string,
    loadApi: () => Promise<YTNamespace>,
    options: { now?: () => number; playTimeoutMs?: number } = {},
  ) {
    super(options)
    this.host = host
    this.videoId = videoId
    this.loadApi = loadApi
  }

  get capabilities() {
    return { rate: this.rateOk }
  }

  async load(): Promise<void> {
    this.setState('loading')
    let api: YTNamespace
    try {
      api = await this.loadApi()
    } catch {
      const failure = new TransportFailure('network', 'The YouTube player could not be loaded.')
      this.fail(failure)
      throw failure
    }
    if (this.destroyed) return
    await new Promise<void>((resolve, reject) => {
      this.waiter = { resolve, reject }
      this.player = new api.Player(this.host, {
        videoId: this.videoId,
        playerVars: { playsinline: 1, rel: 0, modestbranding: 1 },
        events: {
          onReady: () => this.onReady(),
          onStateChange: (event) => this.onState(event.data),
          onError: (event) => this.fail(failureFor(event.data)),
        },
      })
    })
  }

  async play(): Promise<void> {
    const player = this.player
    if (!player || this.destroyed || this.state === 'error') return
    if (this.state === 'ended') player.seekTo(0, true)
    await this.startAndWait(() => player.playVideo())
  }

  pause(): void {
    if (!this.destroyed) this.player?.pauseVideo()
  }

  seek(ms: number): void {
    if (this.destroyed || !this.player) return
    const target = Math.max(0, this.length === null ? ms : Math.min(ms, this.length))
    this.sample = null
    this.player.seekTo(target / 1000, true)
  }

  setRate(rate: Rate): void {
    assertRate(rate)
    this.rate = rate
    if (this.rateOk && !this.destroyed) this.player?.setPlaybackRate(rate)
  }

  currentTimeMs(): number {
    if (this.destroyed || !this.player) return 0
    const raw = this.player.getCurrentTime() * 1000
    if (this.state !== 'playing') {
      this.sample = null
      return Math.round(raw)
    }
    const at = this.now()
    if (this.sample === null || this.sample.raw !== raw) this.sample = { raw, at }
    return Math.round(raw + Math.min(at - this.sample.at, MAX_ESTIMATE_MS) * this.rate)
  }

  durationMs(): number | null {
    return this.length
  }

  destroy(): void {
    if (this.destroyed) return
    this.player?.destroy()
    this.player = null
    this.markDestroyed()
  }

  private onReady(): void {
    const player = this.player
    if (!player) return
    this.length = Math.round(player.getDuration() * 1000)
    const rates = player.getAvailablePlaybackRates()
    this.rateOk = rates.includes(0.5) && rates.includes(0.75)
    this.setState('ready')
    this.waiter?.resolve()
    this.waiter = null
  }

  private onState(code: number): void {
    this.sample = null
    if (code === 1) this.setState('playing')
    else if (code === 2) this.setState('paused')
    else if (code === 0) this.setState('ended')
    else if (code === 3) this.setState('loading') // buffering
    else if (code === 5) this.setState('ready') // cued
  }
}
