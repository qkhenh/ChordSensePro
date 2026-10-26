import { EmbedTransport } from './base'
import type { SCNamespace, SCWidget } from './embeds'
import { assertRate, TransportFailure, type Rate, type Transport } from './types'

const MAX_ESTIMATE_MS = 1000

/**
 * Drives a SoundCloud Widget. The widget reports progress only a few times a second and cannot
 * change speed (`capabilities.rate` is false), so the UI hides the speed control for it.
 */
export class SoundCloudTransport extends EmbedTransport implements Transport {
  readonly kind = 'soundcloud' as const
  readonly capabilities = { rate: false }
  private readonly frame: HTMLIFrameElement
  private readonly loadApi: () => Promise<SCNamespace>
  private widget: SCWidget | null = null
  private events: string[] = []
  private length: number | null = null
  private sample = { ms: 0, at: 0 }

  constructor(
    frame: HTMLIFrameElement,
    loadApi: () => Promise<SCNamespace>,
    options: { now?: () => number; playTimeoutMs?: number } = {},
  ) {
    super(options)
    this.frame = frame
    this.loadApi = loadApi
  }

  async load(): Promise<void> {
    this.setState('loading')
    let api: SCNamespace
    try {
      api = await this.loadApi()
    } catch {
      const failure = new TransportFailure('network', 'The SoundCloud player could not be loaded.')
      this.fail(failure)
      throw failure
    }
    if (this.destroyed) return
    await new Promise<void>((resolve, reject) => {
      this.waiter = { resolve, reject }
      const widget = api.Widget(this.frame)
      const { Events } = api.Widget
      this.widget = widget
      const bind = (event: string, fn: (data?: unknown) => void) => {
        widget.bind(event, fn)
        this.events.push(event)
      }
      bind(Events.READY, () =>
        widget.getDuration((ms) => {
          this.length = ms
          this.setState('ready')
          this.waiter?.resolve()
          this.waiter = null
        }),
      )
      bind(Events.PLAY, () => {
        this.sample = { ms: this.sample.ms, at: this.now() }
        this.setState('playing')
      })
      bind(Events.PAUSE, () => this.setState('paused'))
      bind(Events.FINISH, () => this.setState('ended'))
      bind(Events.PLAY_PROGRESS, (data) => {
        const position = (data as { currentPosition?: number } | undefined)?.currentPosition
        if (typeof position === 'number') this.sample = { ms: position, at: this.now() }
      })
      bind(Events.ERROR, () =>
        this.fail(new TransportFailure('unavailable', 'This track cannot be played here.')),
      )
    })
  }

  async play(): Promise<void> {
    const widget = this.widget
    if (!widget || this.destroyed || this.state === 'error') return
    await this.startAndWait(() => widget.play())
  }

  pause(): void {
    if (!this.destroyed) this.widget?.pause()
  }

  seek(ms: number): void {
    if (this.destroyed || !this.widget) return
    const target = Math.max(0, this.length === null ? ms : Math.min(ms, this.length))
    this.widget.seekTo(target)
    this.sample = { ms: target, at: this.now() }
  }

  setRate(rate: Rate): void {
    assertRate(rate) // valid speeds are accepted and ignored: the widget has no such control
  }

  currentTimeMs(): number {
    if (this.destroyed) return 0
    if (this.state !== 'playing') return Math.round(this.sample.ms)
    return Math.round(this.sample.ms + Math.min(this.now() - this.sample.at, MAX_ESTIMATE_MS))
  }

  durationMs(): number | null {
    return this.length
  }

  destroy(): void {
    if (this.destroyed) return
    for (const event of this.events) this.widget?.unbind(event)
    this.events = []
    this.widget = null
    this.markDestroyed()
  }
}
