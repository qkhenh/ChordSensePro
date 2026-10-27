import type { BeatSource } from './beats'
import { scheduleClick, type ClickOptions, type Timbre } from './click'

/** Where the song is, paired with the audio clock at the same instant. */
export interface PlaybackClock {
  sample(): {
    songMs: number
    /** `AudioContext.currentTime` at the moment `songMs` was read. */
    ctxTime: number
    /** Playback speed: 1, 0.75, … */
    rate: number
    playing: boolean
  }
}

export interface SchedulerOptions {
  /** How far ahead, in real milliseconds, clicks are handed to the audio clock. */
  lookaheadMs?: number
  tickMs?: number
  timbre?: Timbre
  volume?: number
  /** Injected in tests; returns a function that cancels the click if it has not sounded yet. */
  schedule?: (ctx: BaseAudioContext, when: number, opts: ClickOptions) => (() => void) | void
}

const BACK_SLACK_MS = 250

interface Queued {
  when: number
  cancel: () => void
}

/**
 * Look-ahead metronome: a short timer hands upcoming beats to the audio clock, which then
 * sounds them with sample accuracy no matter how late a timer callback runs. Beats that slipped
 * by while the page was throttled are dropped, never played in a burst.
 */
export class MetronomeScheduler {
  private readonly ctx: BaseAudioContext
  private source: BeatSource
  private readonly clock: PlaybackClock
  private readonly lookaheadMs: number
  private readonly tickMs: number
  private readonly schedule: NonNullable<SchedulerOptions['schedule']>
  private timbre: Timbre
  private volume: number
  private scheduledUntilMs = Number.NEGATIVE_INFINITY
  private queued: Queued[] = []
  private timer: ReturnType<typeof setInterval> | undefined

  constructor(ctx: BaseAudioContext, source: BeatSource, clock: PlaybackClock, opts: SchedulerOptions = {}) {
    this.ctx = ctx
    this.source = source
    this.clock = clock
    this.lookaheadMs = opts.lookaheadMs ?? 120
    this.tickMs = opts.tickMs ?? 25
    this.timbre = opts.timbre ?? 'woodblock'
    this.volume = opts.volume ?? 0.8
    this.schedule = opts.schedule ?? scheduleClick
  }

  start(): void {
    if (this.timer !== undefined) return
    this.reset()
    this.tick()
    this.timer = setInterval(() => this.tick(), this.tickMs)
  }

  stop(): void {
    if (this.timer !== undefined) clearInterval(this.timer)
    this.timer = undefined
    this.cancelQueued(this.clock.sample().ctxTime)
  }

  /** Forget what was scheduled (the song position jumped: seek, loop wrap, speed change). */
  reset(): void {
    this.scheduledUntilMs = Number.NEGATIVE_INFINITY
    this.cancelQueued(this.clock.sample().ctxTime)
  }

  setSource(source: BeatSource): void {
    this.source = source
    this.reset()
  }

  setTimbre(timbre: Timbre): void {
    this.timbre = timbre
  }

  setVolume(volume: number): void {
    this.volume = volume
  }

  tick(): void {
    const { songMs, ctxTime, rate, playing } = this.clock.sample()
    if (!playing) {
      // Paused: clicks already handed to the audio clock must not sound after the pause.
      this.cancelQueued(ctxTime)
      this.scheduledUntilMs = Number.NEGATIVE_INFINITY
      return
    }
    if (songMs < this.scheduledUntilMs - this.lookaheadMs * rate - BACK_SLACK_MS) {
      this.cancelQueued(ctxTime) // the song went backwards behind our back (a scrub in an embedded player)
      this.scheduledUntilMs = Number.NEGATIVE_INFINITY
    }
    this.queued = this.queued.filter((c) => c.when > ctxTime)
    const horizon = songMs + this.lookaheadMs * rate
    const from = Math.max(this.scheduledUntilMs, songMs) // beats already behind us are dropped
    if (from >= horizon) return
    for (const beat of this.source.beatsBetween(from, horizon)) {
      const when = ctxTime + (beat.ms - songMs) / 1000 / rate
      const cancel = this.schedule(this.ctx, when, {
        accent: beat.accent,
        timbre: this.timbre,
        volume: this.volume,
      })
      if (cancel) this.queued.push({ when, cancel })
    }
    this.scheduledUntilMs = horizon
  }

  private cancelQueued(ctxTime: number): void {
    for (const click of this.queued) if (click.when > ctxTime) click.cancel()
    this.queued = []
  }
}
