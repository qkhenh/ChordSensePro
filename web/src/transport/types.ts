export type TransportState = 'idle' | 'loading' | 'ready' | 'playing' | 'paused' | 'ended' | 'error'

export type Rate = 0.5 | 0.75 | 1
export const RATES: readonly Rate[] = [0.5, 0.75, 1]

export interface TransportError {
  code: 'autoplay_blocked' | 'unavailable' | 'embed_blocked' | 'network'
  message: string
}

/** The error object a transport rejects with and reports; `code` is what the UI translates. */
export class TransportFailure extends Error implements TransportError {
  readonly code: TransportError['code']

  constructor(code: TransportError['code'], message: string) {
    super(message)
    this.name = 'TransportFailure'
    this.code = code
  }
}

/**
 * One way of playing a song, whatever the source: an `<audio>` element we own, or a YouTube /
 * SoundCloud player we only drive. The practice workspace talks to this and nothing else.
 */
export interface Transport {
  readonly kind: 'html' | 'youtube' | 'soundcloud'
  /** What the source can do; SoundCloud's widget cannot change speed. */
  readonly capabilities: { rate: boolean }
  readonly state: TransportState
  load(): Promise<void>
  play(): Promise<void>
  pause(): void
  /** Position in song milliseconds; clamped to the song. */
  seek(ms: number): void
  /** Speed with pitch preserved. Anything but 0.5, 0.75 and 1 throws a RangeError. */
  setRate(rate: Rate): void
  currentTimeMs(): number
  durationMs(): number | null
  on(event: 'state', fn: () => void): () => void
  on(event: 'error', fn: (error: TransportError) => void): () => void
  destroy(): void
}

export function assertRate(rate: number): asserts rate is Rate {
  if (!(RATES as readonly number[]).includes(rate)) {
    throw new RangeError(`unsupported playback rate ${rate}`)
  }
}
