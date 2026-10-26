/** The small slice of the YouTube IFrame and SoundCloud Widget APIs that the transports use. */

export interface YTPlayer {
  playVideo(): void
  pauseVideo(): void
  seekTo(seconds: number, allowSeekAhead: boolean): void
  getCurrentTime(): number
  getDuration(): number
  setPlaybackRate(rate: number): void
  getAvailablePlaybackRates(): number[]
  destroy(): void
}

export interface YTNamespace {
  Player: new (
    host: HTMLElement,
    options: {
      videoId: string
      playerVars?: Record<string, string | number>
      events: {
        onReady(event: unknown): void
        onStateChange(event: { data: number }): void
        onError(event: { data: number }): void
      }
    },
  ) => YTPlayer
}

export interface SCWidget {
  bind(event: string, listener: (data?: unknown) => void): void
  unbind(event: string): void
  play(): void
  pause(): void
  seekTo(ms: number): void
  getPosition(callback: (ms: number) => void): void
  getDuration(callback: (ms: number) => void): void
}

export interface SCNamespace {
  Widget: ((frame: HTMLIFrameElement) => SCWidget) & {
    Events: {
      READY: string
      PLAY: string
      PAUSE: string
      FINISH: string
      PLAY_PROGRESS: string
      ERROR: string
    }
  }
}
