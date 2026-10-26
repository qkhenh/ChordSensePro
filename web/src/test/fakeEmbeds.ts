import type { SCNamespace, YTNamespace } from '../transport/embeds'

/** A scriptable YouTube IFrame Player API. */
export function createFakeYouTube() {
  const state = {
    time: 0,
    duration: 0,
    rate: 1,
    rates: [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
    blockAutoplay: false,
    destroyed: false,
    seeks: [] as [number, boolean][],
  }
  const created: { host: unknown; videoId: string }[] = []
  let events!: {
    onReady(): void
    onStateChange(e: { data: number }): void
    onError(e: { data: number }): void
  }

  class Player {
    constructor(host: unknown, opts: { videoId: string; events: typeof events }) {
      events = opts.events
      created.push({ host, videoId: opts.videoId })
    }
    playVideo() {
      if (!state.blockAutoplay) events.onStateChange({ data: 1 })
    }
    pauseVideo() {
      events.onStateChange({ data: 2 })
    }
    seekTo(seconds: number, allowSeekAhead: boolean) {
      state.seeks.push([seconds, allowSeekAhead])
      state.time = seconds
    }
    getCurrentTime() {
      return state.time
    }
    getDuration() {
      return state.duration
    }
    setPlaybackRate(rate: number) {
      state.rate = rate
    }
    getAvailablePlaybackRates() {
      return state.rates
    }
    destroy() {
      state.destroyed = true
    }
  }

  return {
    api: { Player } as unknown as YTNamespace,
    created,
    state,
    driver: {
      finishLoading(durationMs: number) {
        state.duration = durationMs / 1000
        events.onReady()
      },
      breakDown: () => events.onError({ data: 150 }),
      rate: () => state.rate,
      emitState: (data: number) => events.onStateChange({ data }),
      emitError: (data: number) => events.onError({ data }),
    },
  }
}

/** A scriptable SoundCloud Widget API. */
export function createFakeSoundCloud() {
  const state = { position: 0, duration: 0, blockAutoplay: false, unbound: [] as string[] }
  const bound = new Map<string, (data?: unknown) => void>()
  const Events = {
    READY: 'ready',
    PLAY: 'play',
    PAUSE: 'pause',
    FINISH: 'finish',
    PLAY_PROGRESS: 'playProgress',
    ERROR: 'error',
  }
  const fire = (event: string, data?: unknown) => bound.get(event)?.(data)

  const widget = {
    bind: (event: string, fn: (data?: unknown) => void) => bound.set(event, fn),
    unbind: (event: string) => {
      state.unbound.push(event)
      bound.delete(event)
    },
    play: () => {
      if (!state.blockAutoplay) fire(Events.PLAY)
    },
    pause: () => fire(Events.PAUSE),
    seekTo: (ms: number) => {
      state.position = ms
    },
    getPosition: (cb: (ms: number) => void) => cb(state.position),
    getDuration: (cb: (ms: number) => void) => cb(state.duration),
  }
  const api = { Widget: Object.assign(() => widget, { Events }) } as unknown as SCNamespace

  return {
    api,
    state,
    driver: {
      finishLoading(durationMs: number) {
        state.duration = durationMs
        fire(Events.READY)
      },
      breakDown: () => fire(Events.ERROR),
      rate: () => 1,
      progress(ms: number) {
        state.position = ms
        fire(Events.PLAY_PROGRESS, { currentPosition: ms })
      },
      finish: () => fire(Events.FINISH),
    },
  }
}
