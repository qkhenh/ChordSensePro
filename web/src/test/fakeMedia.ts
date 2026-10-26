/** A scriptable HTMLAudioElement: tests decide when it loads, fails and ends. */
export class FakeAudio extends EventTarget {
  src = ''
  currentTime = 0
  duration = Number.NaN
  playbackRate = 1
  preservesPitch = false
  preload = ''
  paused = true
  ended = false
  readyState = 0
  error: { code: number } | null = null
  /** When true, `play()` rejects the way a browser does before any user gesture. */
  blockAutoplay = false
  loads = 0

  load() {
    this.loads += 1
    this.readyState = 0
    this.error = null
  }

  async play() {
    if (this.blockAutoplay) throw new DOMException('play() was blocked', 'NotAllowedError')
    this.paused = false
    this.ended = false
    this.dispatchEvent(new Event('play'))
    this.dispatchEvent(new Event('playing'))
  }

  pause() {
    if (this.paused) return
    this.paused = true
    this.dispatchEvent(new Event('pause'))
  }

  removeAttribute(name: string) {
    if (name === 'src') this.src = ''
  }

  // --- test controls -------------------------------------------------------------

  finishLoading(durationMs: number) {
    this.duration = durationMs / 1000
    this.readyState = 4
    this.dispatchEvent(new Event('loadedmetadata'))
    this.dispatchEvent(new Event('canplay'))
  }

  fail(code = 2) {
    this.error = { code }
    this.dispatchEvent(new Event('error'))
  }

  end() {
    this.paused = true
    this.ended = true
    this.currentTime = this.duration
    this.dispatchEvent(new Event('pause'))
    this.dispatchEvent(new Event('ended'))
  }
}
