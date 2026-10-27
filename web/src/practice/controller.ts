import type { Grid } from '../grid/grid'
import { LoopController } from '../loop/loop'
import type { Transport } from '../transport/types'

type Seekable = Pick<Transport, 'seek' | 'currentTimeMs'> & Partial<Pick<Transport, 'state' | 'play'>>

// A loop that touches the end of the song: the audio may stop a hair before the loop's end.
const END_SLACK_MS = 250

/**
 * What the person does to the playhead in the practice workspace: jump, step by bar, skip by
 * time, loop. Every move goes through here so the loop always knows the playhead was moved by
 * the user and never "wraps" because of a jump.
 */
export class PracticeController {
  readonly loop: LoopController
  private readonly transport: Seekable
  private readonly grid: Grid
  private readonly wrapListeners = new Set<() => void>()
  private readonly jumpListeners = new Set<() => void>()

  constructor(transport: Seekable, grid: Grid) {
    this.transport = transport
    this.grid = grid
    this.loop = new LoopController(grid.durationMs)
  }

  seek(ms: number): void {
    this.transport.seek(ms)
    this.loop.noteSeek(ms)
    this.announce(this.jumpListeners)
  }

  seekToBar(index: number): void {
    this.seek(this.grid.barStartMs(index))
  }

  /** One bar forward or back; from the pre-roll "forward" is the first bar and "back" is 0. */
  stepBars(delta: number): void {
    const bars = this.grid.bars.length
    if (bars === 0) return
    const current = this.grid.barAtMs(this.transport.currentTimeMs())
    if (current < 0) {
      if (delta > 0) this.seekToBar(0)
      else this.seek(0)
      return
    }
    this.seekToBar(Math.min(Math.max(current + delta, 0), bars - 1))
  }

  skip(deltaMs: number): void {
    const target = this.transport.currentTimeMs() + deltaMs
    this.seek(Math.min(Math.max(target, 0), this.grid.durationMs))
  }

  /** Called with the playhead position on every frame; true when the loop wrapped. */
  frame(positionMs: number): boolean {
    const ended = this.transport.state === 'ended'
    let back = this.loop.tick(positionMs)
    const range = this.loop.range
    if (back === null && ended && range && positionMs >= range.startMs && range.endMs >= this.grid.durationMs - END_SLACK_MS) {
      back = range.startMs // the song ended inside a loop that runs to its end
      this.loop.noteSeek(back)
    }
    if (back === null) return false
    this.transport.seek(back)
    if (ended) void this.transport.play?.().catch(() => {}) // an ended element does not play on by itself
    this.announce(this.wrapListeners)
    this.announce(this.jumpListeners)
    return true
  }

  /** The metronome resets here (T13): a wrap is a jump the click schedule must not play through. */
  onWrap(listener: () => void): () => void {
    this.wrapListeners.add(listener)
    return () => {
      this.wrapListeners.delete(listener)
    }
  }

  /** The playhead moved by something other than playing: a seek, a step, a skip or a loop wrap. */
  onJump(listener: () => void): () => void {
    this.jumpListeners.add(listener)
    return () => {
      this.jumpListeners.delete(listener)
    }
  }

  private announce(listeners: Set<() => void>): void {
    for (const listener of [...listeners]) listener()
  }
}
