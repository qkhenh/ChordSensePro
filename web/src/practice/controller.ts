import type { Grid } from '../grid/grid'
import { LoopController } from '../loop/loop'
import type { Transport } from '../transport/types'

type Seekable = Pick<Transport, 'seek' | 'currentTimeMs'>

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

  constructor(transport: Seekable, grid: Grid) {
    this.transport = transport
    this.grid = grid
    this.loop = new LoopController(grid.durationMs)
  }

  seek(ms: number): void {
    this.transport.seek(ms)
    this.loop.noteSeek(ms)
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
    const back = this.loop.tick(positionMs)
    if (back === null) return false
    this.transport.seek(back)
    for (const listener of [...this.wrapListeners]) listener()
    return true
  }

  /** The metronome resets here (T13): a wrap is a jump the click schedule must not play through. */
  onWrap(listener: () => void): () => void {
    this.wrapListeners.add(listener)
    return () => {
      this.wrapListeners.delete(listener)
    }
  }
}
