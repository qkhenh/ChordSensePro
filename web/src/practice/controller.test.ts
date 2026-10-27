import { describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { PracticeController } from './controller'

const meta: AnalysisMetadata = {
  duration_ms: 60_000,
  key: { tonic: 'C', mode: 'major', confidence: 0.9 },
  beat_grid: {
    tempo_bpm: 120,
    tempo_confidence: 0.9,
    time_signature: { numerator: 4, denominator: 4 },
    beats_per_bar: 4,
    first_beat_offset_ms: 1000,
    downbeats_ms: null,
  },
  model_version: 'm',
  pipeline_version: 'p',
}
const grid = buildGrid(meta, []) // pre-roll to 1000 ms, then 2000 ms bars: bar k starts at 1000 + 2000k

function setup(startMs = 0) {
  const position = { ms: startMs }
  const seek = vi.fn((ms: number) => {
    position.ms = ms
  })
  const controller = new PracticeController({ seek, currentTimeMs: () => position.ms }, grid)
  return { controller, seek, position }
}

describe('seeking', () => {
  it('moves the transport and tells the loop so it does not drag the playhead back', () => {
    const { controller, seek } = setup()
    controller.loop.setFromBars(grid, 8, 15) // 17 000 .. 33 000
    controller.seek(40_000)
    expect(seek).toHaveBeenCalledWith(40_000)
    expect(controller.frame(40_000)).toBe(false)
    expect(controller.frame(40_500)).toBe(false)
    expect(seek).toHaveBeenCalledTimes(1)
  })

  it('seeks to the start of a bar', () => {
    const { controller, seek } = setup()
    controller.seekToBar(8)
    expect(seek).toHaveBeenCalledWith(17_000)
  })
})

describe('stepping by bars', () => {
  it('goes to the next and the previous bar', () => {
    const { controller, seek } = setup(1000 + 2000 * 3 + 500) // inside bar 3
    controller.stepBars(1)
    expect(seek).toHaveBeenLastCalledWith(1000 + 2000 * 4)
    controller.stepBars(-1)
    expect(seek).toHaveBeenLastCalledWith(1000 + 2000 * 3)
  })

  it('stays on the first and the last bar at the edges', () => {
    const first = setup(1200)
    first.controller.stepBars(-1)
    expect(first.seek).toHaveBeenLastCalledWith(1000)
    const last = setup(59_000)
    last.controller.stepBars(1)
    expect(last.seek).toHaveBeenLastCalledWith(grid.barStartMs(grid.bars.length - 1))
  })

  it('treats the pre-roll as before the first bar', () => {
    const forward = setup(300)
    forward.controller.stepBars(1)
    expect(forward.seek).toHaveBeenLastCalledWith(1000)
    const back = setup(300)
    back.controller.stepBars(-1)
    expect(back.seek).toHaveBeenLastCalledWith(0)
  })
})

describe('skipping by time', () => {
  it('moves by the given milliseconds and clamps to the song', () => {
    const { controller, seek } = setup(10_000)
    controller.skip(5000)
    expect(seek).toHaveBeenLastCalledWith(15_000)
    controller.skip(-20_000)
    expect(seek).toHaveBeenLastCalledWith(0)
    controller.skip(1_000_000)
    expect(seek).toHaveBeenLastCalledWith(60_000)
  })
})

describe('loop wrap', () => {
  it('seeks back to A once when the playhead passes B and announces it', () => {
    const { controller, seek } = setup()
    controller.loop.setFromBars(grid, 8, 15) // 17 000 .. 33 000
    const wrapped = vi.fn()
    controller.onWrap(wrapped)
    controller.frame(32_990)
    expect(controller.frame(33_010)).toBe(true)
    expect(seek).toHaveBeenCalledTimes(1)
    expect(seek).toHaveBeenCalledWith(17_000)
    expect(wrapped).toHaveBeenCalledTimes(1)
    expect(controller.frame(17_016)).toBe(false)
    expect(wrapped).toHaveBeenCalledTimes(1)
  })

  it('does nothing without a loop, and after the loop is cleared', () => {
    const { controller, seek } = setup()
    controller.frame(10)
    controller.frame(99_000)
    controller.loop.setFromBars(grid, 8, 15)
    controller.frame(20_000)
    controller.loop.clear()
    controller.frame(34_000)
    expect(seek).not.toHaveBeenCalled()
  })

  it('stops notifying a listener that unsubscribed', () => {
    const { controller } = setup()
    controller.loop.setFromBars(grid, 8, 15)
    const wrapped = vi.fn()
    const off = controller.onWrap(wrapped)
    off()
    controller.frame(32_990)
    controller.frame(33_010)
    expect(wrapped).not.toHaveBeenCalled()
  })
})

describe('jumps', () => {
  it('announces every seek, step, skip and loop wrap, but not a plain frame', () => {
    const { controller } = setup(5000)
    const jumped = vi.fn()
    controller.onJump(jumped)
    controller.frame(5100)
    expect(jumped).not.toHaveBeenCalled()
    controller.seek(9000)
    controller.stepBars(1)
    controller.skip(1000)
    expect(jumped).toHaveBeenCalledTimes(3)
    controller.loop.setFromBars(grid, 8, 15)
    controller.frame(32_990)
    controller.frame(33_010)
    expect(jumped).toHaveBeenCalledTimes(4)
  })

  it('stops notifying a listener that unsubscribed', () => {
    const { controller } = setup()
    const jumped = vi.fn()
    controller.onJump(jumped)()
    controller.seek(1000)
    expect(jumped).not.toHaveBeenCalled()
  })
})

describe('a loop that reaches the end of the song', () => {
  function atEnd(state: string, startMs = 59_000) {
    const position = { ms: startMs }
    const seek = vi.fn((ms: number) => {
      position.ms = ms
    })
    const play = vi.fn(async () => {})
    const transport = { seek, currentTimeMs: () => position.ms, state, play }
    const controller = new PracticeController(transport as never, grid)
    return { controller, seek, play, position, transport }
  }

  it('goes back to A and keeps playing when the element has already ended', () => {
    const { controller, seek, play, position, transport } = atEnd('playing')
    controller.loop.setFromBars(grid, 27, 29) // 55 000 .. 60 000, the end of the song
    controller.frame(59_000)
    transport.state = 'ended'
    position.ms = 60_000
    expect(controller.frame(60_000)).toBe(true)
    expect(seek).toHaveBeenCalledWith(55_000)
    expect(play).toHaveBeenCalledTimes(1)
  })

  it('also wraps when the audio is a few ms shorter than the loop end', () => {
    const { controller, seek, play, position, transport } = atEnd('playing')
    controller.loop.setFromBars(grid, 27, 29)
    controller.frame(59_000)
    transport.state = 'ended'
    position.ms = 59_990
    expect(controller.frame(59_990)).toBe(true)
    expect(seek).toHaveBeenCalledWith(55_000)
    expect(play).toHaveBeenCalledTimes(1)
  })

  it('leaves a song that ends outside a mid-song loop alone', () => {
    const { controller, seek, play, position, transport } = atEnd('playing')
    controller.loop.setFromBars(grid, 8, 15)
    controller.seek(40_000)
    seek.mockClear()
    transport.state = 'ended'
    position.ms = 60_000
    expect(controller.frame(60_000)).toBe(false)
    expect(seek).not.toHaveBeenCalled()
    expect(play).not.toHaveBeenCalled()
  })
});
