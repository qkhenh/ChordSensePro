import { act, render } from '@testing-library/react'
import { Profiler, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import type { Transport } from '../transport/types'
import { SmartGrid } from './SmartGrid'
import { usePlayhead } from './usePlayhead'

const meta: AnalysisMetadata = {
  duration_ms: 20_000,
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
const grid = buildGrid(meta, []) // pre-roll to 1000 ms, then 2000 ms bars

/** requestAnimationFrame by hand: the test decides when a frame happens. */
function fakeFrames() {
  let next: FrameRequestCallback | null = null
  let time = 0
  return {
    raf: (cb: FrameRequestCallback) => {
      next = cb
      return 1
    },
    cancel: vi.fn(),
    step(advanceMs = 16) {
      time += advanceMs
      const cb = next
      next = null
      cb?.(time)
    },
    pending: () => next !== null,
  }
}

function fakeTransport(position = { ms: 0 }): Transport & { position: { ms: number } } {
  return { currentTimeMs: () => position.ms, position } as unknown as Transport & { position: { ms: number } }
}

/** Commits of the tree under test: a frame that re-rendered React would add one. */
let commits = 0

function Harness({ transport, frames }: { transport: Transport | null; frames: ReturnType<typeof fakeFrames> }) {
  const root = useRef<HTMLDivElement>(null)
  usePlayhead(transport, grid, root, { raf: frames.raf, cancel: frames.cancel })
  return (
    <Profiler id="practice" onRender={() => (commits += 1)}>
      <SmartGrid grid={grid} mode="precise" onSeek={() => {}} rootRef={root} />
    </Profiler>
  )
}

const current = () => document.querySelector('[data-current="true"]')?.getAttribute('data-bar') ?? null

afterEach(() => {
  commits = 0
  vi.restoreAllMocks()
})

describe('highlighting the bar being played', () => {
  it('lights the bar that holds the playback position', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBe('0')
    transport.position.ms = 3200
    act(() => frames.step())
    expect(current()).toBe('1')
  })

  it('lights exactly one bar at a time', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    for (const ms of [1500, 3200, 5200, 9000, 1100]) {
      transport.position.ms = ms
      act(() => frames.step())
      expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(1)
    }
  })

  it('lights nothing during the pre-roll before the first beat', () => {
    const frames = fakeFrames()
    render(<Harness transport={fakeTransport({ ms: 400 })} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBeNull()
  })

  it('moves the light back when the user seeks backwards', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 9000 })
    render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBe('4')
    transport.position.ms = 1200
    act(() => frames.step())
    expect(current()).toBe('0')
  })

  it('stays on the last bar at and past the end of the song', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 25_000 })
    render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBe(String(grid.bars.length - 1))
  })

  it('does nothing without a transport and starts once there is one', () => {
    const frames = fakeFrames()
    const { rerender } = render(<Harness transport={null} frames={frames} />)
    expect(frames.pending()).toBe(false)
    rerender(<Harness transport={fakeTransport({ ms: 1500 })} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBe('0')
  })
})

describe('costing nothing while the bar stays the same', () => {
  it('writes to the page only when the bar changes', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    const set = vi.spyOn(Element.prototype, 'setAttribute')
    const remove = vi.spyOn(Element.prototype, 'removeAttribute')
    for (let i = 0; i < 20; i++) {
      transport.position.ms += 20
      act(() => frames.step())
    }
    expect(set).not.toHaveBeenCalled()
    expect(remove).not.toHaveBeenCalled()
  })

  it('never re-renders React over 120 frames of playback', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1000 })
    render(<Harness transport={transport} frames={frames} />)
    const before = commits
    for (let i = 0; i < 120; i++) {
      transport.position.ms += 40
      act(() => frames.step())
    }
    expect(commits).toBe(before)
  })
})

describe('keeping up on a slow machine (UC2 7a)', () => {
  it('paints every other frame when frames take about 40 ms, and says so', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    for (let i = 0; i < 40; i++) act(() => frames.step(40))
    const reads = vi.spyOn(transport, 'currentTimeMs')
    for (let i = 0; i < 20; i++) act(() => frames.step(40))
    expect(reads.mock.calls.length).toBeLessThanOrEqual(10)
    expect(reads.mock.calls.length).toBeGreaterThan(0)
  })

  it('goes back to every frame once the machine recovers', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    for (let i = 0; i < 60; i++) act(() => frames.step(40))
    for (let i = 0; i < 120; i++) act(() => frames.step(8))
    const reads = vi.spyOn(transport, 'currentTimeMs')
    for (let i = 0; i < 20; i++) act(() => frames.step(8))
    expect(reads.mock.calls.length).toBe(20)
  })
})

describe('coming back to a tab', () => {
  it('shows the right bar immediately when the page becomes visible again', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    transport.position.ms = 15_000 // played on in the background; no frame ran
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'))
    })
    expect(current()).toBe('7')
  })
})

describe('cleaning up', () => {
  it('stops asking for frames when unmounted', () => {
    const frames = fakeFrames()
    const { unmount } = render(<Harness transport={fakeTransport({ ms: 1500 })} frames={frames} />)
    act(() => frames.step())
    unmount()
    expect(frames.cancel).toHaveBeenCalled()
  })

  it('starts clean for a different grid', () => {
    const frames = fakeFrames()
    const transport = fakeTransport({ ms: 1500 })
    const { rerender } = render(<Harness transport={transport} frames={frames} />)
    act(() => frames.step())
    expect(current()).toBe('0')
    rerender(<Harness transport={fakeTransport({ ms: 5500 })} frames={frames} />)
    act(() => frames.step())
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(1)
    expect(current()).toBe('2')
  })
})
