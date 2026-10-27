import { act, render } from '@testing-library/react'
import { useRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import type { Transport } from '../transport/types'
import { usePlayhead } from './usePlayhead'

const meta: AnalysisMetadata = {
  duration_ms: 20_000,
  key: { tonic: 'C', mode: 'major', confidence: 0.9 },
  beat_grid: {
    tempo_bpm: 120,
    tempo_confidence: 0.9,
    time_signature: { numerator: 4, denominator: 4 },
    beats_per_bar: 4,
    first_beat_offset_ms: 0,
    downbeats_ms: null,
  },
  model_version: 'm',
  pipeline_version: 'p',
}
const grid = buildGrid(meta, [])

function Harness({ transport, onFrame, raf }: { transport: Transport; onFrame: (ms: number) => void; raf: PlayheadRaf }) {
  const root = useRef<HTMLDivElement>(null)
  usePlayhead(transport, grid, root, { raf: raf.raf, cancel: () => {}, onFrame })
  return <div ref={root} />
}
type PlayheadRaf = { raf: (cb: FrameRequestCallback) => number; step(dt: number): void }

function frames(): PlayheadRaf {
  let next: FrameRequestCallback | null = null
  let time = 0
  return {
    raf: (cb) => {
      next = cb
      return 1
    },
    step(dt) {
      time += dt
      const cb = next
      next = null
      cb?.(time)
    },
  }
}

describe('onFrame', () => {
  it('reports the position on every frame, even those the painting skips on a slow machine', () => {
    const position = { ms: 100 }
    const transport = { currentTimeMs: () => position.ms } as unknown as Transport
    const onFrame = vi.fn()
    const raf = frames()
    render(<Harness transport={transport} onFrame={onFrame} raf={raf} />)
    for (let i = 0; i < 12; i++) {
      position.ms += 70
      act(() => raf.step(70)) // ~14 fps: the playhead falls back to painting every 4th frame
    }
    expect(onFrame).toHaveBeenCalledTimes(12)
    expect(onFrame).toHaveBeenLastCalledWith(100 + 70 * 12)
  })
})
