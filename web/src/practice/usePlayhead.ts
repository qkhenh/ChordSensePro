import { useEffect, type RefObject } from 'react'
import type { Grid } from '../grid/grid'
import type { Transport } from '../transport/types'

export interface PlayheadOptions {
  raf?: (callback: FrameRequestCallback) => number
  cancel?: (handle: number) => void
}

const WINDOW = 30 // frames the speed estimate looks back over
const MIN_SAMPLES = 5
const SLOW_MS = 34 // slower than ~30 fps: paint every 2nd frame
const VERY_SLOW_MS = 60 // slower than ~16 fps: paint every 4th frame
const RECOVERED_MS = 24 // back above ~40 fps: paint every frame again

const defaultRaf = (callback: FrameRequestCallback) => requestAnimationFrame(callback)
const defaultCancel = (handle: number) => cancelAnimationFrame(handle)

/**
 * Lights the bar being played by writing `data-current` on its element, once per change.
 * Everything happens in a requestAnimationFrame loop outside React: no state, no re-render,
 * and nothing at all is written while the bar stays the same.
 *
 * On a slow machine it paints less often (UC2 7a) so audio and input keep priority, and when the
 * tab comes back from the background it shows the right bar at once instead of waiting a frame.
 */
export function usePlayhead(
  transport: Transport | null,
  grid: Grid,
  root: RefObject<HTMLElement | null>,
  options: PlayheadOptions = {},
): void {
  const raf = options.raf ?? defaultRaf
  const cancel = options.cancel ?? defaultCancel

  useEffect(() => {
    const container = root.current
    if (!transport || !container) return
    for (const lit of container.querySelectorAll('[data-current]')) lit.removeAttribute('data-current')

    let shown = -1
    let handle = 0
    let last: number | null = null
    let stride = 1
    let skipped = 0
    const gaps: number[] = []

    const paint = () => {
      const index = grid.barAtMs(transport.currentTimeMs())
      if (index === shown) return
      if (shown >= 0) container.querySelector(`[data-bar="${shown}"]`)?.removeAttribute('data-current')
      if (index >= 0) container.querySelector(`[data-bar="${index}"]`)?.setAttribute('data-current', 'true')
      shown = index
    }

    const adapt = (now: number) => {
      if (last !== null) {
        gaps.push(now - last)
        if (gaps.length > WINDOW) gaps.shift()
      }
      last = now
      if (gaps.length < MIN_SAMPLES) return
      const average = gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length
      if (average > VERY_SLOW_MS) stride = 4
      else if (average > SLOW_MS) stride = 2
      else if (average < RECOVERED_MS) stride = 1
    }

    const frame: FrameRequestCallback = (now) => {
      handle = raf(frame)
      adapt(now)
      skipped += 1
      if (skipped < stride) return
      skipped = 0
      paint()
    }

    const onVisible = () => paint()
    document.addEventListener('visibilitychange', onVisible)
    handle = raf(frame)
    return () => {
      document.removeEventListener('visibilitychange', onVisible)
      cancel(handle)
    }
  }, [transport, grid, root, raf, cancel])
}
