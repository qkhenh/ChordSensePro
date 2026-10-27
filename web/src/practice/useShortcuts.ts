import { useEffect, useRef } from 'react'
import type { PracticeController } from './controller'

const SKIP_MS = 5000
const TYPING = 'input, textarea, select, [contenteditable]'
const ACTIVATES_ON_SPACE = 'button, a[href], [role="button"]'

/**
 * Space plays and pauses; ← / → move one bar and Shift+← / → five seconds (D5). Nothing happens
 * while the person types, and Space is left to a focused button or link, which already react to
 * it themselves.
 */
export function useShortcuts(controller: PracticeController | null, toggle: () => void): void {
  const latest = useRef({ controller, toggle })
  useEffect(() => {
    latest.current = { controller, toggle }
  })

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const target = event.target instanceof Element ? event.target : null
      if (target?.closest(TYPING)) return
      const { controller: practice, toggle: playPause } = latest.current

      if (event.key === ' ') {
        if (target?.closest(ACTIVATES_ON_SPACE)) return
        event.preventDefault() // the page would scroll
        playPause()
      } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        if (!practice) return
        const direction = event.key === 'ArrowRight' ? 1 : -1
        event.preventDefault()
        if (event.shiftKey) practice.skip(direction * SKIP_MS)
        else practice.stepBars(direction)
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [])
}
