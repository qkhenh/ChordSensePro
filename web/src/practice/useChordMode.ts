import { useState } from 'react'
import type { ChordMode } from '../format/chord'

const KEY = 'chordsense.chordMode'

function read(): ChordMode {
  try {
    const stored = localStorage.getItem(KEY)
    return stored === 'basic' || stored === 'precise' ? stored : 'precise'
  } catch {
    return 'precise' // storage blocked: the choice just will not survive a reload
  }
}

/** Basic / Precise (FR-14), remembered on this device. Purely a display choice: no request is made. */
export function useChordMode(): [ChordMode, (mode: ChordMode) => void] {
  const [mode, setMode] = useState<ChordMode>(read)
  const change = (next: ChordMode) => {
    setMode(next)
    try {
      localStorage.setItem(KEY, next)
    } catch {
      /* see above */
    }
  }
  return [mode, change]
}
