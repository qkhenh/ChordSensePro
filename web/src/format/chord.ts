import type { ChordEvent } from '../api/types'

export type ChordMode = 'basic' | 'precise'

/**
 * How a chord is written on the grid. Precise is the analysed symbol (`G9sus4`); Basic keeps only
 * the root, minor/diminished and the bass of an inversion (`G`), so a beginner sees the triad.
 */
export function formatChord(event: Pick<ChordEvent, 'chord' | 'symbol'>, mode: ChordMode): string {
  if (mode === 'precise') return event.symbol
  const { root, quality, bass } = event.chord
  const triad = quality === 'minor' ? 'm' : quality === 'diminished' ? 'dim' : ''
  return `${root}${triad}${bass ? `/${bass}` : ''}`
}
