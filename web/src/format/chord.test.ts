import { describe, expect, it } from 'vitest'
import type { ChordLabel } from '../api/types'
import { formatChord } from './chord'

const ROOTS: ChordLabel['root'][] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
const QUALITIES: ChordLabel['quality'][] = ['major', 'minor', 'dominant', 'diminished', 'suspended']
const EXTENSIONS: ChordLabel['extension'][] = [
  'none', 'b5', '#5', 'maj7', 'min7', 'dom7', 'add9', 'b9', 'nat9', '#9', 'nat11', '#11', 'nat13', 'b13', 'alt',
]

const event = (symbol: string, root: ChordLabel['root'], quality: ChordLabel['quality'], extension: ChordLabel['extension'] = 'none', bass: ChordLabel['bass'] = null) => ({
  symbol,
  chord: { root, quality, extension, bass },
})

describe('precise', () => {
  it('shows the symbol exactly as analysed', () => {
    expect(formatChord(event('G9sus4', 'G', 'suspended', 'nat9'), 'precise')).toBe('G9sus4')
    expect(formatChord(event('Bm7b5', 'B', 'diminished', 'min7'), 'precise')).toBe('Bm7b5')
  })
})

describe('basic', () => {
  it.each([
    ['Cmaj9', 'C', 'major', 'nat9', null, 'C'],
    ['Cmaj7', 'C', 'major', 'maj7', null, 'C'],
    ['Am7', 'A', 'minor', 'min7', null, 'Am'],
    ['G7', 'G', 'dominant', 'dom7', null, 'G'],
    ['Bm7b5', 'B', 'diminished', 'min7', null, 'Bdim'],
    ['Caug', 'C', 'major', '#5', null, 'C'],
    ['G9sus4', 'G', 'suspended', 'nat9', null, 'G'],
    ['Dsus2', 'D', 'suspended', 'none', null, 'D'],
    ['Em', 'E', 'minor', 'none', null, 'Em'],
    ['F#dim', 'F#', 'diminished', 'none', null, 'F#dim'],
    ['C/E', 'C', 'major', 'none', 'E', 'C/E'],
    ['Am7/G', 'A', 'minor', 'min7', 'G', 'Am/G'],
    ['G7/B', 'G', 'dominant', 'dom7', 'B', 'G/B'],
    ['Cm9/Bb', 'C', 'minor', 'nat9', 'A#', 'Cm/A#'],
  ] as const)('%s reads as %s', (symbol, root, quality, extension, bass, expected) => {
    expect(formatChord(event(symbol, root, quality, extension, bass), 'basic')).toBe(expected)
  })

  it('looks only at root, minor/diminished and bass: the extension never changes the result', () => {
    for (const root of ROOTS) {
      for (const quality of QUALITIES) {
        for (const bass of [null, root === 'C' ? 'G' : 'C'] as const) {
          const shown = new Set(EXTENSIONS.map((ext) => formatChord(event('x', root, quality, ext, bass), 'basic')))
          expect(shown.size).toBe(1)
        }
      }
    }
  })

  it('always produces a plain triad name, whatever the chord', () => {
    const plain = /^[A-G]#?(m|dim)?(\/[A-G]#?)?$/
    for (const root of ROOTS) {
      for (const quality of QUALITIES) {
        for (const extension of EXTENSIONS) {
          expect(formatChord(event('x', root, quality, extension, null), 'basic')).toMatch(plain)
        }
      }
    }
  })

  it('never loses the bass note of an inversion', () => {
    for (const quality of QUALITIES) {
      expect(formatChord(event('x', 'C', quality, 'maj7', 'E'), 'basic')).toMatch(/\/E$/)
    }
  })
})
