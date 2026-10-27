import { render } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata, ChordEvent } from '../api/types'
import { buildGrid } from '../grid/grid'
import { I18nProvider } from '../i18n/I18nProvider'
import { dictionaries } from '../i18n/messages'
import { SmartGrid } from './SmartGrid'

const meta: AnalysisMetadata = {
  duration_ms: 16_000,
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

const chord = (
  id: string,
  start: number,
  symbol: string,
  root: ChordEvent['chord']['root'],
  quality: ChordEvent['chord']['quality'],
  extension: ChordEvent['chord']['extension'] = 'none',
): ChordEvent => ({
  event_id: id,
  start_ms: start,
  end_ms: start + 2000,
  chord: { root, quality, extension, bass: null },
  symbol,
  confidence: null,
  is_user_edited: false,
})

// bars 0, 2 and 4 are plain triads (the same in both modes); bars 1 and 3 differ
const events = [
  chord('a', 0, 'C', 'C', 'major'),
  chord('b', 2000, 'Dm7', 'D', 'minor', 'min7'),
  chord('c', 4000, 'F', 'F', 'major'),
  chord('d', 6000, 'G7', 'G', 'dominant', 'dom7'),
  chord('e', 8000, 'Am', 'A', 'minor'),
]

afterEach(() => vi.restoreAllMocks())

describe('switching between Basic and Precise', () => {
  it('redraws only the bars whose name changes', () => {
    const label = vi.spyOn(dictionaries.vi.practice, 'bar')
    const grid = buildGrid(meta, events)
    const rootRef = createRef<HTMLDivElement>()
    const ui = (mode: 'basic' | 'precise') => (
      <I18nProvider>
        <SmartGrid grid={grid} mode={mode} rootRef={rootRef} onSeek={() => {}} />
      </I18nProvider>
    )
    const { rerender } = render(ui('precise'))
    label.mockClear()
    rerender(ui('basic'))
    const redrawn = new Set(label.mock.calls.map(([n]) => n))
    expect(redrawn).toEqual(new Set([2, 4])) // Dm7 -> Dm and G7 -> G, numbered from 1
  })
})
