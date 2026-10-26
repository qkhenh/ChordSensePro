import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createRef } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata, ChordEvent } from '../api/types'
import { formatChord } from '../format/chord'
import { buildGrid } from '../grid/grid'
import { I18nProvider } from '../i18n/I18nProvider'
import { SmartGrid } from './SmartGrid'

vi.mock('../format/chord', async (original) => {
  const real = await original<typeof import('../format/chord')>()
  return { ...real, formatChord: vi.fn(real.formatChord) }
})

const meta = (duration = 16_000): AnalysisMetadata => ({
  duration_ms: duration,
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
})

const chord = (id: string, start: number, end: number, symbol: string, quality: 'major' | 'minor' | 'suspended', extension: ChordEvent['chord']['extension'] = 'none'): ChordEvent => ({
  event_id: id,
  start_ms: start,
  end_ms: end,
  chord: { root: 'G', quality, extension, bass: null },
  symbol,
  confidence: null,
  is_user_edited: false,
})

const events = [
  chord('a', 0, 2000, 'Gm', 'minor'),
  chord('b', 2000, 3000, 'G9sus4', 'suspended', 'nat9'),
  chord('c', 3000, 6000, 'G', 'major'),
]

function show(props: Partial<Parameters<typeof SmartGrid>[0]> = {}) {
  const grid = props.grid ?? buildGrid(meta(), events)
  const onSeek = props.onSeek ?? vi.fn()
  const view = render(<SmartGrid grid={grid} mode="precise" onSeek={onSeek} rootRef={createRef()} {...props} />)
  return { grid, onSeek, ...view }
}

beforeEach(() => {
  vi.mocked(formatChord).mockClear() // braces: a returned function would be run as cleanup
})

describe('layout', () => {
  it('has one button per bar, numbered from 1', () => {
    show()
    const cells = screen.getAllByRole('button')
    expect(cells).toHaveLength(8) // 16 s at 2 s a bar
    expect(cells[0]).toHaveAccessibleName(/^Ô nhịp 1\b/)
    expect(cells[7]).toHaveAccessibleName(/^Ô nhịp 8\b/)
  })

  it('names the grid for screen readers', () => {
    show()
    expect(screen.getByRole('group', { name: 'Lưới hợp âm' })).toBeInTheDocument()
  })

  it('gives every cell a data-bar index the playhead can find', () => {
    show()
    const indexes = screen.getAllByRole('button').map((b) => b.getAttribute('data-bar'))
    expect(indexes).toEqual(['0', '1', '2', '3', '4', '5', '6', '7'])
  })

  it('starts with no bar highlighted', () => {
    show()
    expect(document.querySelector('[data-current="true"]')).toBeNull()
  })
})

describe('chords', () => {
  it('writes the chords of a bar in time order', () => {
    show()
    const first = screen.getAllByRole('button')[0]
    expect(within(first).getByText('Gm')).toBeInTheDocument()
    const second = screen.getAllByRole('button')[1]
    expect(within(second).getByText('G9sus4')).toBeInTheDocument()
    expect(within(second).getByText('G')).toBeInTheDocument()
  })

  it('shows the simplified name in basic mode', () => {
    show({ mode: 'basic' })
    expect(screen.queryByText('G9sus4')).not.toBeInTheDocument()
    expect(within(screen.getAllByRole('button')[1]).getAllByText('G').length).toBeGreaterThan(0)
  })

  it('dims a chord that is only carried over from the bar before', () => {
    show()
    const carried = within(screen.getAllByRole('button')[2]).getByText('G') // 3000-6000 spans bars 1 and 2
    expect(carried).toHaveAttribute('data-carried', 'true')
    const starting = within(screen.getAllByRole('button')[1]).getAllByText('G')[0]
    expect(starting).not.toHaveAttribute('data-carried')
  })

  it('leaves bars without a chord empty but still clickable', () => {
    show()
    const empty = screen.getAllByRole('button')[6]
    expect(empty).toHaveTextContent('7')
    expect(empty).not.toBeDisabled()
  })

  it('mentions the chords in the cell\'s accessible name', () => {
    show()
    expect(screen.getAllByRole('button')[0]).toHaveAccessibleName('Ô nhịp 1: Gm')
  })
})

describe('seeking', () => {
  it('a click on a bar asks to seek to where that bar begins', async () => {
    const { onSeek } = show()
    await userEvent.click(screen.getAllByRole('button')[4])
    expect(onSeek).toHaveBeenCalledWith(8000)
  })

  it('the keyboard works too', async () => {
    const { onSeek } = show()
    screen.getAllByRole('button')[2].focus()
    await userEvent.keyboard('{Enter}')
    expect(onSeek).toHaveBeenCalledWith(4000)
  })
})

describe('staying cheap', () => {
  it('does not write the chord names again when the parent redraws with the same props', () => {
    const grid = buildGrid(meta(), events)
    const onSeek = vi.fn()
    const ref = createRef<HTMLDivElement>()
    const { rerender } = render(<SmartGrid grid={grid} mode="precise" onSeek={onSeek} rootRef={ref} />)
    const calls = vi.mocked(formatChord).mock.calls.length
    rerender(<SmartGrid grid={grid} mode="precise" onSeek={onSeek} rootRef={ref} />)
    expect(vi.mocked(formatChord).mock.calls.length).toBe(calls)
  })

  it('writes the names again only when the mode changes', () => {
    const grid = buildGrid(meta(), events)
    const onSeek = vi.fn()
    const ref = createRef<HTMLDivElement>()
    const { rerender } = render(<SmartGrid grid={grid} mode="precise" onSeek={onSeek} rootRef={ref} />)
    const calls = vi.mocked(formatChord).mock.calls.length
    rerender(<SmartGrid grid={grid} mode="basic" onSeek={onSeek} rootRef={ref} />)
    expect(vi.mocked(formatChord).mock.calls.length).toBeGreaterThan(calls)
  })

  it('draws a 400-bar chart in full (its speed is measured in a real browser: e2e/sync.perf)', () => {
    const grid = buildGrid(meta(800_000), [])
    expect(grid.bars.length).toBeGreaterThanOrEqual(400)
    show({ grid })
    expect(screen.getAllByRole('button').length).toBe(grid.bars.length)
  })
})

describe('language', () => {
  it('speaks English when asked', () => {
    localStorage.setItem('chordsense.lang', 'en')
    render(
      <I18nProvider>
        <SmartGrid grid={buildGrid(meta(), events)} mode="precise" onSeek={vi.fn()} rootRef={createRef()} />
      </I18nProvider>,
    )
    expect(screen.getByRole('group', { name: 'Chord grid' })).toBeInTheDocument()
    expect(screen.getAllByRole('button')[0]).toHaveAccessibleName('Bar 1: Gm')
  })
})
