import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { AnalysisMetadata } from '../api/types'
import { buildGrid } from '../grid/grid'
import { I18nProvider } from '../i18n/I18nProvider'
import { SmartGrid } from './SmartGrid'

const meta: AnalysisMetadata = {
  duration_ms: 40_000,
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
const grid = buildGrid(meta, []) // 20 bars of 2000 ms

const bar = (n: number) => screen.getByRole('button', { name: new RegExp(`^Ô nhịp ${n}(:|$)`) })

function show(props: Partial<Parameters<typeof SmartGrid>[0]> = {}) {
  const onSeek = vi.fn()
  const onSelect = vi.fn()
  render(
    <I18nProvider>
      <SmartGrid grid={grid} mode="precise" rootRef={createRef()} onSeek={onSeek} onSelect={onSelect} {...props} />
    </I18nProvider>,
  )
  return { onSeek, onSelect }
}

describe('loop band', () => {
  it('marks exactly the bars of the loop', () => {
    show({ loop: { from: 8, to: 15 } })
    const marked = screen.getAllByRole('button').filter((b) => b.dataset.loop === 'true')
    expect(marked.map((b) => b.dataset.bar)).toEqual(['8', '9', '10', '11', '12', '13', '14', '15'])
  })

  it('marks nothing without a loop', () => {
    show({ loop: null })
    expect(screen.getAllByRole('button').some((b) => b.dataset.loop === 'true')).toBe(false)
  })
})

describe('choosing a loop', () => {
  it('selects the bars between the two ends of a drag, and does not seek', async () => {
    const user = userEvent.setup()
    const { onSeek, onSelect } = show()
    await user.pointer([{ keys: '[MouseLeft>]', target: bar(9) }, { target: bar(16) }, { keys: '[/MouseLeft]', target: bar(16) }])
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(8, 15)
    expect(onSeek).not.toHaveBeenCalled()
  })

  it('reports a backwards drag as it happened, the loop sorts it out', async () => {
    const user = userEvent.setup()
    const { onSelect } = show()
    await user.pointer([{ keys: '[MouseLeft>]', target: bar(16) }, { target: bar(9) }, { keys: '[/MouseLeft]', target: bar(9) }])
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(15, 8)
  })

  it('seeks, and selects nothing, on a plain click', async () => {
    const user = userEvent.setup()
    const { onSeek, onSelect } = show()
    await user.click(bar(9))
    expect(onSeek).toHaveBeenCalledExactlyOnceWith(16_000)
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('selects with two Shift+clicks for keyboard and touch users', async () => {
    const user = userEvent.setup()
    const { onSeek, onSelect } = show()
    await user.keyboard('{Shift>}')
    await user.click(bar(9))
    expect(onSelect).not.toHaveBeenCalled()
    await user.click(bar(16))
    await user.keyboard('{/Shift}')
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(8, 15)
    expect(onSeek).not.toHaveBeenCalled()
  })

  it('starts a new pair after a completed Shift+click selection', async () => {
    const user = userEvent.setup()
    const { onSelect } = show()
    await user.keyboard('{Shift>}')
    await user.click(bar(2))
    await user.click(bar(4))
    await user.click(bar(10))
    expect(onSelect).toHaveBeenCalledTimes(1)
    await user.click(bar(12))
    expect(onSelect).toHaveBeenLastCalledWith(9, 11)
    await user.keyboard('{/Shift}')
  })
})
