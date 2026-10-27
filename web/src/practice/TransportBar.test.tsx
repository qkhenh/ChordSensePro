import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Profiler } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { I18nProvider } from '../i18n/I18nProvider'
import type { Transport } from '../transport/types'
import { TransportBar } from './TransportBar'

function fakeTransport(canRate = true) {
  const position = { ms: 0 }
  const transport = {
    currentTimeMs: () => position.ms,
    capabilities: { rate: canRate },
  } as unknown as Transport
  return { transport, position }
}

function show(props: Partial<Parameters<typeof TransportBar>[0]> = {}) {
  const { transport, position } = fakeTransport()
  const onToggle = vi.fn()
  const onRate = vi.fn()
  let commits = 0
  const view = render(
    <I18nProvider>
      <Profiler id="bar" onRender={() => (commits += 1)}>
        <TransportBar
          transport={transport}
          playing={false}
          canPlay
          durationMs={125_000}
          rate={1}
          onToggle={onToggle}
          onRate={onRate}
          {...props}
        />
      </Profiler>
    </I18nProvider>,
  )
  return { position, onToggle, onRate, commits: () => commits, ...view }
}

describe('play button', () => {
  it('offers Play, or Pause while playing, and calls back on click', async () => {
    const { onToggle, rerender } = show()
    await userEvent.click(screen.getByRole('button', { name: 'Phát' }))
    expect(onToggle).toHaveBeenCalledTimes(1)
    rerender(
      <I18nProvider>
        <TransportBar transport={null} playing canPlay durationMs={1000} rate={1} onToggle={onToggle} onRate={vi.fn()} />
      </I18nProvider>,
    )
    expect(screen.getByRole('button', { name: 'Tạm dừng' })).toBeInTheDocument()
  })

  it('is disabled until the song can be played', () => {
    show({ canPlay: false })
    expect(screen.getByRole('button', { name: 'Phát' })).toBeDisabled()
  })
})

describe('clock', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('shows position and length, refreshed four times a second without re-rendering', () => {
    const { position, commits } = show()
    const clock = screen.getByTestId('clock')
    expect(clock).toHaveTextContent('0:00 / 2:05')
    const before = commits()
    position.ms = 65_900
    act(() => {
      vi.advanceTimersByTime(250)
    })
    expect(clock).toHaveTextContent('1:05 / 2:05')
    act(() => {
      vi.advanceTimersByTime(2000)
    })
    expect(commits()).toBe(before)
  })

  it('writes to the page at most four times a second', () => {
    const { position } = show()
    const clock = screen.getByTestId('clock')
    const write = vi.spyOn(clock, 'textContent', 'set')
    for (let i = 1; i <= 8; i++) {
      position.ms = i * 1000
      act(() => {
        vi.advanceTimersByTime(125)
      })
    }
    expect(write.mock.calls.length).toBeLessThanOrEqual(4)
  })

  it('stops when the bar goes away', () => {
    const { unmount } = show()
    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe('speed', () => {
  it('lists 0.5×, 0.75× and 1× with the current one pressed', async () => {
    const { onRate } = show({ rate: 0.75 })
    expect(screen.getByRole('button', { name: '0.75×' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: '1×' })).toHaveAttribute('aria-pressed', 'false')
    await userEvent.click(screen.getByRole('button', { name: '0.5×' }))
    expect(onRate).toHaveBeenCalledWith(0.5)
  })

  it('is switched off, with the reason, for a source that cannot change speed', () => {
    show({ transport: fakeTransport(false).transport })
    for (const name of ['0.5×', '0.75×', '1×']) expect(screen.getByRole('button', { name })).toBeDisabled()
    expect(screen.getByText(/không hỗ trợ đổi tốc độ/i)).toBeInTheDocument()
  })
})
