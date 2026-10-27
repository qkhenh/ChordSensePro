import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { I18nProvider } from '../i18n/I18nProvider'
import { LoopControls } from './LoopControls'

const show = (range: { from: number; to: number } | null, onClear = vi.fn()) => {
  render(
    <I18nProvider>
      <LoopControls range={range} onClear={onClear} />
    </I18nProvider>,
  )
  return onClear
}

describe('loop controls', () => {
  it('names the looped bars counting from 1 and clears on request', async () => {
    const onClear = show({ from: 8, to: 15 })
    expect(screen.getByText(/ô nhịp 9–16/i)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Xoá loop' }))
    expect(onClear).toHaveBeenCalledTimes(1)
  })

  it('explains how to choose a loop when there is none', () => {
    show(null)
    expect(screen.queryByRole('button', { name: 'Xoá loop' })).not.toBeInTheDocument()
    expect(screen.getByText(/kéo/i)).toBeInTheDocument()
  })
})
