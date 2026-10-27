import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { I18nProvider } from '../i18n/I18nProvider'
import { ChordModeToggle } from './ChordModeToggle'
import { useChordMode } from './useChordMode'

const KEY = 'chordsense.chordMode'

function Harness() {
  const [mode, setMode] = useChordMode()
  return (
    <I18nProvider>
      <ChordModeToggle mode={mode} onChange={setMode} />
      <output data-testid="mode">{mode}</output>
    </I18nProvider>
  )
}

beforeEach(() => localStorage.clear())
afterEach(() => vi.restoreAllMocks())

describe('chord mode', () => {
  it('starts on Precise and marks the active choice', () => {
    render(<Harness />)
    expect(screen.getByTestId('mode')).toHaveTextContent('precise')
    expect(screen.getByRole('button', { name: 'Chi tiết' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Cơ bản' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('switches on click and remembers the choice after a reload', async () => {
    const first = render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Cơ bản' }))
    expect(screen.getByTestId('mode')).toHaveTextContent('basic')
    expect(localStorage.getItem(KEY)).toBe('basic')
    first.unmount()
    render(<Harness />)
    expect(screen.getByTestId('mode')).toHaveTextContent('basic')
  })

  it('ignores a stored value it does not know', () => {
    localStorage.setItem(KEY, 'jazz')
    render(<Harness />)
    expect(screen.getByTestId('mode')).toHaveTextContent('precise')
  })

  it('still works when storage is blocked', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    render(<Harness />)
    await userEvent.click(screen.getByRole('button', { name: 'Cơ bản' }))
    expect(screen.getByTestId('mode')).toHaveTextContent('basic')
  })
})
