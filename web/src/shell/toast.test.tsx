import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ToastSpec } from '../api/errors'
import { useToast } from './context'
import { ToastProvider } from './toast'

function Trigger({ spec = { kind: 'error', message: 'Boom' } }: { spec?: ToastSpec }) {
  const { show } = useToast()
  return (
    <button type="button" onClick={() => show(spec)}>
      go
    </button>
  )
}

afterEach(() => vi.useRealTimers())

describe('toast', () => {
  it('announces an error as an alert and a warning as a polite status', async () => {
    const user = userEvent.setup()
    render(
      <ToastProvider>
        <Trigger />
        <Trigger spec={{ kind: 'warning', message: 'Careful' }} />
      </ToastProvider>,
    )
    const [error, warning] = screen.getAllByRole('button', { name: 'go' })
    await user.click(error)
    await user.click(warning)
    expect(screen.getByRole('alert')).toHaveTextContent('Boom')
    expect(screen.getByRole('status')).toHaveTextContent('Careful')
  })

  it('dismisses itself after a few seconds', () => {
    vi.useFakeTimers()
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    )
    act(() => screen.getByRole('button', { name: 'go' }).click())
    expect(screen.getByRole('alert')).toBeInTheDocument()
    act(() => void vi.advanceTimersByTime(6000))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('can be dismissed by hand', async () => {
    const user = userEvent.setup()
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    )
    await user.click(screen.getByRole('button', { name: 'go' }))
    await user.click(screen.getByRole('button', { name: 'Đóng' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('refuses to be used outside its provider', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<Trigger />)).toThrow(/ToastProvider/)
    spy.mockRestore()
  })
})
