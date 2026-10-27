import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { PracticeController } from './controller'
import { useShortcuts } from './useShortcuts'

function setup() {
  const controller = { stepBars: vi.fn(), skip: vi.fn() }
  const toggle = vi.fn()
  function Harness() {
    useShortcuts(controller as unknown as PracticeController, toggle)
    return (
      <div>
        <input aria-label="bpm" />
        <textarea aria-label="notes" />
        <select aria-label="pick">
          <option>a</option>
        </select>
        <div aria-label="editable" contentEditable suppressContentEditableWarning tabIndex={0} />
        <button type="button">press</button>
        <a href="#x">link</a>
      </div>
    )
  }
  const view = render(<Harness />)
  return { controller, toggle, view }
}

describe('keyboard shortcuts', () => {
  it('Space plays and pauses', () => {
    const { toggle } = setup()
    fireEvent.keyDown(document.body, { key: ' ', code: 'Space' })
    expect(toggle).toHaveBeenCalledTimes(1)
  })

  it('arrows move one bar, Shift+arrows move five seconds', () => {
    const { controller } = setup()
    fireEvent.keyDown(document.body, { key: 'ArrowRight' })
    fireEvent.keyDown(document.body, { key: 'ArrowLeft' })
    fireEvent.keyDown(document.body, { key: 'ArrowRight', shiftKey: true })
    fireEvent.keyDown(document.body, { key: 'ArrowLeft', shiftKey: true })
    expect(controller.stepBars.mock.calls).toEqual([[1], [-1]])
    expect(controller.skip.mock.calls).toEqual([[5000], [-5000]])
  })

  it.each([['bpm'], ['notes'], ['pick'], ['editable']])('does nothing while typing in %s', (label) => {
    const { controller, toggle } = setup()
    const field = screen.getByLabelText(label)
    for (const key of [' ', 'ArrowLeft', 'ArrowRight']) fireEvent.keyDown(field, { key })
    expect(toggle).not.toHaveBeenCalled()
    expect(controller.stepBars).not.toHaveBeenCalled()
  })

  it('leaves Space to a focused button or link, so it does not toggle twice', () => {
    const { toggle } = setup()
    fireEvent.keyDown(screen.getByRole('button', { name: 'press' }), { key: ' ' })
    fireEvent.keyDown(screen.getByRole('link', { name: 'link' }), { key: ' ' })
    expect(toggle).not.toHaveBeenCalled()
  })

  it('keeps arrows working when a button has focus', () => {
    const { controller } = setup()
    fireEvent.keyDown(screen.getByRole('button', { name: 'press' }), { key: 'ArrowRight' })
    expect(controller.stepBars).toHaveBeenCalledWith(1)
  })

  it('ignores browser shortcuts such as Ctrl+Left', () => {
    const { controller } = setup()
    fireEvent.keyDown(document.body, { key: 'ArrowLeft', ctrlKey: true })
    fireEvent.keyDown(document.body, { key: 'ArrowLeft', metaKey: true })
    fireEvent.keyDown(document.body, { key: 'ArrowLeft', altKey: true })
    expect(controller.stepBars).not.toHaveBeenCalled()
  })

  it('stops listening when the page is left', () => {
    const { toggle, view } = setup()
    view.unmount()
    fireEvent.keyDown(document.body, { key: ' ' })
    expect(toggle).not.toHaveBeenCalled()
  })
})
