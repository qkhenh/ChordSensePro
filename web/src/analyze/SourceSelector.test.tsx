import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MAX_FILE_BYTES } from './validate'
import { SourceSelector } from './SourceSelector'

function setup(props: Partial<Parameters<typeof SourceSelector>[0]> = {}) {
  const onSubmit = vi.fn()
  render(<SourceSelector onSubmit={onSubmit} {...props} />)
  // applyAccept off: the component must reject bad files itself, not rely on the picker.
  return { onSubmit, user: userEvent.setup({ applyAccept: false }) }
}

const tab = (name: string) => screen.getByRole('tab', { name })
const mp3 = () => new File([new Uint8Array(2048)], 'Autumn Leaves.mp3', { type: 'audio/mpeg' })

function huge() {
  const file = new File([], 'big.mp3', { type: 'audio/mpeg' })
  Object.defineProperty(file, 'size', { value: MAX_FILE_BYTES + 1 })
  return file
}

describe('tabs', () => {
  it('shows Upload, Record and Link with Upload selected', () => {
    setup()
    expect(screen.getAllByRole('tab').map((t) => t.textContent)).toEqual(['Tải lên', 'Ghi âm', 'Liên kết'])
    expect(tab('Tải lên')).toHaveAttribute('aria-selected', 'true')
  })

  it('shows Record but keeps it disabled with an explanation', () => {
    setup()
    const record = tab('Ghi âm')
    expect(record).toBeDisabled()
    expect(record).toHaveAccessibleDescription(/sớm có mặt/i)
  })

  it('switches tabs with a click', async () => {
    const { user } = setup()
    await user.click(tab('Liên kết'))
    expect(tab('Liên kết')).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByLabelText(/dán liên kết/i)).toBeInTheDocument()
  })

  it('moves with the arrow keys, skipping the disabled tab', async () => {
    const { user } = setup()
    tab('Tải lên').focus()
    await user.keyboard('{ArrowRight}')
    expect(tab('Liên kết')).toHaveAttribute('aria-selected', 'true')
    expect(tab('Liên kết')).toHaveFocus()
    await user.keyboard('{ArrowRight}')
    expect(tab('Tải lên')).toHaveAttribute('aria-selected', 'true')
    await user.keyboard('{ArrowLeft}')
    expect(tab('Liên kết')).toHaveAttribute('aria-selected', 'true')
  })
})

describe('upload', () => {
  it('lists the chosen file and submits it on request', async () => {
    const { user, onSubmit } = setup()
    const file = mp3()
    await user.upload(screen.getByLabelText(/chọn tệp/i), file)
    expect(screen.getByText('Autumn Leaves.mp3')).toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
    expect(onSubmit).toHaveBeenCalledWith({ kind: 'file', file })
  })

  it('accepts a file dropped on the zone', () => {
    const { onSubmit } = setup()
    fireEvent.drop(screen.getByTestId('dropzone'), { dataTransfer: { files: [mp3()] } })
    expect(screen.getByText('Autumn Leaves.mp3')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
    expect(onSubmit).toHaveBeenCalledOnce()
  })

  it('explains an oversized file and offers nothing to submit', async () => {
    const { user, onSubmit } = setup()
    await user.upload(screen.getByLabelText(/chọn tệp/i), huge())
    expect(screen.getByRole('alert')).toHaveTextContent('Tệp lớn hơn 100 MB')
    expect(screen.queryByRole('button', { name: 'Phân tích bài hát' })).not.toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('explains a file of the wrong type', async () => {
    const { user } = setup()
    await user.upload(screen.getByLabelText(/chọn tệp/i), new File(['x'], 'notes.txt', { type: 'text/plain' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/MP3, WAV, FLAC, M4A hoặc OGG/)
  })

  it('forgets the error once a good file is chosen', async () => {
    const { user } = setup()
    await user.upload(screen.getByLabelText(/chọn tệp/i), huge())
    await user.upload(screen.getByLabelText(/chọn tệp/i), mp3())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('link', () => {
  it('submits a valid link', async () => {
    const { user, onSubmit } = setup()
    await user.click(tab('Liên kết'))
    await user.type(screen.getByLabelText(/dán liên kết/i), 'https://youtu.be/dQw4w9WgXcQ')
    await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
    expect(onSubmit).toHaveBeenCalledWith({ kind: 'url', url: 'https://youtu.be/dQw4w9WgXcQ' })
  })

  it('submits with the Enter key', async () => {
    const { user, onSubmit } = setup()
    await user.click(tab('Liên kết'))
    await user.type(screen.getByLabelText(/dán liên kết/i), 'https://youtu.be/dQw4w9WgXcQ{Enter}')
    expect(onSubmit).toHaveBeenCalledOnce()
  })

  it('gives the specific reason for a bad link and suggests uploading instead (AF1)', async () => {
    const { user, onSubmit } = setup()
    await user.click(tab('Liên kết'))
    await user.type(screen.getByLabelText(/dán liên kết/i), 'ftp://example.com/a.mp3')
    await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Chỉ nhận liên kết http hoặc https')
    expect(onSubmit).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Tải tệp lên thay thế' }))
    expect(tab('Tải lên')).toHaveAttribute('aria-selected', 'true')
  })

  it('names an internal address as such', async () => {
    const { user } = setup()
    await user.click(tab('Liên kết'))
    await user.type(screen.getByLabelText(/dán liên kết/i), 'http://169.254.169.254/latest/meta-data/')
    await user.click(screen.getByRole('button', { name: 'Phân tích bài hát' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/nội bộ/i)
  })
})

describe('state from the parent', () => {
  it('shows the server error', () => {
    setup({ serverError: 'Email này đã được đăng ký' })
    expect(screen.getByRole('alert')).toHaveTextContent('Email này đã được đăng ký')
  })

  it('disables submitting while busy', async () => {
    const { user } = setup({ busy: true })
    await user.upload(screen.getByLabelText(/chọn tệp/i), mp3())
    expect(screen.getByRole('button', { name: /đang gửi/i })).toBeDisabled()
  })

  it('speaks English when asked', async () => {
    const { I18nProvider } = await import('../i18n/I18nProvider')
    localStorage.setItem('chordsense.lang', 'en')
    render(
      <I18nProvider>
        <SourceSelector onSubmit={vi.fn()} />
      </I18nProvider>,
    )
    expect(screen.getAllByRole('tab').map((t) => t.textContent)).toEqual(['Upload', 'Record', 'Link'])
  })
})
