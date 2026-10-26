import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ErrorCode, JobProgress } from '../api/types'
import { I18nProvider } from '../i18n/I18nProvider'
import { JobTracker } from './JobTracker'

const job = (over: Partial<JobProgress> = {}): JobProgress => ({
  job_id: 'j1',
  state: 'processing',
  stage: 'separating',
  progress: 0.45,
  error: null,
  result_chart_id: null,
  ...over,
})

const failed = (code: ErrorCode, retryable = false): JobProgress =>
  job({ state: 'failed', error: { code, message: 'server text', retryable } })

function setup(value: JobProgress, props: Partial<Parameters<typeof JobTracker>[0]> = {}) {
  const onRetry = vi.fn()
  const onDismiss = vi.fn()
  render(<JobTracker job={value} onRetry={onRetry} onDismiss={onDismiss} {...props} />)
  return { onRetry, onDismiss, user: userEvent.setup() }
}

const steps = () => screen.getAllByRole('listitem')
const stateOf = (item: HTMLElement) => item.getAttribute('data-state')

describe('stepper', () => {
  it('marks finished steps, the running one and the ones still to come', () => {
    setup(job())
    expect(steps().map((s) => s.textContent)).toEqual([
      'Chuẩn hoá âm thanh',
      'Tách nhạc cụ',
      'Dò nhịp',
      'Nhận diện hợp âm',
    ])
    expect(steps().map(stateOf)).toEqual(['done', 'active', 'pending', 'pending'])
    expect(steps()[1]).toHaveAttribute('aria-current', 'step')
  })

  it('adds the fetching step only for links', () => {
    setup(job({ stage: 'fetching', progress: 0.05 }), { withFetch: true })
    expect(steps()[0]).toHaveTextContent('Tải từ liên kết')
    expect(steps().map(stateOf)).toEqual(['active', 'pending', 'pending', 'pending', 'pending'])
  })

  it('waits in the queue before the first step starts', () => {
    setup(job({ state: 'queued', stage: null, progress: 0 }))
    expect(screen.getByText('Đang xếp hàng')).toBeInTheDocument()
    expect(steps().map(stateOf)).toEqual(['pending', 'pending', 'pending', 'pending'])
  })

  it('shows the whole run as complete when done', () => {
    setup(job({ state: 'done', stage: 'recognizing', progress: 1, result_chart_id: 'c1' }))
    expect(steps().map(stateOf)).toEqual(['done', 'done', 'done', 'done'])
    expect(screen.getByText('Hoàn tất')).toBeInTheDocument()
  })

  it('reports the percentage as a progress bar', () => {
    setup(job({ progress: 0.45 }))
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '45')
  })

  it('offers to stop watching while the job runs', async () => {
    const { user, onDismiss } = setup(job())
    await user.click(screen.getByRole('button', { name: 'Dừng theo dõi' }))
    expect(onDismiss).toHaveBeenCalledOnce()
    expect(screen.queryByRole('button', { name: 'Thử lại' })).not.toBeInTheDocument()
  })
})

describe('failures', () => {
  it.each<[ErrorCode, RegExp]>([
    ['unsupported_format', /không phải âm thanh hợp lệ/i],
    ['file_too_large', /lớn hơn 100 MB/],
    ['duration_too_long', /dài hơn 10 phút/],
    ['url_unresolvable', /không lấy được âm thanh/i],
    ['ssrf_blocked', /an toàn/i],
    ['quota_exceeded', /hết lượt/i],
    ['analysis_timeout', /quá nhiều thời gian/i],
    ['ai_failure', /không phân tích được/i],
  ])('explains %s in plain Vietnamese', (code, text) => {
    setup(failed(code))
    expect(screen.getByRole('alert')).toHaveTextContent(text)
    expect(screen.getByRole('alert')).not.toHaveTextContent('server text')
  })

  it('offers a retry only when the failure is retryable and a retry exists', async () => {
    const { user, onRetry } = setup(failed('ai_failure', true))
    await user.click(screen.getByRole('button', { name: 'Thử lại' }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('has no retry for a failure that retrying cannot fix', () => {
    setup(failed('unsupported_format', false))
    expect(screen.queryByRole('button', { name: 'Thử lại' })).not.toBeInTheDocument()
  })

  it('has no retry when the source is no longer known (after a page reload)', () => {
    render(<JobTracker job={failed('ai_failure', true)} onDismiss={vi.fn()} />)
    expect(screen.queryByRole('button', { name: 'Thử lại' })).not.toBeInTheDocument()
  })

  it('closes the failed run with a plain "close"', async () => {
    const { user, onDismiss } = setup(failed('ai_failure', true))
    await user.click(screen.getByRole('button', { name: 'Đóng' }))
    expect(onDismiss).toHaveBeenCalledOnce()
  })

  it('still says something useful for an error code it does not know', () => {
    setup(job({ state: 'failed', error: { code: 'brand_new' as ErrorCode, message: 'x', retryable: false } }))
    expect(within(screen.getByRole('alert')).getByText(/có lỗi/i)).toBeInTheDocument()
  })
})

describe('language', () => {
  it('speaks English when asked', () => {
    localStorage.setItem('chordsense.lang', 'en')
    render(
      <I18nProvider>
        <JobTracker job={job()} onDismiss={vi.fn()} />
      </I18nProvider>,
    )
    expect(steps()[1]).toHaveTextContent('Separating instruments')
    expect(screen.getByRole('button', { name: 'Stop watching' })).toBeInTheDocument()
  })
})
