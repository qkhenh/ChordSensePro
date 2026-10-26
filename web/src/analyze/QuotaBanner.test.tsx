import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ApiError } from '../api/http'
import { I18nProvider } from '../i18n/I18nProvider'
import { QuotaBanner } from './QuotaBanner'

function quotaError(limit: number | null = 5, resetsAt: string | null = '2026-10-27T00:00:00Z') {
  const error = new ApiError(429, 'quota_exceeded', 'Daily quota used')
  error.limit = limit
  error.resetsAt = resetsAt ? new Date(resetsAt) : null
  return error
}

describe('QuotaBanner', () => {
  it('says how many analyses were allowed and when new ones arrive', () => {
    render(<QuotaBanner error={quotaError()} />)
    const banner = screen.getByRole('alert')
    expect(banner).toHaveTextContent('Bạn đã dùng hết 5 lượt phân tích hôm nay')
    expect(banner).toHaveTextContent(/làm mới|có lúc/i)
    expect(banner.textContent).toContain(
      new Date('2026-10-27T00:00:00Z').toLocaleString('vi', { dateStyle: 'short', timeStyle: 'short' }),
    )
  })

  it('offers the Pro upgrade (EF2)', () => {
    render(<QuotaBanner error={quotaError()} />)
    expect(screen.getByRole('link', { name: 'Nâng cấp Pro' })).toBeInTheDocument()
  })

  it('still reads well when the server sent no numbers', () => {
    render(<QuotaBanner error={quotaError(null, null)} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Bạn đã hết lượt phân tích hôm nay')
    expect(screen.getByRole('alert')).not.toHaveTextContent('null')
  })

  it('speaks English when asked', () => {
    localStorage.setItem('chordsense.lang', 'en')
    render(
      <I18nProvider>
        <QuotaBanner error={quotaError()} />
      </I18nProvider>,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('You have used all 5 analyses for today')
    expect(screen.getByRole('link', { name: 'Upgrade to Pro' })).toBeInTheDocument()
  })
})
