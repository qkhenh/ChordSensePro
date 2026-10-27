import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createMemoryRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import { describe, expect, it } from 'vitest'
import { createApiClient, memoryStorage } from '../api/http'
import type { ChartSummary } from '../api/types'
import { buildRoutes } from '../routes'
import { apiError, authResponse, http, HttpResponse, server } from '../test/server'

const API = '*/api/v1'

function summary(n: number, overrides: Partial<ChartSummary> = {}): ChartSummary {
  return {
    id: `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`,
    title: `Song ${n}`,
    artist: `Artist ${n}`,
    source_kind: 'file',
    playback_kind: 'audio',
    duration_ms: 125_000,
    tempo_bpm: 120.4,
    key: 'C major',
    is_customized: false,
    version: 1,
    updated_at: '2026-10-26T10:00:00Z',
    ...overrides,
  }
}

function renderLibrary() {
  server.use(http.post(`${API}/auth/refresh`, () => HttpResponse.json(authResponse())))
  const client = createApiClient({ storage: memoryStorage('refresh-stored') })
  const router = createMemoryRouter(buildRoutes(client), { initialEntries: ['/library'] })
  render(<RouterProvider router={router} />)
  return router
}

describe('library page', () => {
  it('shows a skeleton while loading, then a card per chart with its facts', async () => {
    server.use(
      http.get(`${API}/charts`, () =>
        HttpResponse.json({
          items: [
            summary(1),
            summary(2, { source_kind: 'url', title: 'Linked', artist: null }),
            summary(3, { source_kind: 'demo', title: 'Demo tune' }),
          ],
          next_cursor: null,
        }),
      ),
    )
    renderLibrary()
    expect(await screen.findByTestId('library-skeleton')).toBeInTheDocument()
    const first = await screen.findByRole('link', { name: /Song 1/ })
    expect(first).toHaveAttribute('href', '/charts/00000000-0000-4000-8000-000000000001')
    expect(within(first).getByText('Artist 1')).toBeInTheDocument()
    expect(within(first).getByText('Tải lên')).toBeInTheDocument()
    expect(within(first).getByText('2:05')).toBeInTheDocument()
    expect(within(first).getByText(/120 BPM/)).toBeInTheDocument()
    expect(within(first).getByText('C major')).toBeInTheDocument()
    expect(within(screen.getByRole('link', { name: /Linked/ })).getByText('Liên kết')).toBeInTheDocument()
    expect(within(screen.getByRole('link', { name: /Demo tune/ })).getByText('Demo')).toBeInTheDocument()
    expect(screen.queryByTestId('library-skeleton')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Tải thêm' })).not.toBeInTheDocument()
  })

  it('loads the next page on demand and hides the button after the last one', async () => {
    const cursors: (string | null)[] = []
    server.use(
      http.get(`${API}/charts`, ({ request }) => {
        const cursor = new URL(request.url).searchParams.get('cursor')
        cursors.push(cursor)
        return cursor
          ? HttpResponse.json({ items: [summary(2)], next_cursor: null })
          : HttpResponse.json({ items: [summary(1)], next_cursor: 'c1' })
      }),
    )
    renderLibrary()
    await screen.findByRole('link', { name: /Song 1/ })
    await userEvent.click(screen.getByRole('button', { name: 'Tải thêm' }))
    expect(await screen.findByRole('link', { name: /Song 2/ })).toBeInTheDocument()
    expect(cursors).toEqual([null, 'c1'])
    expect(screen.queryByRole('button', { name: 'Tải thêm' })).not.toBeInTheDocument()
  })

  it('offers the workspace when the library is empty', async () => {
    server.use(http.get(`${API}/charts`, () => HttpResponse.json({ items: [], next_cursor: null })))
    renderLibrary()
    const link = await screen.findByRole('link', { name: /Phân tích bài đầu tiên/ })
    expect(link).toHaveAttribute('href', '/workspace')
  })

  it('tells the user when the list cannot be loaded', async () => {
    server.use(http.get(`${API}/charts`, () => apiError(500, 'internal_error')))
    renderLibrary()
    expect(await screen.findByText(/Có lỗi xảy ra/)).toBeInTheDocument()
  })
})
